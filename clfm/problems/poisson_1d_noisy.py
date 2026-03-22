import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

from clfm.problems.poisson_1d import (
    Poisson1DDataset,
    Poisson1DLoss,
    v_func,
    f_func,
)
from clfm.utils.utils import grad


class NoisyPoisson1DDataset(Poisson1DDataset):

    def __init__(
        self,
        N: int = None,
        num_sensors: int = None,
        x_min: float = 0.0,
        x_max: float = torch.pi,
        A_mean: float = 0.2,
        A_std: float = 0.05,
        noise_std: float = 0.01,
        samples_dir: str = None,
    ):
        super().__init__(
            N=N,
            num_sensors=num_sensors,
            x_min=x_min,
            x_max=x_max,
            A_mean=A_mean,
            A_std=A_std,
            samples_dir=samples_dir,
        )
        self.noise_std = noise_std
        self.u_samples_clean = self.u_samples.clone()
        noise = torch.randn_like(self.u_samples) * noise_std
        self.u_samples_noisy = self.u_samples + noise
        # __getitem__ returns noisy data (via self.u_samples)
        self.u_samples = self.u_samples_noisy

    def __getitem__(self, idx: int):
        sample = (self.u_samples_noisy[idx], self.x_sensor)
        return sample


class NoisyPoisson1DLoss(Poisson1DLoss):

    def __init__(
        self,
        num_colloc: int,
        rec_mode: str = "standard",
        noise_std: float = 0.01,
        beta: float = 5.0,
        noise_std_init: float = 0.05,
        num_sensors: int = 25,
    ):
        super().__init__(num_colloc=num_colloc)
        self.rec_mode = rec_mode
        self.noise_std = noise_std
        self.beta = beta
        self.num_sensors = num_sensors
        self.dataset = None  # set externally for validation access to clean data

        if rec_mode == "learned_noise":
            self.log_sigma_n = nn.Parameter(
                torch.tensor(noise_std_init).log()
            )

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        f = vae.decode(z, x)
        u, _ = f.chunk(2, dim=2)
        u = u.squeeze(2)

        if self.rec_mode == "standard":
            return torch.mean(torch.square(u - u_true))

        elif self.rec_mode == "softplus":
            per_sample_sq_error = torch.sum(torch.square(u - u_true), dim=1)
            noise_floor = self.num_sensors * self.noise_std ** 2
            return torch.mean(
                nn.functional.softplus(
                    self.beta * (per_sample_sq_error - noise_floor)
                )
            ) / self.beta

        elif self.rec_mode == "learned_noise":
            sigma_n_sq = torch.exp(2.0 * self.log_sigma_n)
            mse = torch.mean(torch.square(u - u_true))
            m = self.num_sensors
            return 0.5 * mse / sigma_n_sq + 0.5 * m * 2.0 * self.log_sigma_n

        else:
            raise ValueError(f"Unknown rec_mode: {self.rec_mode}")

    def validate(self, vae, *args):
        if self.dataset is None:
            return {}

        metrics = {}
        u_clean = self.dataset.u_samples_clean
        x_sensor = self.dataset.x_sensor

        with torch.no_grad():
            from clfm.utils import reparameterize
            u_noisy = self.dataset.u_samples_noisy.to(vae.device)
            x_in = x_sensor.unsqueeze(0).expand(u_noisy.shape[0], -1, -1).to(vae.device)
            mu, logvar = vae.encode(u_noisy)
            z = reparameterize(mu, logvar)
            f = vae.decode(z, x_in)
            u_hat, _ = f.chunk(2, dim=2)
            u_hat = u_hat.squeeze(2).cpu()
            metrics["val_u_mse_vs_clean"] = torch.mean(
                torch.square(u_hat - u_clean)
            ).item()

        if self.rec_mode == "learned_noise":
            metrics["val_learned_sigma"] = torch.exp(self.log_sigma_n).item()

        return metrics
