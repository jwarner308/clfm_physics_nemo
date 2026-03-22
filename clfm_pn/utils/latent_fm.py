import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchdyn.core import NeuralODE

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from clfm_pn.nn.fully_connected_nets import PNFlowModel
from clfm_pn.utils import reparameterize


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format.

    https://github.com/atong01/conditional-flow-matching/tree/main
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.repeat(x.shape[0]).unsqueeze(-1)
        return self.model(torch.cat([x, t], dim=-1))


def train_lfm(
    vae,
    train_data,
    latent_dim=64,
    hidden_layer_size=128,
    num_hidden_layers=3,
    num_epochs=1000,
    sigma_min=0.01,
    batch_size=256,
    lr=0.001,
    device="cuda",
    num_workers=0,
    gradient_clip_norm=None,
    save_path=None,
):
    """Train a latent flow model using PhysicsNemo training utilities.

    Uses DistributedManager for device management and LaunchLogger for logging.
    """
    dist = DistributedManager()
    if device is None:
        device = dist.device

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    flow = PNFlowModel(
        latent_dim=latent_dim,
        hidden_size=hidden_layer_size,
        num_hidden_layers=num_hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    loss_hist = []

    LaunchLogger.initialize()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        with LaunchLogger("train", epoch=epoch) as log:
            for batch in train_loader:
                optimizer.zero_grad()
                u, x = batch
                u, x = u.to(device), x.to(device)

                with torch.no_grad():
                    z1 = reparameterize(*vae.encode(u))

                z0 = torch.randn_like(z1)
                t = torch.rand(z1.shape[0], device=device).unsqueeze(-1)
                mu_t = t * z1 + (1.0 - t) * z0
                zt = mu_t + sigma_min * torch.randn_like(mu_t)
                vt = flow(torch.cat([zt, t], dim=-1))
                loss = F.mse_loss(vt, z1 - z0)
                loss.backward()

                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        flow.parameters(), gradient_clip_norm
                    )

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            log.log_minibatch({"loss": avg_loss})
            loss_hist.append(avg_loss)

    if save_path is not None:
        save_checkpoint(
            save_path,
            models=flow,
            optimizer=optimizer,
            epoch=num_epochs,
        )

    return flow, loss_hist


def sample_lfm(
    num_samples, x_grid, flow_model, vae, num_time_steps=100, device="cuda"
):
    """Generate samples using latent flow matching.

    Uses a Neural ODE to simulate the learned flow in latent space,
    then decodes the results through the VAE decoder.
    """
    x_gen = x_grid[None, :, :].repeat(num_samples, 1, 1).to(device)

    node = NeuralODE(
        torch_wrapper(flow_model),
        solver="rk4",
        sensitivity="adjoint",
    ).to(device)

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(num_samples, flow_model.latent_dim, device=device),
            t_span=torch.linspace(0.0, 1.0, num_time_steps, device=device),
        )
    z = traj[-1, :, :]

    with torch.no_grad():
        y_gen = vae.decode(z, x_gen)

    return y_gen
