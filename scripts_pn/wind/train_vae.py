from pathlib import Path
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils import save_checkpoint

from clfm_pn.problems.wind import WindDataset, WindLoss
from clfm_pn.nn.unet1d import Encoder1d
from clfm_pn.nn.vae import FunctionalVAE
from clfm_pn.nn.fully_connected_nets import PNFCTrunk, EnhancedBranchNetwork
from clfm_pn.utils import kl_divergence, reparameterize

TRAIN_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_train_data.hdf5"
TEST_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_test_data.hdf5"


@hydra.main(version_base="1.3", config_path="conf", config_name="config_vae")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.training.seed)

    # Data
    train_data = WindDataset(
        str(TRAIN_DATA_FILE), cfg.data.sparse_sensors, cfg.data.num_sensors
    )
    val_data = WindDataset(str(TEST_DATA_FILE))

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    loss_fn = WindLoss(cfg.loss.num_colloc, cfg.loss.T, train_data)

    # Wind uses UNet1D encoder (not FullyConnected)
    encoder = Encoder1d(
        in_channels=len(train_data.sensor_idx), latent_dim=cfg.arch.latent_dim
    )
    branch = EnhancedBranchNetwork(
        latent_dim=cfg.arch.latent_dim,
        hidden_dim=cfg.arch.h_branch,
        output_dim=cfg.arch.p_deeponet,
        num_blocks=cfg.arch.nhl_branch,
    )
    trunk = PNFCTrunk(
        input_size=train_data.grid.ndim,
        hidden_size=cfg.arch.h_trunk,
        output_size=cfg.arch.p_deeponet,
        num_outputs=1,  # wind has 1 output field
        num_hidden_layers=cfg.arch.nhl_trunk,
    )
    vae = FunctionalVAE(
        encoder=encoder,
        branch=branch,
        trunk=trunk,
        num_fields=1,
        grid=train_data.grid,
    ).to(device)
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.training.lr)

    # Training loop
    result_dir = Path(cfg.output.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    LaunchLogger.initialize()

    for epoch in range(cfg.training.epochs):
        vae.train()
        epoch_loss = 0.0
        num_batches = 0

        with LaunchLogger("train", epoch=epoch) as log:
            for batch in train_loader:
                optimizer.zero_grad()
                u, x = batch
                u, x = u.to(device), x.to(device)

                mu, logvar = vae.encode(u)
                z = reparameterize(mu, logvar)

                rec_loss = loss_fn.reconstruction(vae, z, x, u)
                res_loss, metrics = loss_fn.residual(vae, z)
                kld_loss = kl_divergence(mu, logvar)

                loss = (
                    rec_loss
                    + cfg.loss.kld_weight * kld_loss
                    + cfg.loss.res_weight * res_loss
                )
                loss.backward()

                if cfg.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        vae.parameters(), cfg.training.grad_clip
                    )

                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            log.log_minibatch(
                {
                    "total_loss": avg_loss,
                    "reconstruction": rec_loss.item(),
                    "kld": kld_loss.item(),
                    **metrics,
                }
            )

        # Validation
        if (epoch + 1) % 100 == 0:
            vae.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_u, val_x = val_batch
                    val_u, val_x = val_u.to(device), val_x.to(device)
                    val_metrics = loss_fn.validate(vae, val_u, val_x)
                    break

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    result_dir / "vae_best.pt",
                    models=vae,
                    optimizer=optimizer,
                    epoch=epoch,
                )

    save_checkpoint(
        result_dir / "vae_final.pt",
        models=vae,
        optimizer=optimizer,
        epoch=cfg.training.epochs,
    )


if __name__ == "__main__":
    main()
