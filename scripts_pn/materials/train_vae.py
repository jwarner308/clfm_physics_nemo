import torch
from torch.utils.data import DataLoader
from pathlib import Path
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.utils import save_checkpoint

from clfm_pn.problems.materials import MaterialsTrain, MaterialsVal, MaterialsLoss
from clfm_pn.nn.vae import FunctionalVAE
from clfm_pn.nn.fully_connected_nets import PNFCEncoder, PNFCTrunk, EnhancedBranchNetwork
from clfm_pn.utils import kl_divergence, reparameterize


@hydra.main(version_base="1.3", config_path="conf", config_name="config_vae")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.training.seed)

    # Data
    train_data = MaterialsTrain(cfg.data.N_data)
    val_data = MaterialsVal(train_data.X_u, num_samples=cfg.data.num_test_data)

    batch_size = min(cfg.data.N_data, cfg.training.batch_size)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=128,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    loss_fn = MaterialsLoss(
        train_data,
        num_colloc=cfg.data.num_colloc,
        lbc_weight=cfg.loss.lbc_weight,
        rbc_weight=cfg.loss.rbc_weight,
    )

    # Model — materials uses EnhancedBranchNetwork
    encoder = PNFCEncoder(
        input_size=train_data.num_sensors,
        hidden_size=cfg.arch.h_encoder,
        output_size=cfg.arch.latent_dim,
        num_hidden_layers=cfg.arch.nhl_encoder,
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
        num_outputs=train_data.num_fields,
        num_hidden_layers=cfg.arch.nhl_trunk,
    )
    vae = FunctionalVAE(
        encoder=encoder,
        branch=branch,
        trunk=trunk,
        num_fields=train_data.num_fields,
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
                    val_metrics = loss_fn.validate(
                        vae,
                        val_batch[0].to(device),
                        val_batch[1].to(device),
                        val_batch[2].to(device),
                    )
                    break  # single batch validation

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
