from pathlib import Path
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import load_checkpoint

from clfm_pn.problems.wind import WindDataset
from clfm_pn.nn.unet1d import Encoder1d
from clfm_pn.nn.vae import FunctionalVAE
from clfm_pn.nn.fully_connected_nets import PNFCTrunk, EnhancedBranchNetwork
from clfm_pn.utils.latent_fm import train_lfm

TRAIN_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_train_data.hdf5"


@hydra.main(version_base="1.3", config_path="conf", config_name="config_lfm")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    result_dir = Path(cfg.output.result_dir)

    train_data = WindDataset(
        str(TRAIN_DATA_FILE), cfg.data.sparse_sensors, cfg.data.num_sensors
    )

    # Reconstruct and load VAE
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
        num_outputs=1,
        num_hidden_layers=cfg.arch.nhl_trunk,
    )
    vae = FunctionalVAE(
        encoder=encoder,
        branch=branch,
        trunk=trunk,
        num_fields=1,
        grid=train_data.grid,
    ).to(device)

    vae_ckpt = cfg.output.get("vae_checkpoint", None) or str(
        result_dir / "vae_best.pt"
    )
    load_checkpoint(vae_ckpt, models=vae, device=device)
    vae.eval()

    flow, loss_hist = train_lfm(
        vae,
        train_data,
        latent_dim=cfg.arch.latent_dim,
        hidden_layer_size=cfg.flow_matching.hidden_size,
        num_hidden_layers=cfg.flow_matching.num_hidden_layers,
        num_epochs=cfg.flow_matching.epochs,
        sigma_min=cfg.flow_matching.sigma_min,
        batch_size=cfg.flow_matching.batch_size,
        lr=cfg.flow_matching.lr,
        device=device,
        save_path=str(result_dir / "flow_model.pt"),
    )

    checkpoint = {
        "architecture": {
            "latent_dim": cfg.arch.latent_dim,
            "hidden_size": cfg.flow_matching.hidden_size,
            "num_hidden_layers": cfg.flow_matching.num_hidden_layers,
        },
        "state_dict": flow.state_dict(),
    }
    torch.save(checkpoint, result_dir / "fm.pth")


if __name__ == "__main__":
    main()
