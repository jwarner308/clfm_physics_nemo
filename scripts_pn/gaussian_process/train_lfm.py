import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint

from clfm_pn.problems.gaussian_process import GPDataset
from clfm_pn.nn.vae import FunctionalVAE
from clfm_pn.nn.fully_connected_nets import PNFCBranch, PNFCEncoder, PNFCTrunk
from clfm_pn.utils.latent_fm import train_lfm


@hydra.main(version_base="1.3", config_path="conf", config_name="config_lfm")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    result_dir = Path(cfg.output.result_dir)

    train_data = GPDataset(
        samples_dir=str(result_dir),
        N=cfg.data.N_data,
        num_sensors=cfg.data.num_sensors,
        cov_len=cfg.data.gp_cov_len,
        variance=cfg.data.gp_var,
        mean="LINEAR",
    )

    # Reconstruct and load VAE
    encoder = PNFCEncoder(
        input_size=cfg.data.num_sensors,
        hidden_size=cfg.arch.h_encoder,
        output_size=cfg.arch.latent_dim,
        num_hidden_layers=cfg.arch.nhl_encoder,
    )
    branch = PNFCBranch(
        input_size=cfg.arch.latent_dim,
        hidden_size=cfg.arch.h_branch,
        output_size=cfg.arch.p_deeponet,
        num_hidden_layers=cfg.arch.nhl_branch,
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
