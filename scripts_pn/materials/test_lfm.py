import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import load_checkpoint

from clfm_pn.problems.materials import MaterialsTrain, MaterialsVal
from clfm_pn.nn.fully_connected_nets import PNFCEncoder, PNFCTrunk, PNFlowModel, EnhancedBranchNetwork
from clfm_pn.nn.vae import FunctionalVAE
from clfm_pn.utils.latent_fm import sample_lfm
from clfm_pn.utils.utils import dense_grid_eval


@hydra.main(version_base="1.3", config_path="conf", config_name="config_lfm")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    NUM_SAMPLES = 500
    NUM_TIME_STEPS = 100

    result_dir = Path(cfg.output.result_dir)

    train_data = MaterialsTrain(cfg.data.N_data)

    # Reconstruct and load VAE
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

    vae_ckpt = cfg.output.get("vae_checkpoint", None) or str(
        result_dir / "vae_best.pt"
    )
    load_checkpoint(vae_ckpt, models=vae, device=device)
    vae.eval()

    # Load flow model
    fm_checkpoint = torch.load(result_dir / "fm.pth", map_location=device)
    flow_model = PNFlowModel(**fm_checkpoint["architecture"]).to(device)
    flow_model.load_state_dict(fm_checkpoint["state_dict"])
    flow_model.eval()

    # Generate samples on a dense grid
    x_grid = dense_grid_eval(train_data.grid, (25, 25)).to(device)
    y_gen = sample_lfm(NUM_SAMPLES, x_grid, flow_model, vae, NUM_TIME_STEPS, device)

    ux_gen, uy_gen, e_gen = y_gen.chunk(3, dim=2)
    print(f"Generated {NUM_SAMPLES} samples")
    print(f"ux - mean: {ux_gen.mean():.4f}, std: {ux_gen.std():.4f}")
    print(f"uy - mean: {uy_gen.mean():.4f}, std: {uy_gen.std():.4f}")
    print(f"E  - mean: {e_gen.mean():.4f}, std: {e_gen.std():.4f}")

    torch.save(
        {"y_gen": y_gen.cpu(), "x_grid": x_grid.cpu()},
        result_dir / "generated_samples.pt",
    )


if __name__ == "__main__":
    main()
