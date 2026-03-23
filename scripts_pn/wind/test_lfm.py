from pathlib import Path
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import load_checkpoint

from clfm_pn.problems.wind import WindDataset, dense_eval
from clfm_pn.nn.unet1d import Encoder1d
from clfm_pn.nn.fully_connected_nets import PNFCTrunk, PNFlowModel, EnhancedBranchNetwork
from clfm_pn.nn.vae import FunctionalVAE
from clfm_pn.utils.latent_fm import sample_lfm

TEST_DATA_FILE = Path(__file__).parent / "../../data/wind/wind_test_data.hdf5"


@hydra.main(version_base="1.3", config_path="conf", config_name="config_lfm")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    NUM_SAMPLES = 100
    NUM_TIME_STEPS = 100

    result_dir = Path(cfg.output.result_dir)

    test_data = WindDataset(str(TEST_DATA_FILE))

    # Reconstruct and load VAE
    encoder = Encoder1d(
        in_channels=len(test_data.sensor_idx), latent_dim=cfg.arch.latent_dim
    )
    branch = EnhancedBranchNetwork(
        latent_dim=cfg.arch.latent_dim,
        hidden_dim=cfg.arch.h_branch,
        output_dim=cfg.arch.p_deeponet,
        num_blocks=cfg.arch.nhl_branch,
    )
    trunk = PNFCTrunk(
        input_size=test_data.grid.ndim,
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
        grid=test_data.grid,
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

    # Generate samples
    x_grid = test_data.points.to(device)
    y_gen = sample_lfm(NUM_SAMPLES, x_grid, flow_model, vae, NUM_TIME_STEPS, device)

    print(f"Generated {NUM_SAMPLES} wind samples")
    print(f"Output shape: {y_gen.shape}")
    print(f"Mean: {y_gen.mean():.4f}, Std: {y_gen.std():.4f}")

    torch.save(
        {"y_gen": y_gen.cpu(), "x_grid": x_grid.cpu()},
        result_dir / "generated_samples.pt",
    )


if __name__ == "__main__":
    main()
