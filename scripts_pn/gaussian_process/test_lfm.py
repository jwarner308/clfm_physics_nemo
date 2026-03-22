import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint

from clfm_pn.problems.gaussian_process import GPDataset
from clfm_pn.nn.fully_connected_nets import PNFCBranch, PNFCEncoder, PNFCTrunk, PNFlowModel
from clfm_pn.nn.vae import FunctionalVAE
from clfm_pn.utils.latent_fm import sample_lfm


@hydra.main(version_base="1.3", config_path="conf", config_name="config_lfm")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    NUM_SAMPLES = 1000
    NUM_TIME_STEPS = 100

    result_dir = Path(cfg.output.result_dir)

    test_data = GPDataset(
        N=NUM_SAMPLES,
        num_sensors=cfg.data.num_sensors,
        cov_len=cfg.data.gp_cov_len,
        variance=cfg.data.gp_var,
        mean="LINEAR",
    )
    x_grid = test_data.x.to(device)

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
        input_size=test_data.grid.ndim,
        hidden_size=cfg.arch.h_trunk,
        output_size=cfg.arch.p_deeponet,
        num_outputs=test_data.num_fields,
        num_hidden_layers=cfg.arch.nhl_trunk,
    )
    vae = FunctionalVAE(
        encoder=encoder,
        branch=branch,
        trunk=trunk,
        num_fields=test_data.num_fields,
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
    y_gen = sample_lfm(NUM_SAMPLES, x_grid, flow_model, vae, NUM_TIME_STEPS, device)
    y_gen = y_gen.squeeze().cpu()

    # Print comparison statistics
    true_samples = test_data.samples
    print(f"True  - mean: {true_samples.mean():.4f}, std: {true_samples.std():.4f}")
    print(f"Gen   - mean: {y_gen.mean():.4f}, std: {y_gen.std():.4f}")

    torch.save(
        {"y_gen": y_gen, "y_true": true_samples, "x": test_data.x},
        result_dir / "generated_samples.pt",
    )


if __name__ == "__main__":
    main()
