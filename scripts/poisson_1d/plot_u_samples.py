"""
Load saved checkpoints from a compare_noise_methods run and plot
true vs. generated u(x) sample ensembles for each method.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

from clfm.problems.poisson_1d_noisy import NoisyPoisson1DDataset, NoisyPoisson1DLoss
from clfm.problems.poisson_1d import Poisson1DLoss, Poisson1DDataset
from clfm.nn.vae import FunctionalVAE
from clfm.nn.fully_connected_nets import FCBranch, FCEncoder, FCTrunk
from clfm.utils import reparameterize


# ---- Config (match the original run) ----
RESULT_DIR = Path("results/poisson_1d_noise_0p05_mps")
NOISE_STD = 0.05
N_DATA = 100
NUM_SENSORS = 25
NUM_COLLOC = 50
NUM_DENSE = 200
N_TEST = 500
SEED = 0
LATENT_DIM = 4
P_DEEPONET = 64
H_ENCODER = H_TRUNK = H_BRANCH = 128
NHL_TRUNK = NHL_BRANCH = 2
NHL_ENCODER = 3
N_PLOT = 50
MODES = ["original", "standard", "softplus", "learned_noise"]

# ---- Recreate datasets with same seeds ----
torch.manual_seed(SEED)
train_data = NoisyPoisson1DDataset(N_DATA, NUM_SENSORS, noise_std=NOISE_STD)

torch.manual_seed(SEED + 999)
test_data = Poisson1DDataset(N=N_TEST, num_sensors=NUM_DENSE)
torch.manual_seed(SEED)

x_dense = torch.linspace(train_data.x_min, train_data.x_max, NUM_DENSE).reshape(-1, 1)


def build_vae(rec_mode):
    if rec_mode == "original":
        loss = Poisson1DLoss(num_colloc=NUM_COLLOC)
    else:
        loss = NoisyPoisson1DLoss(
            num_colloc=NUM_COLLOC,
            rec_mode=rec_mode,
            noise_std=NOISE_STD,
            num_sensors=NUM_SENSORS,
        )
        loss.dataset = train_data

    encoder = FCEncoder(NUM_SENSORS, H_ENCODER, LATENT_DIM, NHL_ENCODER)
    branch = FCBranch(LATENT_DIM, H_BRANCH, P_DEEPONET, NHL_BRANCH)
    trunk = FCTrunk(train_data.grid.ndim, H_TRUNK, P_DEEPONET,
                    num_outputs=train_data.num_fields, num_hidden_layers=NHL_TRUNK)
    return FunctionalVAE(
        encoder=encoder, branch=branch, trunk=trunk,
        num_fields=train_data.num_fields, grid=train_data.grid,
        lr=1e-3, res_weight=0.001, kld_weight=1e-6, loss=loss,
    )


def load_vae(rec_mode):
    ckpt_pattern = str(RESULT_DIR / rec_mode / "run" / "version_0" / "checkpoints" / "*.ckpt")
    ckpts = glob(ckpt_pattern)
    assert ckpts, f"No checkpoint found for {rec_mode} at {ckpt_pattern}"
    vae = build_vae(rec_mode)
    state = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    vae.load_state_dict(state["state_dict"])
    vae.eval()
    return vae


# ---- Load all VAEs and generate u samples ----
x_dense_batch = x_dense.unsqueeze(0).expand(N_DATA, -1, -1)
u_noisy = train_data.u_samples_noisy
u_clean = train_data.u_samples_clean
u_true_dense = test_data.u_samples  # (N_TEST, NUM_DENSE)

results = {}
for mode in MODES:
    print(f"Loading {mode} ...")
    vae = load_vae(mode)
    with torch.no_grad():
        mu, logvar = vae.encode(u_noisy)
        z = reparameterize(mu, logvar)
        f_dense = vae.decode(z, x_dense_batch)
        u_hat_dense, _ = f_dense.chunk(2, dim=2)
        u_hat_dense = u_hat_dense.squeeze(2)
    results[mode] = u_hat_dense

x_d = x_dense.squeeze().numpy()

# ---- Figure: True vs Generated u(x) sample ensembles ----
fig, axes = plt.subplots(2, len(MODES), figsize=(5 * len(MODES), 8))

for col, mode in enumerate(MODES):
    u_hat = results[mode]
    n_true_plot = min(N_PLOT, u_true_dense.shape[0])

    # Row 0: true u samples
    ax_true = axes[0, col]
    for i in range(n_true_plot):
        ax_true.plot(x_d, u_true_dense[i].numpy(), color="gray", alpha=0.15, linewidth=0.8)
    ax_true.set_title(f"true u samples")
    ax_true.set_xlabel("x")
    ax_true.set_ylabel("u(x)")
    yl = ax_true.get_ylim()

    # Row 1: generated u samples
    ax_gen = axes[1, col]
    for i in range(min(N_PLOT, u_hat.shape[0])):
        ax_gen.plot(x_d, u_hat[i].numpy(), color="steelblue", alpha=0.15, linewidth=0.8)
    ax_gen.set_title(f"{mode}: gen u samples")
    ax_gen.set_xlabel("x")
    ax_gen.set_ylabel("u(x)")
    ax_gen.set_ylim(yl)

fig.suptitle(
    f"u(x) Sample Ensembles: True vs Generated (noise_std={NOISE_STD})",
    fontsize=13,
)
fig.tight_layout()
out_path = RESULT_DIR / "u_samples_overlay.png"
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
plt.close(fig)
