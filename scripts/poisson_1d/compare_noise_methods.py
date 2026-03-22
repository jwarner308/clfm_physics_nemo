"""
Compare noise-handling strategies for the Poisson 1D inverse problem:
  0. original  — repo's Poisson1DLoss fed noisy data (no noise awareness)
  1. standard  — NoisyPoisson1DLoss with plain MSE (same math, different class)
  2. softplus  — discrepancy-principle relaxation
  3. learned_noise — Gaussian NLL with learnable sigma_n
"""

from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from clfm.problems.poisson_1d_noisy import NoisyPoisson1DDataset, NoisyPoisson1DLoss
from clfm.problems.poisson_1d import Poisson1DLoss, v_func
from clfm.nn.vae import FunctionalVAE
from clfm.nn.fully_connected_nets import FCBranch, FCEncoder, FCTrunk
from clfm.utils import reparameterize


def build_and_train(args, rec_mode, train_data, result_dir):
    """Build a VAE with the given rec_mode, train it, return (vae, metrics_df, log_dir)."""
    torch.manual_seed(args.seed)

    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)

    if rec_mode == "original":
        loss = Poisson1DLoss(num_colloc=args.num_colloc)
    else:
        loss = NoisyPoisson1DLoss(
            num_colloc=args.num_colloc,
            rec_mode=rec_mode,
            noise_std=args.noise_std,
            beta=args.beta_softplus,
            noise_std_init=args.noise_std_init,
            num_sensors=args.num_sensors,
        )
        loss.dataset = train_data

    encoder = FCEncoder(
        input_size=train_data.num_sensors,
        hidden_size=args.h_encoder,
        output_size=args.latent_dim,
        num_hidden_layers=args.nhl_encoder,
    )
    branch = FCBranch(
        input_size=args.latent_dim,
        hidden_size=args.h_branch,
        output_size=args.p_deeponet,
        num_hidden_layers=args.nhl_branch,
    )
    trunk = FCTrunk(
        input_size=train_data.grid.ndim,
        hidden_size=args.h_trunk,
        output_size=args.p_deeponet,
        num_outputs=train_data.num_fields,
        num_hidden_layers=args.nhl_trunk,
    )
    vae = FunctionalVAE(
        encoder=encoder,
        branch=branch,
        trunk=trunk,
        num_fields=train_data.num_fields,
        grid=train_data.grid,
        lr=args.lr,
        res_weight=args.res_weight,
        kld_weight=args.kld_weight,
        loss=loss,
    )

    log_dir = Path(result_dir) / rec_mode
    logger = CSVLogger(save_dir=str(log_dir), name="run")
    checkpoint = ModelCheckpoint(monitor="total_loss", mode="min", save_top_k=1)

    gradient_clip_val = args.grad_clip if args.grad_clip > 0 else None
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint],
        gradient_clip_val=gradient_clip_val,
        enable_progress_bar=True,
    )
    trainer.fit(vae, train_loader)

    metrics_path = Path(logger.log_dir) / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None

    return vae, metrics, logger.log_dir


def evaluate(vae, train_data, test_data, num_dense=200):
    """Encode training data, decode on a dense grid, compare against test statistics."""
    u_noisy = train_data.u_samples_noisy
    u_clean = train_data.u_samples_clean
    x_sensor = train_data.x_sensor

    # Dense evaluation grid
    x_dense = torch.linspace(
        train_data.x_min, train_data.x_max, num_dense
    ).reshape(-1, 1)
    x_dense_batch = x_dense.unsqueeze(0).expand(u_noisy.shape[0], -1, -1)

    # Also decode at sensor locations for MSE comparison
    x_sensor_batch = x_sensor.unsqueeze(0).expand(u_noisy.shape[0], -1, -1)

    with torch.no_grad():
        mu, logvar = vae.encode(u_noisy)
        z = reparameterize(mu, logvar)

        # Sensor-point reconstruction
        f_sensor = vae.decode(z, x_sensor_batch)
        u_hat_sensor, v_hat_sensor = f_sensor.chunk(2, dim=2)
        u_hat_sensor = u_hat_sensor.squeeze(2)
        v_hat_sensor = v_hat_sensor.squeeze(2)

        # Dense-grid reconstruction
        f_dense = vae.decode(z, x_dense_batch)
        u_hat_dense, v_hat_dense = f_dense.chunk(2, dim=2)
        u_hat_dense = u_hat_dense.squeeze(2)
        v_hat_dense = v_hat_dense.squeeze(2)

    # Compute true v on the dense grid for test data
    v_true_dense = test_data.v_samples  # (N_test, num_dense)
    u_true_dense = test_data.u_samples

    # Scalar metrics at sensor locations
    mse_u_clean = torch.mean(torch.square(u_hat_sensor - u_clean)).item()
    mse_u_noisy = torch.mean(torch.square(u_hat_sensor - u_noisy)).item()
    mse_v = torch.mean(torch.square(v_hat_sensor - train_data.v_samples)).item()

    # Spatial statistics on dense grid: mean and std vs x
    v_gen_mean = v_hat_dense.mean(dim=0)
    v_gen_std = v_hat_dense.std(dim=0)
    v_true_mean = v_true_dense.mean(dim=0)
    v_true_std = v_true_dense.std(dim=0)

    u_gen_mean = u_hat_dense.mean(dim=0)
    u_gen_std = u_hat_dense.std(dim=0)
    u_true_mean = u_true_dense.mean(dim=0)
    u_true_std = u_true_dense.std(dim=0)

    # Roughness: mean absolute second derivative (finite differences) of v samples
    # High values = high-frequency wiggles from noise fitting
    dx = (train_data.x_max - train_data.x_min) / (num_dense - 1)
    v_d2 = torch.diff(v_hat_dense, n=2, dim=1) / dx**2
    v_true_d2 = torch.diff(v_true_dense, n=2, dim=1) / dx**2
    roughness_gen = torch.mean(torch.abs(v_d2)).item()
    roughness_true = torch.mean(torch.abs(v_true_d2)).item()

    u_d2 = torch.diff(u_hat_dense, n=2, dim=1) / dx**2
    u_true_d2 = torch.diff(u_true_dense, n=2, dim=1) / dx**2
    roughness_u_gen = torch.mean(torch.abs(u_d2)).item()
    roughness_u_true = torch.mean(torch.abs(u_true_d2)).item()

    return {
        "mse_u_vs_clean": mse_u_clean,
        "mse_u_vs_noisy": mse_u_noisy,
        "mse_v_vs_true": mse_v,
        "v_mean_mse": torch.mean(torch.square(v_gen_mean - v_true_mean)).item(),
        "v_std_mse": torch.mean(torch.square(v_gen_std - v_true_std)).item(),
        "roughness_v_gen": roughness_gen,
        "roughness_v_true": roughness_true,
        "roughness_u_gen": roughness_u_gen,
        "roughness_u_true": roughness_u_true,
        # Dense arrays for plotting
        "x_dense": x_dense.squeeze(),
        "u_hat_dense": u_hat_dense,
        "v_hat_dense": v_hat_dense,
        "u_true_dense": u_true_dense,
        "v_true_dense": v_true_dense,
        "v_gen_mean": v_gen_mean,
        "v_gen_std": v_gen_std,
        "v_true_mean": v_true_mean,
        "v_true_std": v_true_std,
        "u_gen_mean": u_gen_mean,
        "u_gen_std": u_gen_std,
        "u_true_mean": u_true_mean,
        "u_true_std": u_true_std,
        # Sensor-point results (for backward compat)
        "u_hat": u_hat_sensor,
        "v_hat": v_hat_sensor,
    }


def main(args):
    torch.manual_seed(args.seed)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    NUM_DENSE = 200
    N_TEST = 500

    # Shared noisy dataset (same seed => same noise realization)
    train_data = NoisyPoisson1DDataset(
        args.N_data, args.num_sensors, noise_std=args.noise_std
    )

    # Clean test data on dense grid for ground-truth statistics
    torch.manual_seed(args.seed + 999)
    from clfm.problems.poisson_1d import Poisson1DDataset
    test_data = Poisson1DDataset(N=N_TEST, num_sensors=NUM_DENSE)
    torch.manual_seed(args.seed)  # restore for reproducible training

    modes = ["original", "standard", "softplus", "learned_noise"]
    results = {}
    all_metrics = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Training with rec_mode = {mode}")
        print(f"{'='*60}")
        vae, metrics, log_dir = build_and_train(
            args, mode, train_data, result_dir
        )
        vae.eval()
        eval_result = evaluate(vae, train_data, test_data, num_dense=NUM_DENSE)
        results[mode] = eval_result
        all_metrics[mode] = metrics
        print(f"  Done. MSE(u vs clean)={eval_result['mse_u_vs_clean']:.6f}, "
              f"MSE(v vs true)={eval_result['mse_v_vs_true']:.6f}, "
              f"roughness_v={eval_result['roughness_v_gen']:.4f} "
              f"(true={eval_result['roughness_v_true']:.4f})")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY  (noise_std={args.noise_std}, epochs={args.epochs}, N={args.N_data})")
    print(f"{'='*90}")
    header = (f"{'Method':<16} {'MSE(u,clean)':<14} {'MSE(v,true)':<14} "
              f"{'v mean MSE':<14} {'v std MSE':<14} {'v roughness':<14} {'(true)':<10}")
    print(header)
    print("-" * 90)
    for mode in modes:
        r = results[mode]
        print(f"{mode:<16} {r['mse_u_vs_clean']:<14.6f} {r['mse_v_vs_true']:<14.6f} "
              f"{r['v_mean_mse']:<14.6f} {r['v_std_mse']:<14.6f} "
              f"{r['roughness_v_gen']:<14.4f} {r['roughness_v_true']:<10.4f}")

    # ---- Figure 1: Sample-level comparison on dense grid ----
    n_modes = len(modes)
    fig, axes = plt.subplots(2, n_modes, figsize=(5 * n_modes, 8))
    sample_idx = 0
    x_sensor = train_data.x_sensor.squeeze().numpy()
    u_noisy = train_data.u_samples_noisy[sample_idx].numpy()

    for col, mode in enumerate(modes):
        r = results[mode]
        x_d = r["x_dense"].numpy()
        u_hat_d = r["u_hat_dense"][sample_idx].numpy()
        v_hat_d = r["v_hat_dense"][sample_idx].numpy()
        u_true_d = r["u_true_dense"][sample_idx].numpy()
        v_true_d = r["v_true_dense"][sample_idx].numpy()

        ax_u = axes[0, col]
        ax_u.plot(x_d, u_true_d, "k-", label="u true", linewidth=1.5)
        ax_u.plot(x_sensor, u_noisy, "r.", label="u noisy", markersize=4, alpha=0.7)
        ax_u.plot(x_d, u_hat_d, "b--", label="u recon", linewidth=1.5)
        ax_u.set_title(f"{mode}\nu recon (dense)")
        ax_u.legend(fontsize=7)
        ax_u.set_xlabel("x")

        ax_v = axes[1, col]
        ax_v.plot(x_d, v_true_d, "k-", label="v true", linewidth=1.5)
        ax_v.plot(x_d, v_hat_d, "b--", label="v inferred", linewidth=1.5)
        ax_v.set_title(f"v inference (dense)")
        ax_v.legend(fontsize=7)
        ax_v.set_xlabel("x")

    fig.suptitle(
        f"Sample Reconstruction on Dense Grid "
        f"(noise_std={args.noise_std}, epochs={args.epochs})",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = result_dir / "sample_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nSample comparison saved to {fig_path}")
    plt.close(fig)

    # ---- Figure 2: v statistics (mean +/- std) vs x ----
    fig2, axes2 = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5))
    for col, mode in enumerate(modes):
        r = results[mode]
        x_d = r["x_dense"].numpy()
        ax = axes2[col]
        # True
        ax.plot(x_d, r["v_true_mean"].numpy(), "k-", label="true mean", linewidth=1.5)
        ax.fill_between(
            x_d,
            (r["v_true_mean"] - r["v_true_std"]).numpy(),
            (r["v_true_mean"] + r["v_true_std"]).numpy(),
            color="gray", alpha=0.3, label="true +/- std",
        )
        # Generated
        ax.plot(x_d, r["v_gen_mean"].numpy(), "b--", label="gen mean", linewidth=1.5)
        ax.fill_between(
            x_d,
            (r["v_gen_mean"] - r["v_gen_std"]).numpy(),
            (r["v_gen_mean"] + r["v_gen_std"]).numpy(),
            color="blue", alpha=0.15, label="gen +/- std",
        )
        ax.set_title(f"{mode}")
        ax.set_xlabel("x")
        ax.set_ylabel("v(x)")
        ax.legend(fontsize=7)

    fig2.suptitle(
        f"v(x) Statistics: Generated vs Test "
        f"(noise_std={args.noise_std})",
        fontsize=13,
    )
    fig2.tight_layout()
    fig2_path = result_dir / "v_statistics.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"v statistics saved to {fig2_path}")
    plt.close(fig2)

    # ---- Figure 3: Overlay of generated v samples vs true v samples ----
    n_plot = min(50, train_data.N)
    fig3, axes3 = plt.subplots(2, n_modes, figsize=(5 * n_modes, 8))
    for col, mode in enumerate(modes):
        r = results[mode]
        x_d = r["x_dense"].numpy()

        # True samples
        ax_true = axes3[0, col]
        for i in range(min(n_plot, r["v_true_dense"].shape[0])):
            ax_true.plot(x_d, r["v_true_dense"][i].numpy(),
                         color="gray", alpha=0.15, linewidth=0.8)
        ax_true.set_title(f"true v samples")
        ax_true.set_xlabel("x")
        ax_true.set_ylabel("v(x)")
        if col == 0:
            ax_true.set_ylim(auto=True)
        # fix y-limits across columns
        yl = ax_true.get_ylim()

        # Generated samples
        ax_gen = axes3[1, col]
        for i in range(n_plot):
            ax_gen.plot(x_d, r["v_hat_dense"][i].numpy(),
                        color="blue", alpha=0.15, linewidth=0.8)
        ax_gen.set_title(f"{mode}: gen v samples")
        ax_gen.set_xlabel("x")
        ax_gen.set_ylabel("v(x)")
        ax_gen.set_ylim(yl)

    fig3.suptitle(
        f"v(x) Sample Ensembles: True vs Generated "
        f"(noise_std={args.noise_std})",
        fontsize=13,
    )
    fig3.tight_layout()
    fig3_path = result_dir / "v_samples_overlay.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"v samples overlay saved to {fig3_path}")
    plt.close(fig3)

    # ---- Figure 4: Roughness / spectral content ----
    # Show mean power spectrum of v samples (generated vs true)
    fig4, axes4 = plt.subplots(1, n_modes, figsize=(5 * n_modes, 4))
    for col, mode in enumerate(modes):
        r = results[mode]
        ax = axes4[col]

        # FFT of v samples
        v_gen_np = r["v_hat_dense"].numpy()
        v_true_np = r["v_true_dense"][:v_gen_np.shape[0]].numpy()

        ps_gen = np.mean(np.abs(np.fft.rfft(v_gen_np, axis=1))**2, axis=0)
        ps_true = np.mean(np.abs(np.fft.rfft(v_true_np, axis=1))**2, axis=0)
        freqs = np.fft.rfftfreq(NUM_DENSE, d=(train_data.x_max - train_data.x_min) / NUM_DENSE)

        ax.semilogy(freqs[1:], ps_true[1:], "k-", label="true", linewidth=1.5)
        ax.semilogy(freqs[1:], ps_gen[1:], "b--", label="generated", linewidth=1.5)
        ax.set_title(f"{mode}")
        ax.set_xlabel("frequency")
        ax.set_ylabel("power")
        ax.legend(fontsize=8)

    fig4.suptitle(
        f"v(x) Power Spectrum: Generated vs True "
        f"(noise_std={args.noise_std})",
        fontsize=13,
    )
    fig4.tight_layout()
    fig4_path = result_dir / "v_power_spectrum.png"
    fig4.savefig(fig4_path, dpi=150)
    print(f"Power spectrum saved to {fig4_path}")
    plt.close(fig4)

    # ---- Figure 5: Training loss curves ----
    fig5, ax5 = plt.subplots(1, 1, figsize=(8, 5))
    for mode in modes:
        m = all_metrics[mode]
        if m is not None and "total_loss" in m.columns:
            loss_vals = m["total_loss"].dropna()
            ax5.plot(loss_vals.values, label=mode, alpha=0.8)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Total Loss")
    ax5.set_title("Training Loss Curves")
    ax5.legend()
    ax5.set_yscale("log")
    fig5.tight_layout()
    fig5_path = result_dir / "training_curves.png"
    fig5.savefig(fig5_path, dpi=150)
    print(f"Training curves saved to {fig5_path}")
    plt.close(fig5)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--N_data", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--p_deeponet", type=int, default=64)
    parser.add_argument("--h_encoder", type=int, default=128)
    parser.add_argument("--h_trunk", type=int, default=128)
    parser.add_argument("--h_branch", type=int, default=128)
    parser.add_argument("--nhl_trunk", type=int, default=2)
    parser.add_argument("--nhl_branch", type=int, default=2)
    parser.add_argument("--nhl_encoder", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_sensors", type=int, default=25)
    parser.add_argument("--num_colloc", type=int, default=50)
    parser.add_argument("--res_weight", type=float, default=0.001)
    parser.add_argument("--kld_weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--beta_softplus", type=float, default=5.0)
    parser.add_argument("--noise_std_init", type=float, default=0.05)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default="results/poisson_1d_noisy_comparison")
    main(parser.parse_args())
