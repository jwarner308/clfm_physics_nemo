from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from clfm.problems.poisson_1d_noisy import NoisyPoisson1DDataset, NoisyPoisson1DLoss
from clfm.nn.vae import FunctionalVAE
from clfm.nn.fully_connected_nets import FCBranch, FCEncoder, FCTrunk
from poisson_utils import get_logging_path_and_name


def main(args):
    torch.manual_seed(args.seed)

    train_data = NoisyPoisson1DDataset(
        args.N_data, args.num_sensors, noise_std=args.noise_std
    )
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)

    loss = NoisyPoisson1DLoss(
        num_colloc=args.num_colloc,
        rec_mode=args.rec_mode,
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

    save_dir, name = get_logging_path_and_name(
        root_dir=args.result_dir,
        params_dict=vars(args),
    )
    logger = CSVLogger(save_dir=save_dir, name=name)
    checkpoint = ModelCheckpoint(monitor="total_loss", mode="min", save_top_k=1)

    gradient_clip_val = args.grad_clip if args.grad_clip > 0 else None
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[
            checkpoint,
        ],
        gradient_clip_val=gradient_clip_val,
    )
    trainer.fit(vae, train_loader)
    train_data.store_samples(logger.log_dir)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    # problem set up
    parser.add_argument("--N_data", type=int, default=100)
    # NN architecture / capacity
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
    # method / run specifics
    parser.add_argument("--num_sensors", type=int, default=25)
    parser.add_argument("--num_colloc", type=int, default=50)
    parser.add_argument("--res_weight", type=float, default=0.001)
    parser.add_argument("--kld_weight", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    # noise specifics
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--rec_mode", type=str, default="softplus",
                        choices=["standard", "softplus", "learned_noise"])
    parser.add_argument("--beta_softplus", type=float, default=5.0)
    parser.add_argument("--noise_std_init", type=float, default=0.05)
    # misc:
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--result_dir", type=str, default="results/poisson_1d_noisy")
    main(parser.parse_args())
