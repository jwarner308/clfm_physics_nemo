import torch.nn as nn
from einops.layers.torch import Rearrange
from physicsnemo.models.mlp import FullyConnected

DEFAULT_ACTIVATION = "gelu"


class PNFCEncoder(nn.Module):
    """Encoder network using PhysicsNemo's FullyConnected.

    Maps flattened sensor data to (mu, logvar) parameters of the latent distribution.
    """

    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        self.net = FullyConnected(
            in_features=input_size,
            layer_size=hidden_size,
            out_features=output_size * 2,
            num_layers=num_hidden_layers,
            activation_fn=DEFAULT_ACTIVATION,
        )

    def forward(self, x):
        x_flat = x.flatten(start_dim=1)
        return self.net(x_flat)


class PNFCTrunk(nn.Module):
    """Trunk network of the DeepONet using PhysicsNemo's FullyConnected.

    Processes spatial/temporal coordinates and reshapes output for the
    DeepONet inner product.
    """

    def __init__(
        self, input_size, hidden_size, output_size, num_outputs, num_hidden_layers
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers

        self.net = FullyConnected(
            in_features=input_size,
            layer_size=hidden_size,
            out_features=output_size * num_outputs,
            num_layers=num_hidden_layers,
            activation_fn=DEFAULT_ACTIVATION,
        )
        self.post = nn.Sequential(
            nn.GELU(),
            Rearrange("b n (i f) -> b n i f", f=num_outputs),
        )

    def forward(self, x):
        return self.post(self.net(x))


class PNFCBranch(nn.Module):
    """Branch network of the DeepONet using PhysicsNemo's FullyConnected.

    Processes latent vectors to produce coefficients for the DeepONet
    inner product.
    """

    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        self.net = FullyConnected(
            in_features=input_size,
            layer_size=hidden_size,
            out_features=output_size,
            num_layers=num_hidden_layers,
            activation_fn=DEFAULT_ACTIVATION,
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow in deeper networks."""

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.activation = nn.SiLU()
        self.proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, x):
        return self.activation(x + self.net(x))


class EnhancedBranchNetwork(nn.Module):
    """Enhanced branch network with residual connections for the materials problem."""

    def __init__(self, latent_dim, hidden_dim, output_dim, num_blocks=3):
        super().__init__()
        self.input_size = latent_dim  # needed by VAE.latent_dim property
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class PNFlowModel(nn.Module):
    """Flow matching velocity network using PhysicsNemo's FullyConnected.

    Takes concatenated [z, t] input and predicts velocity field.
    """

    def __init__(self, latent_dim, hidden_size, num_hidden_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        self.net = FullyConnected(
            in_features=latent_dim + 1,
            layer_size=hidden_size,
            out_features=latent_dim,
            num_layers=num_hidden_layers,
            activation_fn=DEFAULT_ACTIVATION,
        )

    def forward(self, x):
        return self.net(x)
