import torch
from torch import nn, Tensor
from einops import einsum

from clfm_pn.utils import kl_divergence, reparameterize


class FunctionalVAE(nn.Module):
    """Variational Autoencoder with a DeepONet function decoder.

    This is the PhysicsNemo-compatible version. Training logic (loss computation,
    optimizer setup) is handled externally by PhysicsNemo's training infrastructure
    (StaticCaptureTraining, LaunchLogger, Hydra configs).

    Parameters
    ----------
    encoder : nn.Module
        Encoder mapping input data to latent space (outputs concatenated mu, logvar).
    branch : nn.Module
        Branch network of the DeepONet processing the latent representation.
    trunk : nn.Module
        Trunk network of the DeepONet processing spatial/temporal coordinates.
    num_fields : int
        Number of output fields/channels.
    grid : nn.Module
        Grid object with normalize, sample, and dense methods.
    """

    def __init__(self, encoder, branch, trunk, num_fields, grid):
        super().__init__()
        self.grid = grid
        self._encoder = encoder
        self._branch = branch
        self._trunk = trunk
        self._bias = nn.Parameter(torch.zeros(1, 1, num_fields))

    @property
    def latent_dim(self):
        return self._branch.input_size

    def encode(self, u: Tensor):
        return self._encoder(u).chunk(2, dim=1)  # [mu, logvar]

    def decode(self, z: Tensor, x: Tensor):
        """Evaluate latent function representation z at points x with DeepONet.

        z: (batch x interact_dim)
        x: (batch x num_points x point_dim)
        """
        x = self.grid.normalize(x)
        x = self._trunk(x)
        z = self._branch(z)
        return einsum(z, x, "b i, b n i f -> b n f") + self._bias

    def forward(self, u: Tensor, x: Tensor) -> Tensor:
        return self.decode(reparameterize(*self.encode(u)), x)
