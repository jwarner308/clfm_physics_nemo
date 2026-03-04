from pathlib import Path
import h5py
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

from fgm.utils.grid import RectangularGrid
from fgm.utils.utils import reparameterize


# constants used for coherence evaluation:
# decay coefficients determine how fast coh. decays in each direction
C = [3.0, 3.0, 0.5]
# constants determining mean wind velocity
U_STAR = 1.90       # shear velocity
Z_0 = 0.015         # roughness length


def dense_eval(grid: RectangularGrid, discretization: tuple):
    '''
    Returns a (N**ndim, ndim) discretized grid over the domain.
    '''
    assert len(discretization) == grid.ndim
    ranges = [torch.linspace(0.0, 1.0, d, device=grid.low.device)
              for d in discretization]
    coords = torch.meshgrid(*ranges)
    x = torch.stack(coords, dim=-1).reshape(-1, grid.ndim)
    # Scale and shift to the grid's domain
    low = grid.low.unsqueeze(0)
    high = grid.high.unsqueeze(0)
    x = x * (high - low) + low
    return x


def get_true_coherence(freq, x_coord_j, x_coord_k):
    '''
    Evaluate prescribed coherence function for given frequency and spatial coordinates
    '''
    mean_vel_j = 2.5 * U_STAR * np.log(x_coord_j[2] / Z_0)
    mean_vel_k = 2.5 * U_STAR * np.log(x_coord_k[2] / Z_0)
    numerator = freq * np.linalg.norm(np.multiply(C, x_coord_j - x_coord_k))
    denominator = np.abs(mean_vel_j) + np.abs(mean_vel_k)
    coherence = np.exp(-numerator / denominator)
    return coherence


def stft(signal: Tensor, nperseg: int, hop_length: int = None, window: Tensor = None):
    '''
    signal: Tensor of size (batch x time)
    '''
    # parameters
    n_fft = nperseg
    if window is None:
        window = torch.hann_window(nperseg).to(signal.device)
    else:
        window = window.to(signal.device)

    if hop_length == None:
        noverlap = nperseg // 2
        hop_length = nperseg - noverlap

    '''
    Removed padding. But only circular padding was working
    '''
    # ## decide on number of padding elements on each end of the signal
    # padding_left = (n_fft // 2) if (n_fft % 2 == 0) else (n_fft - 1) // 2
    # padding_right = (n_fft // 2)
    # signal = F.pad(signal, (padding_left, padding_right), mode = 'circular') ## strange that I have to use circular padding

    # ## convert the signal into sliding windows
    segments = signal.unfold(1, nperseg, hop_length)
    # individually detrend each window
    segments = segments - segments.mean(dim=2, keepdim=True)
    # apply hann window
    segments = window[None, None, :] * segments

    # compute fft on each window
    w = torch.fft.fft(segments, n=n_fft)
    # the other half contains redundant information somehow
    return w[:, :, :nperseg // 2 + 1]
    # return w


def csd(signal1: Tensor, signal2: Tensor, nperseg: int, hop_length: int = None, window: Tensor = None):
    '''
    signal1: Tensor of size (batch x time)
    signal2: Tensor of size (batch x time)
    '''

    stft1 = stft(signal1, nperseg, hop_length, window=window)
    stft2 = stft(signal2, nperseg, hop_length, window=window)

    # mean over number of samples
    result = torch.mean(stft1 * stft2.conj(), dim=0)
    # (windows x num_freq/2)
    return result


def coherence(signal1: Tensor, signal2: Tensor, nperseg: int = None, hop_length: int = None, window: Tensor = None):
    '''
    signal1: Tensor of size (batch x time)
    signal2: Tensor of size (batch x time)
    '''
    assert len(signal1.shape) == 2
    assert signal1.shape == signal2.shape

    if nperseg == None:
        nperseg = signal1.shape[-1]

    Pjj = csd(signal1, signal1, nperseg, hop_length, window=window)
    Pkk = csd(signal2, signal2, nperseg, hop_length, window=window)
    Pjk = csd(signal1, signal2, nperseg, hop_length, window=window)
    # Compute coherence
    coh = (torch.abs(Pjk) ** 2 / (Pjj * Pkk)).real
    # mean over number of windows
    return coh.mean(dim=0)


'''
Generate a vectorized verrsion of the coherence function to operate on batches inputs signals:
(batch x num_colloc x time) X (batch x num_colloc x time) --> (num_colloc x frequency)
'''
v_coherence = torch.vmap(coherence, in_dims=1)

DEFAULT_NUM_SENSORS = 100


class WindDataset(Dataset):
    def __init__(self, data_file, sparse_sensors=False, num_sensors=None):
        super().__init__()

        # hardcoded path that assumes the existence of a data file at this particular location.
        path = (Path(__file__).parent / data_file).resolve()
        self.f = h5py.File(path)
        '''
        Hard coding lower and upper bound on v1 component. As long as these values are sufficiently extreme its ok.
        My goal is to normalize the values in a bounded range.
        '''
        self.v1_min = -2.0
        self.v1_max = 47.0
        self._v1_span = self.v1_max - self.v1_min

        if num_sensors is None:
            num_sensors = DEFAULT_NUM_SENSORS
        # constructing grid
        # sparse sensors mimick sodar - vertical strips; skip every two columns of data
        # inds: 0-9, 30-39, 60-69, 90-99
        if sparse_sensors:
            self.sensor_idx = torch.cat(
                [torch.arange(start, start + 10) for start in range(0, 100, 30)])
        else:
            self.sensor_idx = torch.arange(0, num_sensors)

        # extracting space and time grids from dataset
        x_grid = torch.tensor(np.array(self.f['x_grid']))[self.sensor_idx]
        t_grid = torch.tensor(np.array(self.f['t_grid']))

        low = torch.cat([
            x_grid.min(dim=0).values,
            t_grid.min(dim=0, keepdim=True).values
        ])

        high = torch.cat([
            x_grid.max(dim=0).values,
            t_grid.max(dim=0, keepdim=True).values
        ])

        self.grid = RectangularGrid(low, high)

        x_points = torch.tensor(np.array(self.f['x_grid']), dtype=torch.float)[
            self.sensor_idx]
        t_points = torch.tensor(np.array(self.f['t_grid']), dtype=torch.float)
        T = t_points.numel()

        x_points = x_points[:, None, :].repeat(1, T, 1)
        t_points = t_points[None, :, None].repeat(x_points.shape[0], 1, 1)
        self.points = torch.cat([x_points, t_points], dim=2).reshape(-1, 4)

    @property
    def dt(self):
        t = np.array(self.f['t_grid'])
        return t[1] - t[0]

    def __len__(self):
        return len(self.f['wind_samples'])

    def __getitem__(self, idx: int):
        u = torch.tensor(
            np.array(self.f['wind_samples'][str(idx)]['v1']), dtype=torch.float)[self.sensor_idx, :]
        # normalize approximately between zero and one
        u_norm = (u - self.v1_min) / self._v1_span
        return u_norm, self.points


class WindLoss(nn.Module):
    '''
    Loss function for the wind problem.

    num_colloc: number of collocation points (space).
    T: number of evenly spaced time points to sample signals at.
    '''

    def __init__(self, num_colloc: int, T: int, dataset: WindDataset):
        super().__init__()
        self.num_colloc = num_colloc
        self.T = T
        t = np.array(dataset.f['t_grid'])
        self.t_min = t.min()
        self.t_max = t.max()
        self.dt = (self.t_max - self.t_min) / self.T
        self.sensor_idx = dataset.sensor_idx

        self.register_buffer(
            '_fixed_dense_t',
            torch.linspace(self.t_min, self.t_max, self.T).reshape(1, 1, -1, 1),
            persistent=False,
        )
        self.register_buffer(
            '_freq_grid',
            torch.fft.rfftfreq(self.T, d=self.dt),
            persistent=False,
        )
        self.register_buffer(
            '_hann_window',
            torch.hann_window(self.T),
            persistent=False,
        )

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        u_hat = vae.decode(z, x)
        u_hat = u_hat.reshape(*u_true.shape)
        return torch.mean(torch.square(u_hat - u_true))

    def residual(self, vae, z: Tensor):
        device = z.device
        batch_size = z.shape[0]
        num_colloc = self.num_colloc
        nperseg = self.T

        # randomly select coordinates in the spatial domain. Multiply: 2* because we want pairs.
        # strip off random dummy time component
        x_collocation = vae.grid.sample(2 * num_colloc)[:, :3]

        # x1 = (num_colloc x 3)
        # x2 = (num_colloc x 3)
        # get the pairs of collocation points
        x1, x2 = x_collocation.chunk(2, dim=0)

        # unfold into (batch x num_colloc x 1 x dim)
        x_collocation = x_collocation.reshape(1, 2 * num_colloc, 1, 3)
        '''
        Repeat along batch dim because signals should be measured at the same collocation point acrosss many samples.
        Repeat along time dimention because we need to slap on the time label for each point.
        '''
        x_collocation = x_collocation.repeat(
            batch_size, 1, self.T, 1)  # repeat points along batch and time series

        # a fixed dense time grid to use instead of random samples
        fixed_dense_t = self._fixed_dense_t.to(device=device)
        fixed_dense_t = fixed_dense_t.repeat(batch_size, 2 * num_colloc, 1, 1)
        collocation = torch.cat([x_collocation, fixed_dense_t], dim=3)

        # flattening collocation into (batch x points x dim)
        collocation = collocation.reshape(batch_size, -1, vae.grid.ndim)
        u_hat = vae.decode(z, collocation)
        u_hat = u_hat.reshape(batch_size, 2 * self.num_colloc, self.T)

        '''
        We want two pairs of signals:
        (batch x num_colloc x T) and (batch x num_colloc x T)
        '''
        u_hat_1, u_hat_2 = u_hat.chunk(2, dim=1)
        coh = v_coherence(u_hat_1, u_hat_2, nperseg=nperseg, window=self._hann_window)
        freq_grid = self._freq_grid

        # compute the true coherence
        true_coh = []
        with torch.no_grad():
            x1_cpu = x1.cpu()
            x2_cpu = x2.cpu()
            freq_cpu = freq_grid.cpu()
            for i in range(num_colloc):
                true_coh.append(
                    torch.tensor([get_true_coherence(
                        f, x1_cpu[i, :], x2_cpu[i, :]) ** 2 for f in freq_cpu])
                )

        # should result in a (num_colloc x frequency) tensor
        true_coh = torch.stack(true_coh).to(coh.device)
        residual = torch.mean(torch.square(true_coh - coh))
        return residual, {'residual': residual.item()}

    def validate(self, vae, u_data: Tensor, x_data: Tensor):

        # If sparse sensors are used for training, need to filter full u_data for encoding:
        u_data_sensors = u_data[:, self.sensor_idx, :]
        z = reparameterize(*vae.encode(u_data_sensors))

        val_disc = (1, 10, 10, 256)
        x = dense_eval(vae.grid, val_disc)
        x = x[None, :, :].repeat(z.shape[0], 1, 1)

        u_hat = vae.decode(z, x)
        u_hat = u_hat.reshape(-1, val_disc[1] * val_disc[2], val_disc[3])

        u_gen_mean = torch.mean(u_hat, [0, 2])
        u_true_mean = torch.mean(u_data, [0, 2])
        u_gen_var = torch.var(u_hat, [0, 2])
        u_true_var = torch.var(u_data, [0, 2])

        mean_mse = torch.mean(torch.square(u_gen_mean - u_true_mean))
        var_mse = torch.mean(torch.square(u_gen_var - u_true_var))

        coh_mse, _ = self.residual(vae, z)

        return {'val_mean_mse': mean_mse.item(),
                'val_var_mse': var_mse.item(),
                'val_coherence_mse': coh_mse.item()}

