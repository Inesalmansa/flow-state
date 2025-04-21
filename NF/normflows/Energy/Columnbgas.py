import normflows as nf
import torch
import torch.nn as nn

class Columnbgas(nn.Module):
    def __init__(self, dim, n_particles):
        super().__init__()
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles

    def _energy(self, x):
        x = x.contiguous()
        e1 = (1/nf.utils.compute_distances(x, self._n_particles, self._n_dimensions)).sum(dim =-1)
        e2 = x.pow(2).sum(dim = 1)
        energies = e1 + e2
        return energies
