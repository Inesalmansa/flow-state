import torch
import torch.nn as nn

class UniformParticle(nn.Module):
    def __init__(self, n_particles, n_dimension, bound):
        """
        Initializes the UniformParticle class.

        Parameters:
        n_particles (int): Number of particles.
        n_dimension (int): Dimension of coordinates of every particle.
        bound (float): Boundary of the particle coordinates (symmetric for all dimensions).
        """
        super().__init__()
        self.n_particles = n_particles
        self.n_dimension = n_dimension
        self.bound = bound

    def sample(self, n_sample):
        """
        Generates particle coordinates following a uniform distribution.

        Parameters:
        n_sample (int): Number of needed samples.

        Returns:
        torch.Tensor: A tensor of shape (n_sample, n_particles * n_dimension) with coordinates sampled uniformly.
        """
        z = torch.empty(
            (n_sample, self.n_particles, self.n_dimension),
            dtype=torch.float32
        ).uniform_(-self.bound, self.bound)
        z = z.reshape(n_sample, self.n_particles * self.n_dimension)
        return z

    def forward(self, n_sample):
        """
        Forward method to integrate with the flow. Calls the sample method.

        Parameters:
        n_sample (int): Number of needed samples.

        Returns:
        torch.Tensor: A tensor of shape (n_sample, n_particles * n_dimension) with coordinates sampled uniformly.
        """
        return self.sample(n_sample)
