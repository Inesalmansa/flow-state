import torch
import torch.nn as nn

class UniformParticle(nn.Module):
    def __init__(self, n_particles, n_dimension, bound, device='cpu'):
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
        self.device = device

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
            dtype=torch.float32,
            device=self.device
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
    
    def log_prob(self, z):
        """
        Calculates the log probability density for given samples.

        Parameters:
        z (torch.Tensor): Input tensor of shape (batch_size, n_particles * n_dimension) containing particle coordinates.

        Returns:
        torch.Tensor: Log probability density for each sample in the batch. Returns -inf for samples outside bounds.
                     Shape is (batch_size,).

        Notes:
        For a uniform distribution over [-bound,bound], the probability density is 1/(2*bound) for each dimension.
        The log probability is thus -log(2*bound) per dimension, summed over all dimensions.
        """
        # Flatten z to (batch, D) if necessary
        # Check if z is within bounds:
        in_bounds = ((z >= -self.bound) & (z <= self.bound)).all(dim=1)
        # Calculate log probability for samples inside the bounds
        D = self.n_particles * self.n_dimension
        constant_log_prob = -D * torch.log(torch.tensor(2 * self.bound))
        log_prob_tensor = torch.full((z.size(0),), constant_log_prob, device=z.device, dtype=z.dtype)
        # Assign -inf for out-of-bound samples
        log_prob_tensor[~in_bounds] = -float('inf')
        return log_prob_tensor
