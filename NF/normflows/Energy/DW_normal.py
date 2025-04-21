import torch
import torch.nn as nn

class DWNormal(nn.Module):
    def __init__(self, n_particles, n_dimension, bound, sigma=None, device='cpu'):
        """
        Initializes the DWNormal class which represents a mixture of two normal distributions.

        For each coordinate the probability density is defined as:
            p(x) = 0.5 * N(x; m1, sigma^2) + 0.5 * N(x; m2, sigma^2)
        where:
            m1 = -0.5 * bound  (1/4 through the box [-bound, bound])
            m2 =  0.5 * bound  (3/4 through the box [-bound, bound])
        and sigma is the standard deviation of each component (default is bound/8).

        Parameters:
            n_particles (int): Number of particles.
            n_dimension (int): Dimension of each particle’s coordinates.
            bound (float): Defines the scale and determines the means.
            sigma (float, optional): Standard deviation for the normal components.
                                     Defaults to bound / 8 if not provided.
            device (str): Device to run the computations (e.g., 'cpu' or 'cuda').
        """
        super().__init__()
        self.n_particles = n_particles
        self.n_dimension = n_dimension
        self.bound = bound
        self.device = device
        self.sigma = sigma if sigma is not None else bound / 8.0

        # Set the means based on the box [–bound, bound]
        self.mean1 = -0.5 * bound  # 1/4 through the box
        self.mean2 = 0.5 * bound   # 3/4 through the box

    def sample(self, n_sample):
        """
        Generates particle coordinates by sampling from the mixture distribution.

        Parameters:
            n_sample (int): Number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (n_sample, n_particles * n_dimension)
                          containing the generated coordinates.
        """
        shape = (n_sample, self.n_particles, self.n_dimension)
        # For each coordinate, randomly choose one of the two normal components
        mask = torch.bernoulli(0.5 * torch.ones(shape, device=self.device))
        # Generate noise scaled by sigma
        noise = self.sigma * torch.randn(shape, device=self.device)
        # If mask == 0, use mean1; if mask == 1, use mean2
        samples = (1 - mask) * self.mean1 + mask * self.mean2 + noise
        samples = samples.reshape(n_sample, self.n_particles * self.n_dimension)
        return samples

    def forward(self, n_sample):
        """
        Forward method to integrate with a normalizing flow.
        Calls the sample method.

        Parameters:
            n_sample (int): Number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (n_sample, n_particles * n_dimension)
                          containing the generated coordinates.
        """
        return self.sample(n_sample)
    
    def log_prob(self, z):
        """
        Calculates the log probability density for a given batch of samples.

        For each coordinate x, the density is:
            p(x) = 0.5 * N(x; mean1, sigma^2) + 0.5 * N(x; mean2, sigma^2)
        The overall log probability is the sum over all coordinates.

        Parameters:
            z (torch.Tensor): Tensor of shape (batch_size, n_particles * n_dimension)
                              containing particle coordinates.

        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing the log probability
                          density of each sample.
        """
        # Reshape z to (batch_size, n_particles, n_dimension)
        z = z.reshape(z.size(0), self.n_particles, self.n_dimension)
        
        # Calculate the normalizing constant for a normal distribution
        norm_const = 1.0 / (self.sigma * torch.sqrt(torch.tensor(2 * 3.141592653589793, device=self.device)))
        # Compute density for both components for each coordinate
        density1 = norm_const * torch.exp(-0.5 * ((z - self.mean1) / self.sigma) ** 2)
        density2 = norm_const * torch.exp(-0.5 * ((z - self.mean2) / self.sigma) ** 2)
        
        # Mixture density for each coordinate
        mixture_density = 0.5 * density1 + 0.5 * density2
        
        # Take log and sum over particles and dimensions
        log_prob = torch.log(mixture_density)
        log_prob = log_prob.sum(dim=(1, 2))
        return log_prob
