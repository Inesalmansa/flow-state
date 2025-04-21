import normflows as nf
import torch
import torch.nn as nn

class SimpleLJ(nn.Module):
    def __init__(self, dim, n_particles,temperature,bound):
        super().__init__()
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles
        self.temperature = temperature
        self.bound = bound


    def _energy(self, x):
        x = x.contiguous()

        x_packed = x.clone()
        x_packed = x_packed.reshape(-1, self._n_particles, self._n_dimensions)
        d_norm = x_packed - 2 * self.bound * torch.round(x_packed / (self.bound * 2))
        zeros_tensor = torch.zeros((d_norm.shape[0], 1, d_norm.shape[2]), device='cuda')  # 在同一设备上创建零张量
        # giving error zeros_tensor = torch.zeros((d_norm.shape[0], 1, d_norm.shape[2])) 
        d_norm = torch.cat((zeros_tensor.to(x.device), d_norm), axis=1)  # Shape: (50, 3, 2)

        expanded_tensor = d_norm.unsqueeze(2)
        diff = expanded_tensor - expanded_tensor.transpose(1, 2)
        distance_matrix = torch.norm(diff, dim=-1)
        num_particles = d_norm.size(1)
        triu_indices = torch.triu_indices(num_particles, num_particles, offset=1)
        unique_distances = distance_matrix[:, triu_indices[0], triu_indices[1]]
        bkpoint = 0.82
        energies = torch.where(unique_distances<=bkpoint,
                               -80*(unique_distances-(bkpoint))+30,
                               4*(pow(1/unique_distances,12)-pow(1/unique_distances,6)))
        energies = energies.sum(dim = 1)


        
        return energies/self.temperature
    

class DoubleWellLJ(SimpleLJ):
    def __init__(self, dim, n_particles, temperature, bound, V0_list=None, r0=1.0, k=10.0):
        super().__init__(dim, n_particles, temperature, bound)
        
        # Default depths for each well if not provided
        if V0_list is None:
            V0_list = [-4.0, -4.0]
            
        self.V0_list = torch.tensor(V0_list, dtype=torch.float32)
        self.r0 = r0
        self.k = k
        
        # Define centers of the two wells
        self.centers = torch.tensor([
            [-bound/2, 0.0],
            [bound/2, 0.0]
        ], dtype=torch.float32)
    
    def double_well_potential(self, positions):
        """
        Compute a double well potential with variable depths for each well in a 2D square box.
        The box spans from -bound to bound in both dimensions.
        
        Parameters:
            positions : torch.Tensor
                Tensor of positions with shape (batch_size, n_particles, 2).
                
        Returns:
            torch.Tensor: The computed potential for each batch, summed over all particles.
        """
        batch_size, n_particles, dims = positions.shape
        
        # Define the box length
        L = 2 * self.bound
        
        # Initialize the potential for each batch
        V_batch = torch.zeros(batch_size, device=positions.device)
        
        # Ensure centers and V0_list are on the same device as positions
        centers = self.centers.to(positions.device)
        V0_list = self.V0_list.to(positions.device)
        
        # Process each particle
        for p in range(n_particles):
            x = positions[:, p, 0]
            y = positions[:, p, 1]
            
            # Initialize potential for this particle
            V_particle = torch.zeros(batch_size, device=positions.device)
            
            # Compute contribution from each well
            for i, center in enumerate(centers):
                dx = x - center[0]
                dy = y - center[1]
                
                # Apply minimum image convention for periodic boundaries
                dx -= L * torch.round(dx / L)
                dy -= L * torch.round(dy / L)
                
                # Compute the radial distance from the well center
                r = torch.sqrt(dx**2 + dy**2)
                
                # Smooth transition using a hyperbolic tangent
                transition = 0.5 * (1 + torch.tanh(self.k * (r - self.r0)))
                
                V_particle += V0_list[i] * (1 - transition)
            
            # Add this particle's potential to the batch total
            V_batch += V_particle
        
        return V_batch
    
    def _energy(self, x):
        # Calculate the Lennard-Jones energy from the parent class
        lj_energy = super()._energy(x)
        
        # Reshape x to (batch_size, n_particles, n_dimensions)
        batch_size = x.shape[0]
        reshaped_x = x.view(batch_size, self._n_particles, self._n_dimensions)
        
        # Calculate the double well potential for all particles
        dw_energy = self.double_well_potential(reshaped_x)
        
        # Combine both energies
        total_energy = lj_energy + dw_energy
        
        return total_energy
    

