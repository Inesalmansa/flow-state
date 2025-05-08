import sys
import numpy as np
import time
from utils import get_project_root
project_root = get_project_root()
sys.path.append(project_root + "/MCMC/")
from potential import lennard_jones_energy_virial, double_well_potential


class EnergyCalculator:
    def __init__(
        self, 
        num_particles, 
        initial_particles, 
        simulation_box, 
        num_wells=0,           # Number of external potential wells (0, 1, or 2)
        V0_list=[-4.0,-4.2], 
        r0=1.0,              # radius of bottom of well
        k=10.0,                # The steepness of the transition from the flat bottom to zero potential. Default is 10.0.
        timing=True, 
        checking=False
    ):
        """
        Initialize the EnergyCalculator with optional external potential parameters.

        Parameters:
        - num_particles: Number of particles in the simulation.
        - initial_particles: Initial positions of the particles.
        - simulation_box: Object representing the simulation box (should have compute_distances method).
        - num_wells: Number of external potential wells (0, 1, or 2).
        - V0_list: Depth of the potential wells.
        - a: Width parameter for the Gaussian wells.
        - timing: If True, prints timing information.
        - checking: If True, enables additional debugging outputs.
        """
        self.sim_box = simulation_box
        self.num_particles = num_particles
        self.num_wells = num_wells
        self.V0_list = V0_list
        self.r0 = r0
        self.k = k
        self.timing = timing
        self.checking = checking
        self.particle_energy_times = []
        self.total_energy_times = []
        self.total_energy, self.total_virial = self.calculate_total_energy_virial(initial_particles)
    
    def calculate_particle_energy_virial(self, positions, particle_index):
        """
        Calculate energy of a single particle with all other particles and external potential.

        Parameters:
        - positions: Array of all particle positions.
        - particle_index: Index of the particle to calculate energy for.

        Returns:
        - particle_energy: Total energy of the particle (LJ + external).
        - particle_virial: Total virial contribution from the particle.
        """
        if self.timing:
            start_time = time.time()
        
        # Get the moved particle's position
        particle_pos = positions[particle_index]
        
        # Remove the particle's own position from the array
        other_positions = np.delete(positions, particle_index, axis=0)
        
        # Calculate distances to other particles
        r = self.sim_box.compute_distances(particle_pos, other_positions, checking=False)
        
        # Check for particles that are too close
        if np.any(r < 0.5):
            particle_energy = float('inf')
            particle_virial = float('inf')
            return particle_energy, particle_virial
        
        # Calculate Lennard-Jones pairwise energies and virials
        pair_energies, pair_virials = lennard_jones_energy_virial(
            r, epsilon=1.0, sigma=1.0, cutoff_constant=2.5, shift=True
        )
        
        # Sum all pairwise energies and virials
        particle_energy = np.sum(pair_energies)
        particle_virial = np.sum(pair_virials)
        
        # Add external potential energy if applicable
        if self.num_wells > 0:
            # Calculate external potential for this particle
            V_ext = double_well_potential(
                position=particle_pos, 
                box_size_x=self.sim_box.box_size_x,    # Assuming simulation_box has size_x
                box_size_y=self.sim_box.box_size_y,   # Assuming simulation_box has size_y
                V0_list=self.V0_list, 
                r0=self.r0,
                k=self.k, 
                num_wells=self.num_wells
            )
            particle_energy += V_ext
            # Note: Virial contribution from external potential is typically zero for fixed external fields
            # If your external potential contributes to virial, include it here accordingly
        
        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"calculate_particle_energy took {elapsed_time:.6f} seconds")
            self.particle_energy_times.append(elapsed_time)
        
        return particle_energy, particle_virial
    
    def update_total_energy_virial(self, energy_dif, virial_dif):
        """
        Update the total energy and virial with the differences.

        Parameters:
        - energy_dif: Difference in energy.
        - virial_dif: Difference in virial.
        """
        self.total_energy += energy_dif
        self.total_virial += virial_dif
    
    def calculate_total_energy_virial(self, positions):
        """
        Calculate total energy using vectorized operations, including external potential.

        Parameters:
        - positions: Array of all particle positions.

        Returns:
        - total_energy: Total potential energy of the system.
        - total_virial: Total virial of the system.
        """
        if self.timing:
            start_time = time.time()
        
        self.total_energy = 0.0
        self.total_virial = 0.0
        
        # Calculate Lennard-Jones pairwise energies and virials
        for i in range(self.num_particles - 1):
            # Get positions of all particles after i to avoid double counting
            other_positions = positions[i+1:]
    
            # Calculate distances using the simulation box's compute_distances method
            r_values = self.sim_box.compute_distances(positions[i], other_positions)
    
            # if self.checking:
            #     print("r_values = ", r_values)
    
            # Check for particles that are too close
            if np.any(r_values < 0.5):
                self.total_energy = float('inf')
                self.total_virial = float('inf')
                return self.total_energy, self.total_virial
            
            # if self.checking:
            #     for r in r_values:
            #         energy, virial = lennard_jones_energy_virial(
            #             r, epsilon=1.0, sigma=1.0, cutoff_constant=2.5, shift=True
            #         )
            #         print(f"lennard_jones_energy({r}) = {energy}, {virial}")
            
            # Calculate pairwise Lennard-Jones energies and virials
            pair_energies, pair_virials = lennard_jones_energy_virial(
                r_values, epsilon=1.0, sigma=1.0, cutoff_constant=2.5, shift=True
            )
    
            # Sum Lennard-Jones contributions
            self.total_energy += np.sum(pair_energies)
            self.total_virial += np.sum(pair_virials)
        
        # Calculate external potential energy for all particles if applicable
        if self.num_wells > 0:
            # Assuming 'positions' is a (N, 2) array where N is the number of particles
            V_ext_total = double_well_potential(
                position=positions, 
                box_size_x=self.sim_box.box_size_x,    # Assuming simulation_box has size_x
                box_size_y=self.sim_box.box_size_y,   # Assuming simulation_box has size_y
                V0_list=self.V0_list, 
                r0=self.r0,
                k=self.k, 
                num_wells=self.num_wells
            ).sum()

            # if self.checking:
            #     print("tot energy without ext pot is:", self.total_energy)
            #     print("external potential is:", V_ext_total)
            #     print("tot energy should now be", self.total_energy + V_ext_total)

            self.total_energy += V_ext_total

            # if self.checking:
            #     print("stored tot energy", self.total_energy)

            # Note: Virial contribution from external potential is typically zero for fixed external fields
            # If your external potential contributes to virial, include it here accordingly

    
        if self.timing:
            elapsed_time = time.time() - start_time
            print(f"calculate_total_energy took {elapsed_time:.6f} seconds")
            self.total_energy_times.append(elapsed_time)
        
        return self.total_energy, self.total_virial
