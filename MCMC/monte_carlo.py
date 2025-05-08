import numpy as np
import time
import logging  # in case logging is used
import torch

from simulation_box import SimulationBox
from energy_calculator import EnergyCalculator


# Edited for NVT and external double well potential
class MonteCarlo:
    def __init__(
        self,
        particles,
        sim_box,
        temperature,
        num_particles,
        num_wells=0,                # Number of external potential wells (0, 1, or 2)
        V0_list=[-0.5,-0.5],                    # Depth of the potential wells
        r0=1.0,
        k=10,                      # Width parameter for the Gaussian wells
        initial_max_displacement=0.5,
        target_acceptance=0.5,
        timing=False,
        checking=False,
        logger=None,              # New logger argument (if None, prints are used)
        seed=None,
        device=None
    ):
        """
        NVT Monte Carlo class with optional external double well potential.

        Parameters
        ----------
        particles : np.ndarray
            Initial array of particle positions, shape (N, 2) in 2D.
        sim_box : SimulationBox
            Simulation box with fixed volume (area in 2D).
        temperature : float
            Temperature of the system (k_B T in reduced units).
        num_particles : int
            Number of particles in the system.
        num_wells : int, optional
            Number of external potential wells (0, 1, or 2). Default is 0.
        V0_list : list, optional
            Depth of the potential wells. Default is [-0.5,-0.5].
        r0 : float, optional
            Parameter for the potential wells. Default is 1.0.
        k : int, optional
            Width parameter for the Gaussian wells. Default is 10.
        initial_max_displacement : float, optional
            Initial maximum displacement for particle moves. Default is 0.5.
        target_acceptance : float, optional
            Target acceptance ratio for adjusting max displacement. Default is 0.5.
        timing : bool, optional
            If True, measure and log timing information. Default is False.
        checking : bool, optional
            If True, log detailed debug info. Default is False.
        logger : logging.Logger or None, optional
            Logger instance to use for logging messages. If None, messages are simply printed.
        seed : int or None, optional
            Random seed for reproducibility.
        """
        self.particles = particles
        self.sim_box = sim_box
        self.half_width = sim_box.box_size_x / 2
        self.beta = 1.0 / temperature
        self.num_particles = num_particles

        # External potential parameters
        self.num_wells = num_wells
        self.V0_list = V0_list
        self.r0 = r0
        self.k = k

        self.max_displacement = initial_max_displacement
        self.target_acceptance = target_acceptance

        # Counters for displacement attempts
        self.attempts_displacement = 0
        self.accepted_displacement = 0
        self.previous_attempts_displacement = 0
        self.previous_accepted_displacement = 0

        self.timing = timing
        self.checking = checking
        self.debug = True  # Used in check_equilibration

        # Logger setup: if one is provided, use it.
        self.logger = logger

        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng()

        # Energy calculator with external potential parameters
        self.energy_calculator = EnergyCalculator(
            num_particles=num_particles,
            initial_particles=particles,
            simulation_box=self.sim_box,
            num_wells=self.num_wells,   # Pass num_wells
            V0_list=self.V0_list,                 # Pass V0_list
            r0=self.r0,                 # Pass r0
            k=self.k,                   # Pass k
            timing=timing,
            checking=True               # Ensure checking is passed correctly
        )

        # Histories for analysis (pressure, volume, densities, etc.)
        self.pressure_history = []
        self.volume_history = [self.sim_box.volume]
        self.densities = []

        self.running_mean_window = 1000
        self.particle_move_times = []
        self.last_configuration = [
            "left" if pos[0] < self.half_width else "right" for pos in self.particles
        ]
        self.local_samples = []
        self.testing_samples = []
        self.nf_model = None 

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _log(self, message, level="info"):
        """
        Internal logging method.
        If a logger is provided, use its logging methods; otherwise, fall back to print.
        """
        if self.logger:
            if level == "debug":
                self.logger.debug(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
            else:
                self.logger.info(message)
        else:
            print(message)

    def particle_displacement(self):
        """
        Attempt to displace a randomly selected particle using Metropolis acceptance.
        """
        self.attempts_displacement += 1

        # Select a random particle
        p = self.rng.integers(self.num_particles)
        if self.attempts_displacement < 5 and self.checking:
            self._log(f"Randomly chosen particle index = {p}", level="debug")

        # Energy/virial of the particle before the move
        eno, viro = self.energy_calculator.calculate_particle_energy_virial(self.particles, p)

        # Random displacement in 2D
        displacement = (self.rng.random(2) - 0.5) * self.max_displacement
        new_positions = self.particles.copy()
        new_positions[p] += displacement

        # Apply periodic boundary conditions (rectangular or square)
        new_positions[p] = self.sim_box.apply_pbc(new_positions[p])

        # Energy/virial of the particle after the move
        enn, virn = self.energy_calculator.calculate_particle_energy_virial(new_positions, p)

        delta_energy = enn - eno
        delta_virial = virn - viro

        # Metropolis acceptance
        if self.metropolis_acceptance_particle_move(eno, enn):
            # Accept the move
            self.particles = new_positions
            self.accepted_displacement += 1
            self.energy_calculator.update_total_energy_virial(delta_energy, delta_virial)

            # --- New configuration check for left/right crossing ---
            new_side = "left" if self.particles[p][0] < self.half_width else "right"
            if new_side != self.last_configuration[p]:
                self._log(
                    f"Particle {p} has crossed from {self.last_configuration[p]} to {new_side}. "
                    f"New configuration: {[ 'left' if pos[0] < self.half_width else 'right' for pos in self.particles ]}",
                    level="info"
                )
                self.last_configuration[p] = new_side
    
    def metropolis_acceptance_particle_move(self, old_energy, new_energy):
        """
        Return True if move is accepted, False otherwise (standard Metropolis criterion).
        """
        if self.timing:
            start_time = time.time()

        if new_energy <= old_energy:
            if self.timing:
                elapsed_time = time.time() - start_time
                self._log(f"metropolis_criterion_accepted took {elapsed_time:.6f} seconds", level="debug")
            return True
        else:
            if np.isinf(new_energy):
                if self.checking:
                    self._log("Attempted move leads to infinite energy; rejecting.", level="debug")
                if self.timing:
                    elapsed_time = time.time() - start_time
                    self._log(f"metropolis_criterion_rejected took {elapsed_time:.6f} seconds", level="debug")
                return False

            try:
                boltzmann_factor = np.exp(-self.beta * (new_energy - old_energy))
            except OverflowError:
                boltzmann_factor = 0.0

            criterion = self.rng.random() < boltzmann_factor            

            if self.timing:
                elapsed_time = time.time() - start_time
                self._log(f"metropolis_criterion_evaluated took {elapsed_time:.6f} seconds", level="debug")

            return criterion
        
    # -----------------------------------------------------------------
    #                         HYBRID MC - NF
    # -----------------------------------------------------------------

    def set_nf_model(self, nf_model):
        """
        Set the trained normalizing flow model for use during testing.
        """
        self.nf_model = nf_model

    def nf_big_move(self, config):
        """
        Attempt to accept a configuration from the Normalizing Flow
        using the flow-augmented Metropolis criterion.
        """
        self.attempts_displacement += 1

        # -- 1) Energies (old and new)
        eno = self.energy_calculator.total_energy
        viro = self.energy_calculator.total_virial

        new_positions = config
        enn, virn = self.energy_calculator.calculate_total_energy_virial(new_positions)

        # -- 2) Negative log-likelihoods (old and new)
        # Convert from MC box coords to the NF's "centered" coords
        old_positions_centered = self.particles - np.array([self.half_width, self.half_width])
        new_positions_centered = new_positions - np.array([self.half_width, self.half_width])

        # Reshape into (batch=1, dimension) so log_prob can read it
        old_torch = torch.tensor(old_positions_centered.reshape(1, -1),
                                dtype=torch.float, device=self.device)
        new_torch = torch.tensor(new_positions_centered.reshape(1, -1),
                                dtype=torch.float, device=self.device)

        # model.log_prob returns log(q(x)), so negative log-likelihood is -log_prob
        old_nll = - self.nf_model.log_prob(old_torch).item()
        new_nll = - self.nf_model.log_prob(new_torch).item()

        # -- 3) Acceptance ratio in log space
        delta_energy = enn - eno
        delta_nll = new_nll - old_nll

        ratio_log = -self.beta * delta_energy - delta_nll
        ratio = np.exp(ratio_log)

        # Debugging for split particles
        if self.checking:
            left_count = sum(pos[0] < self.half_width for pos in self.particles)
            right_count = self.num_particles - left_count
            if left_count > 0 and right_count > 0:
                self._log(f"Split configuration detected: {left_count} left, {right_count} right", level="debug")
                self._log(f"Old energy: {eno}, New energy: {enn}", level="debug")
                self._log(f"Old NLL: {old_nll}, New NLL: {new_nll}", level="debug")
                self._log(f"Acceptance criterion (beta * delta U): {np.exp(-self.beta * delta_energy)}", level="debug")
                self._log(f"Factor of delta NLL: {np.exp(delta_nll)}", level="debug")
                self._log(f"Final acceptance ratio: {ratio}", level="debug")

        # -- 4) Metropolis accept/reject
        if ratio >= 1.0:
            accept = True
        else:
            accept = (self.rng.random() < ratio)

        # Log whether the move is accepted
        if self.checking:
            if left_count > 0 and right_count > 0:
                self._log(f"Move accepted: {accept}", level="debug")

        # -- 5) If accepted, store new positions
        if accept:
            self.particles = new_positions
            self.accepted_displacement += 1
            # Optionally update "last_configuration" if you track well-crossings
        else:
            # Revert to old energy
            self.energy_calculator.calculate_total_energy_virial(self.particles)

        return accept

    def judge_normalizing_flow(self, config):
        """
        Returns the metorpolis criterion for a run - but doesn't accept the new configuration
        """
        self.attempts_displacement += 1

        # Old energy before the proposed move
        eno = self.energy_calculator.total_energy
        viro = self.energy_calculator.total_virial

        if self.checking:
            self._log(f"old energy: {eno}", level="debug")
        
        new_positions = config
        enn, virn = self.energy_calculator.calculate_total_energy_virial(new_positions)

        delta_energy = enn - eno
        delta_virial = virn - viro

        criterion = self.metropolis_acceptance_particle_move(eno, enn)

        self.energy_calculator.total_energy = eno
        self.energy_calculator.total_virial = viro

        return criterion
    
    def bulk_judge_normalizing_flow(self, configs, ref_energy):
        """
        Evaluate a batch of configurations against a reference energy using the Metropolis criterion.

        Each configuration in 'configs' is checked by comparing its total energy against the provided 
        reference energy (ref_energy), rather than the current state of the simulation. This is intended
        to hypothetically assess the acceptance of moves (for example, moves generated by a Normalizing Flow).

        Parameters
        ----------
        configs : list of np.ndarray
            A list of proposed particle configurations.
        ref_energy : float
            The reference total energy to use in the Metropolis acceptance test.

        Returns
        -------
        accepted_moves : int
            The number of configurations that would be accepted based on the Metropolis criterion.
        attempted_moves : int
            The total number of configurations (moves) attempted.
        """
        accepted_moves = 0
        attempted_moves = len(configs)

        for config in configs:
            # Calculate the total energy of the proposed configuration.
            new_energy, _ = self.energy_calculator.calculate_total_energy_virial(config)

            # Use ref_energy as the 'old' energy in the Metropolis criterion.
            if self.metropolis_acceptance_particle_move(ref_energy, new_energy):
                accepted_moves += 1

        # Log the results.
        self._log(
            f"Bulk judge normalizing flow: {accepted_moves} accepted moves out of {attempted_moves} attempted moves (reference energy: {ref_energy:.3f}).",
            level="info"
        )

        return accepted_moves, attempted_moves

    # -----------------------------------------------------------------
    #                      Adjustments (Displacements)
    # -----------------------------------------------------------------
    def adjust_displacement(self):
        """
        Adjust the maximum displacement to target a desired acceptance ratio.
        """
        if self.attempts_displacement > self.previous_attempts_displacement:
            delta_attempts = self.attempts_displacement - self.previous_attempts_displacement
            delta_accepted = self.accepted_displacement - self.previous_accepted_displacement
            frac = delta_accepted / delta_attempts if delta_attempts > 0 else 0

            adjustment_factor = frac / self.target_acceptance
            new_max_displacement = self.max_displacement * adjustment_factor

            ratio = new_max_displacement / self.max_displacement
            if ratio > 1.5:
                new_max_displacement = self.max_displacement * 1.5
            elif ratio < 0.5:
                new_max_displacement = self.max_displacement * 0.5

            if self.checking:
                self._log(
                    f"Adjusting max displacement from {self.max_displacement:.3f} to {new_max_displacement:.3f}, "
                    f"acceptance fraction in last block = {frac:.3f}",
                    level="debug"
                )

            self.max_displacement = new_max_displacement

            self.previous_attempts_displacement = self.attempts_displacement
            self.previous_accepted_displacement = self.accepted_displacement

    def adjust_volume(self):
        """
        Not needed for NVT. Stub method.
        """
        if self.checking:
            self._log("Adjust volume called in NVT ensemble. No action taken.", level="debug")
        return

    # -----------------------------------------------------------------
    #                            Sampling
    # -----------------------------------------------------------------
    def sample(self, cycle_number):
        """
        Sample and log the energy per particle, pressure, and density at the current cycle.
        """
        energy_per_particle = self.energy_calculator.total_energy / self.num_particles
        volume = self.sim_box.volume
        density = self.num_particles / volume

        pressure = density / self.beta + self.energy_calculator.total_virial / (2.0 * volume)

        if self.checking:
            self._log(
                f"Sampling at cycle {cycle_number}: E/N={energy_per_particle:.3f}, P={pressure:.3f}, rho={density:.3f}",
                level="debug"
            )

        self.pressure_history.append(pressure)
        self.volume_history.append(volume)
        self.densities.append(density)

        return (
            cycle_number,
            energy_per_particle,
            density,
            pressure,
            self.sim_box.box_size_x,
            self.sim_box.box_size_y,
            self.particles.copy()
        )

    # -----------------------------------------------------------------
    #                   Equilibration Check (Optional)
    # -----------------------------------------------------------------
    def check_equilibration(self, tolerance=0.05, window=500):
        """
        In an NVT ensemble, check for steady behavior in observables.
        """
        if len(self.pressure_history) < window:
            return False

        recent_pressures = self.pressure_history[-window:]
        mean_pressure = np.mean(recent_pressures)
        pressure_std = np.std(recent_pressures)

        recent_densities = self.densities[-window:]
        mean_density = np.mean(recent_densities)
        density_std = np.std(recent_densities)

        conditions = [
            (pressure_std / mean_pressure < tolerance) if mean_pressure != 0 else False,
            (density_std / mean_density < tolerance) if mean_density != 0 else False,
        ]

        if self.debug:
            self._log("Equilibration check (NVT):", level="debug")
            self._log(f"  Pressure: {mean_pressure:.3f} ± {pressure_std:.3f}", level="debug")
            self._log(f"  Density: {mean_density:.3f} ± {density_std:.3f}", level="debug")
            self._log(f"  Conditions met: {sum(conditions)}/{len(conditions)}", level="debug")

        return all(conditions)
    
    # -----------------------------------------------------------------
    #                    Old NF acceptance defs
    # -----------------------------------------------------------------
    def nf_big_move_OLD(self, config):
        """
        Attempt to accept a configuration (e.g. from a training Normalizing Flow).
        """
        self.attempts_displacement += 1

        # Old energy before the proposed move
        eno = self.energy_calculator.total_energy
        viro = self.energy_calculator.total_virial

        # if self.checking:
        #     self._log(f"old energy: {eno}", level="debug")
        
        new_positions = config
        enn, virn = self.energy_calculator.calculate_total_energy_virial(new_positions)

        delta_energy = enn - eno
        delta_virial = virn - viro

        criterion = self.metropolis_acceptance_particle_move(eno, enn)

        if criterion:
            self.particles = new_positions
            self.accepted_displacement += 1
            
            # New configuration check for left/right crossing for all particles
            for i, pos in enumerate(self.particles):
                new_side = "left" if pos[0] < self.half_width else "right"
                if new_side != self.last_configuration[i]:
                    self._log(
                        f"Particle {i} has crossed from {self.last_configuration[i]} to {new_side}. "
                        f"New configuration: {[ 'left' if p[0] < self.half_width else 'right' for p in self.particles ]}",
                        level="info"
                    )
                    self.last_configuration[i] = new_side
        else:
            # if self.checking:
            #     self._log("move rejected, going back to og energy", level="debug")
            self.energy_calculator.calculate_total_energy_virial(self.particles)
        
        if self.checking:
            self._log(f"old energy: {eno}", level="debug")
            self._log(f"new_energy_stored: {self.energy_calculator.total_energy}", level="debug")
            self._log(f"calculated energy of proposed config: {enn}", level="debug")
        
        return criterion

    def judge_normalizing_flow_OLD(self, config):
        """
        Returns the metorpolis criterion for a run - but doesn't accept the new configuration
        """
        self.attempts_displacement += 1

        # Old energy before the proposed move
        eno = self.energy_calculator.total_energy
        viro = self.energy_calculator.total_virial

        if self.checking:
            self._log(f"old energy: {eno}", level="debug")
        
        new_positions = config
        enn, virn = self.energy_calculator.calculate_total_energy_virial(new_positions)

        delta_energy = enn - eno
        delta_virial = virn - viro

        criterion = self.metropolis_acceptance_particle_move(eno, enn)

        self.energy_calculator.total_energy = eno
        self.energy_calculator.total_virial = viro

        return criterion
    
    def bulk_judge_normalizing_flow_OLD(self, configs, ref_energy):
        """
        Evaluate a batch of configurations against a reference energy using the Metropolis criterion.

        Each configuration in 'configs' is checked by comparing its total energy against the provided 
        reference energy (ref_energy), rather than the current state of the simulation. This is intended
        to hypothetically assess the acceptance of moves (for example, moves generated by a Normalizing Flow).

        Parameters
        ----------
        configs : list of np.ndarray
            A list of proposed particle configurations.
        ref_energy : float
            The reference total energy to use in the Metropolis acceptance test.

        Returns
        -------
        accepted_moves : int
            The number of configurations that would be accepted based on the Metropolis criterion.
        attempted_moves : int
            The total number of configurations (moves) attempted.
        """
        accepted_moves = 0
        attempted_moves = len(configs)

        for config in configs:
            # Calculate the total energy of the proposed configuration.
            new_energy, _ = self.energy_calculator.calculate_total_energy_virial(config)

            # Use ref_energy as the 'old' energy in the Metropolis criterion.
            if self.metropolis_acceptance_particle_move(ref_energy, new_energy):
                accepted_moves += 1

        # Log the results.
        self._log(
            f"Bulk judge normalizing flow: {accepted_moves} accepted moves out of {attempted_moves} attempted moves (reference energy: {ref_energy:.3f}).",
            level="info"
        )

        return accepted_moves, attempted_moves
    