<!-- File: MCMC/README.md -->

# MCMC Module

A custom Markov Chain Monte Carlo simulation framework for particle systems, with implementation of the Lennard‐Jones potential and a double well external potential.

---

## Overview

This module implements a Metropolis sampling algorithm tailored to:

- Double well Lennard‐Jones potential energy  
- Adaptive maximum displacement scheme within equilibration to reach target acceptance rates 
- Periodic boundary conditions  
- Integration of NF “global‐move” proposals with the Metropolis-Hastings criterion

---

## Key Components

- **`monte_carlo.py`**: Core Monte Carlo implementation  
- **`energy_calculator.py`**: Energy computations  
- **`simulation_box.py`**: Simulation domain and boundary handling  
- **`visualise.py`**: Trajectory and state‐space visualisation  

---

## Usage

```python
from MCMC import MonteCarlo, SimulationBox

num_particles = 3
rho = 0.03
aspect_ratio = 1
visualise = True
checking = True

# initialising particles in the left well
particles, sim_box = initialise_low_left(
                num_particles=num_particles,
                rho=rho,
                aspect_ratio=aspect_ratio,
                visualise=visualise,
                checking=checking
            )

# Initialise Monte Carlo sampler
mc = MonteCarlo(
    positions= 
    n_particles=num_particles,
    n_dim=2,
    temperature=1.0,
    rho=rho,
    half_box=5.0,
    seed=42,
    sim_box=sim_box
)

# Equilibration
for step in range(5000):
    mc.particle_displacement()
    if step % 1000 == 0:
        mc.adjust_displacement()

# Production run
samples = []
for step in range(10000):
    mc.particle_displacement()
    if step % 100 == 0:
        samples.append(mc.sample(step))
