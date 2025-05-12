<!-- File: NF/README.md -->

# NF Module

Modified version of the [normflows](https://github.com/VincentStimper/normalizing-flows) package, adapted for molecular systems.

---

## Overview

Primary modifications to support molecular sampling:

1. Double well potential and Lennard Jones potential within `SimpleLJ` energy function  
2. Uniform base distribution including Negative log‐likelihood for train-by-energy

---

## Key Components

- **`normflows/Energy/SimpleLJ.py`**: Extended with double well LJ potential  
- **`normflows/distributions/base.py`**: Modified UniformParticle distribution  
- **`Normalizing_flow_npz_data.py`**: Data utilities for pre‐generated trajectories  

---

## Usage

```python
import torch
import normflows as nf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = nf.Energy.UniformParticle(n_particles=3, n_dim=2, half_box=10.0, device=device)
flow_layers = [
    nf.flows.CircularCoupledRationalQuadraticSpline(
        n_particles*2, n_blocks=8, hidden_units=256, n_bins=32, tail_bound=10.0
    )
    for _ in range(15)
]
model = nf.NormalizingFlow(base, flow_layers).to(device)

# Train the model on sampled data, then sample:
#   loss = -model.log_prob(batch)
#   model.backward_step(loss)
#   samples = model.sample(num_samples)
```

## Citation
The code for this normalizing flow package comes from:

```
@article{Stimper2023, 
  author = {Vincent Stimper and David Liu and Andrew Campbell and Vincent Berenz and Lukas Ryll and Bernhard Schölkopf and José Miguel Hernández-Lobato}, 
  title = {normflows: A PyTorch Package for Normalizing Flows}, 
  journal = {Journal of Open Source Software}, 
  volume = {8},
  number = {86}, 
  pages = {5361}, 
  publisher = {The Open Journal}, 
  doi = {10.21105/joss.05361}, 
  url = {https://doi.org/10.21105/joss.05361}, 
  year = {2023}
} 
```