
```markdown
<!-- File: NF/README.md -->

# NF Module

Modified version of the [normflows](https://github.com/VincentStimper/normalizing-flows) package, adapted for molecular systems.

---

## Overview

Primary modifications to support molecular sampling:

1. Double well potential in `SimpleLJ` energy function  
2. Negative log‐likelihood for the Uniform base distribution  
3. Circular spline flows for periodic boundary conditions  
4. API hooks for MCMC integration  

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