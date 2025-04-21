<!-- File: README.md -->

# Moving in the Flow State  
Hybrid NF–MCMC for a 2D NVT Lennard‐Jones Double Well System

This repository contains the implementation of hybrid algorithms combining Normalizing Flows (NF) with Monte Carlo Markov Chain (MCMC) methods for enhanced sampling of a 2D NVT Lennard‐Jones system with a double well potential.

---

## Repository Structure

- **MCMC/**: Custom Monte Carlo implementation for molecular systems  
- **NF/**: Modified normflows package with adaptations for molecular systems  
- **hybrid_NF_MCMC/**: Hybrid algorithms combining NF with MCMC  
  - `main_algorithm_1.py`: Pre-training approach: Initial training followed by NF‐guided MCMC  
  - `main_algorithm_2.py`: On-the-fly training approach: Iterative training with periodic model updates  
  - `main_mcmc_only.py`: Standard MCMC implementation (baseline)

---

## Key Features

- Double well Lennard‐Jones potential for molecular simulations  
- Custom MCMC framework with adaptive displacement  
- Modified normalizing‐flow architecture optimised for molecular configurations  
- Hybrid algorithms that leverage deep learning for enhanced sampling  
- Comprehensive analysis tools for molecular trajectories  

---

## Installation

```bash
git clone https://github.com/yourusername/flow_state.git
cd flow_state
pip install -r requirements.txt
```

## Usage 
```python
# Run baseline MCMC
python hybrid_NF_MCMC/main_mcmc_only.py --experiment_id "mcmc_baseline"

# Run Algorithm 1 (train once, then sample)
python hybrid_NF_MCMC/main_algorithm_1.py --experiment_id "hybrid_alg1"

# Run Algorithm 2 (iterative training)
python hybrid_NF_MCMC/main_algorithm_2.py --experiment_id "hybrid_alg2"
```

## Visualisation
All plots are generated using LaTeX text style with Imperial College London branding colours.

## Citation
If you use this code in this work please cite:
```@article{Stimper2023, 
  author    = {Vincent Stimper and David Liu and Andrew Campbell and Vincent Berenz and Lukas Ryll and Bernhard Schölkopf and José Miguel Hernández-Lobato}, 
  title     = {normflows: A PyTorch Package for Normalizing Flows}, 
  journal   = {Journal of Open Source Software}, 
  volume    = {8},
  number    = {86}, 
  pages     = {5361}, 
  publisher = {The Open Journal}, 
  doi       = {10.21105/joss.05361}, 
  url       = {https://doi.org/10.21105/joss.05361}, 
  year      = {2023}
