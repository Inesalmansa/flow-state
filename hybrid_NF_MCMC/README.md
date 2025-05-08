<!-- File: hybrid_NF_MCMC/README.md -->

# Hybrid NF–MCMC Algorithms

Implementation of hybrid algorithms combining Normalizing Flows (NF) with MCMC methods to improve sampling efficiency.

---

## Overview

Three modes of operation:

1. **Algorithm 1** (`main_algorithm_1.py`)  
   - Initial MCMC sampling → train NF → NF‐guided “big move” proposals  
2. **Algorithm 2** (`main_algorithm_2.py`)  
   - Iterative loop: sample → retrain NF periodically → continue sampling  
3. **Baseline MCMC** (`main_mcmc_only.py`)  
   - Standard Metropolis–Hastings for comparison  

---

## Key Features

- Configurable sampling and training schedules  
- Dynamic acceptance‐rate adjustment  
- Detailed logging and visualization  
- Well‐statistics and energy landscape analysis  

---

## Algorithm 1 vs Algorithm 2

| Feature              | Algorithm 1                   | Algorithm 2                            |
|----------------------|-------------------------------|----------------------------------------|
| Training schedule    | Single pre‐training phase     | Periodic re‐training during sampling   |
| Adaptivity           | Fixed model after training    | Continually adapts to new samples      |
| Complexity           | Simpler, one‐off training     | More complex, iterative workflow       |

---

## Usage

```bash
# Algorithm 1
python main_algorithm_1.py --experiment_id "hybrid_alg1"

# Algorithm 2
python main_algorithm_2.py --experiment_id "hybrid_alg2"

# Baseline MCMC only
python main_mcmc_only.py --experiment_id "mcmc_baseline"
