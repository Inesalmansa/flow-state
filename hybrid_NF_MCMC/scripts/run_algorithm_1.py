#!/usr/bin/env python
import os
import json
import subprocess

# -----------------------------------------------------------------------------
# Define the parameters for the HPC job and experiment
# -----------------------------------------------------------------------------
parameters = {
    "experiment_id": "algo_1_premade_102400_samples_dV_0.0_samp_fr_150_other_gpu",  # experiment identifier
    "compute_type": "gpu",                   # or "cpu" if running on CPU queue
    "walltime": "24:00:00"                   # maximum allowed wall time for the job
}

# -----------------------------------------------------------------------------
# Create a dedicated directory for the experiment results
# -----------------------------------------------------------------------------
results_dir = f"/home/n2401517d/my_workspace/flow_state/hybrid_NF_MCMC/results/{parameters['experiment_id']}"
os.makedirs(results_dir, exist_ok=True)

# Save experiment parameters to a JSON file in the results directory
parameters_file = os.path.join(results_dir, "pbs_params.json")
with open(parameters_file, "w") as f:
    json.dump(parameters, f, indent=4)

# -----------------------------------------------------------------------------
# Create the PBS script content
# -----------------------------------------------------------------------------
pbs_script_content = f"""#!/bin/bash
#PBS -N {parameters['experiment_id']}
"""

if parameters["compute_type"] == "gpu":
    pbs_script_content += """#PBS -P scbe_r.ni
#PBS -q gpu_v100
#PBS -l select=1:ncpus=4:ngpus=1
"""
elif parameters["compute_type"] == "cpu":
    pbs_script_content += """#PBS -P scbe_r.ni
#PBS -q qintel_wfly
#PBS -l select=1:ncpus=4:mem=16gb
"""

pbs_script_content += f"""#PBS -j oe
#PBS -o {results_dir}/{parameters['experiment_id']}.out
#PBS -l walltime={parameters['walltime']}

# Create the results directory on the compute node (in case it doesn't exist)
mkdir -p {results_dir}

# Change directory to where the algorithm script is located and run it
cd /home/n2401517d/my_workspace/flow_state/hybrid_NF_MCMC/
python -u main_algorithm_1.py \
    --experiment_id {parameters['experiment_id']} > {results_dir}/{parameters['experiment_id']}_prints.out 2>&1
"""

# Write the PBS script to a file in the results directory
pbs_script_file = os.path.join(results_dir, "submit_experiment.pbs")
with open(pbs_script_file, "w") as f:
    f.write(pbs_script_content)

print(f"PBS script written to {pbs_script_file}")
print(f"Parameters saved to {parameters_file}")

# -----------------------------------------------------------------------------
# Execute the PBS script using qsub
# -----------------------------------------------------------------------------
result = subprocess.run(["qsub", pbs_script_file], capture_output=True, text=True)

if result.returncode == 0:
    print(f"Job {parameters['experiment_id']} submitted successfully with ID: {result.stdout.strip()}")
else:
    print(f"Job submission failed for {parameters['experiment_id']}! Error: {result.stderr}") 