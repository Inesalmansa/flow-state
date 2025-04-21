import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from scipy.spatial.distance import squareform, cdist
import pandas as pd
import logging
import sys
import os
import json
from utils import (get_project_root, setup_logger, get_dataloader, calculate_well_statistics, 
                  set_icl_color_cycle, get_icl_heatmap_cmap, plot_loss, 
                  plot_frequency_heatmap, generate_samples, 
                  calculate_pair_correlation, plot_pair_correlation, save_rdf_data, plot_acceptance_rate,
                  plot_avg_free_energy, plot_well_statistics, plot_avg_x_coordinate,
                  plot_multiple_avg_x_coordinates)
import argparse
import csv

set_icl_color_cycle()
cmap_div = get_icl_heatmap_cmap("diverging")

# importing the NF and MC modules
project_root = get_project_root()
nf_path = os.path.join(project_root, "NF")
sys.path.append(nf_path)
sys.path.append(project_root)

import normflows as NF
import MCMC as MC

# EXPERIMENT VARIABLES
NUM_MC_RUNS = 100       # how many mc run running in 'parallel'
MASTER_SEED = 42        # master seed to give seeds for each mc run to run differently
NUM_PARTICLES = 3
NUM_DIM = 2

# mc needed params
TEMP = 1.0
RHO = 0.03
ASPECT_RATIO = 1.0
VISUALISE = True
CHECKING = True
EQUILIBRATION_STEPS = 5000
NUM_WELLS = 2
V0_LIST = [-10.0,-10.5]              # depth of wells
R0 = 1.2                # radius of bottom of well
K_VAL = 15              # steepness of well slopes
HALF_BOX = ((NUM_PARTICLES/RHO)**(1/NUM_DIM))/2
INITIAL_MAX_DISPLACEMENT = 0.65
SAMPLING_FREQUENCY = 150
ADJUSTING_FREQUENCY = 5000

# training parameters
TESTING = True
TOTAL_TRAINING_STEPS = 1e7
PRODUCTION_STEPS = TOTAL_TRAINING_STEPS / NUM_MC_RUNS
BIG_MOVE_ATTEMPTS = 1000
BIG_MOVE_INTERVAL = 1000

# result saving
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--experiment_id", type=str, required=True, help="Unique identifier for the experiment")
args, _ = parser.parse_known_args()
experiment_name = args.experiment_id
directory = f"/home/n2401517d/my_workspace/HMC_NF/results/{experiment_name}"
if not os.path.exists(directory):
    os.makedirs(directory)
# Setup a global experiment logger that logs to a file in the experiment directory.
experiment_log_file = os.path.join(directory, "experiment.log")
experiment_logger = setup_logger("experiment", experiment_log_file, file_level=logging.DEBUG, stream_level=logging.INFO)
experiment_logger.info("half box is: " + str(HALF_BOX))
experiment_logger.info(f"Directory created at: {directory}")

mc_runs_directory = os.path.join(directory, "mc_runs")
os.makedirs(mc_runs_directory, exist_ok=True)
training_rounds_directory = os.path.join(directory, "training_rounds")
os.makedirs(training_rounds_directory, exist_ok=True)

# Save experiment parameters to a JSON file in the parent experiment directory
experiment_params = {
    "NUM_MC_RUNS": NUM_MC_RUNS,
    "MASTER_SEED": MASTER_SEED,
    "NUM_PARTICLES": NUM_PARTICLES,
    "NUM_DIM": NUM_DIM,
    "TEMP": TEMP,
    "RHO": RHO,
    "ASPECT_RATIO": ASPECT_RATIO,
    "VISUALISE": VISUALISE,
    "CHECKING": CHECKING,
    "EQUILIBRATION_STEPS": EQUILIBRATION_STEPS,
    "NUM_WELLS": NUM_WELLS,
    "V0_LIST": V0_LIST,
    "R0": R0,
    "K_VAL": K_VAL,
    "HALF_BOX": HALF_BOX,
    "INITIAL_MAX_DISPLACEMENT": INITIAL_MAX_DISPLACEMENT,
    "SAMPLING_FREQUENCY": SAMPLING_FREQUENCY,
    "ADJUSTING_FREQUENCY": ADJUSTING_FREQUENCY,
    "TESTING": TESTING,
    "BIG_MOVE_ATTEMPTS": BIG_MOVE_ATTEMPTS,
    "BIG_MOVE_INTERVAL": BIG_MOVE_INTERVAL
}
json_file_path = os.path.join(directory, "params.json")
with open(json_file_path, "w") as f:
    json.dump(experiment_params, f, indent=4)
experiment_logger.info(f"Experiment parameters saved to {json_file_path}")

# initialisation of experiment - sets up the monte carlo runs and equilibrates and collects samples
mc_runs = [] 
for i in range(NUM_MC_RUNS):
    seed = i + MASTER_SEED
    np.random.seed(seed)

    # Create a dedicated folder and logger for this Monte Carlo run.
    run_dir = os.path.join(mc_runs_directory, f"run_{i+1:03d}")
    os.makedirs(run_dir, exist_ok=True)
    mc_log_file = os.path.join(run_dir, "mc_run.log")
    mc_logger = setup_logger(f"MC_run_{i+1:03d}", mc_log_file, file_level=logging.DEBUG, stream_level=logging.WARNING)

    # even i runs initialise left and odd initialise right
    if i % 2 == 0:
        particles, sim_box = MC.initialise_low_left(
                num_particles=NUM_PARTICLES,
                rho=RHO,
                aspect_ratio=ASPECT_RATIO,
                visualise=False,
                checking=False
            )
        experiment_logger.info(f"run {i} starts in left")
    else:
        particles, sim_box = MC.initialise_low_right(
                num_particles=NUM_PARTICLES,
                rho=RHO,
                aspect_ratio=ASPECT_RATIO,
                visualise=False,
                checking=False
            )
        experiment_logger.info(f"run {i} starts in right")
        
    mc_run = MC.MonteCarlo(
        particles=particles,
        sim_box=sim_box,
        temperature=TEMP,
        num_particles=NUM_PARTICLES,
        num_wells=NUM_WELLS,   
        V0_list=V0_LIST,                
        r0=R0,
        k=K_VAL,
        initial_max_displacement=INITIAL_MAX_DISPLACEMENT,  
        target_acceptance=0.5,
        timing=False,
        checking=CHECKING,
        logger=mc_logger,     
        seed=seed
    )

    mc_runs.append(mc_run)
    experiment_logger.info(f"Monte Carlo run {i} initialised and stored")

# Plot the potential wells for visualization
MC.visualise.plot_potential(
    box_siz_x=sim_box.box_size_x,
    box_size_y=sim_box.box_size_y,
    V0_list=V0_LIST,
    r0=R0,
    k=K_VAL,
    num_wells=NUM_WELLS,
    output_path=directory
)

experiment_logger.info("All Monte Carlo runs initialised")
experiment_logger.info("list of run instances: " + str(mc_runs))

# equilibration of each mc run
for mc_run in mc_runs:
    for step in range(1, EQUILIBRATION_STEPS + 1):
        mc_run.particle_displacement()
        if step % ADJUSTING_FREQUENCY == 0:
            mc_run.adjust_displacement()
        if step % SAMPLING_FREQUENCY == 0:
            sample = mc_run.sample(step)
            mc_run.local_samples.append(sample)
    
# Plot and save the equilibrated configuration for each MC run
for i, mc_run in enumerate(mc_runs):
    run_dir = os.path.join(mc_runs_directory, f"run_{i+1:03d}")
    fig = plt.figure(figsize=(8, 6))
    current_config = mc_run.particles
    plt.scatter(current_config[:, 0], current_config[:, 1], alpha=0.6)
    plt.xlim(0, 2*HALF_BOX)
    plt.ylim(0, 2*HALF_BOX)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$') 
    plt.title(f'Equilibrated Configuration - MC Run {i+1}')
    
    # Save plot to the MC run's directory
    config_plot_path = os.path.join(run_dir, f"equilibrated_config.png")
    fig.savefig(config_plot_path, bbox_inches='tight')
    plt.close(fig)
    
    mc_run.logger.info(f"Equilibrated configuration plot saved to: {config_plot_path}")

# Initialize MCMC step counter right after equilibration
total_mcmc_steps = 0

# collect initial set of training samples
experiment_logger.info("production runs per cycle: " + str(PRODUCTION_STEPS))
for mc_run in mc_runs:
    for step in range(1, PRODUCTION_STEPS + 1):
        mc_run.particle_displacement()
        total_mcmc_steps += 1  # Count each MCMC step
        if step % SAMPLING_FREQUENCY == 0:
            sample = mc_run.sample(step)
            mc_run.local_samples.append(sample)
            mc_run.testing_samples.append(sample[6])  # store configuration (in MC box coordinates)

free_energy_array = []

for run_idx, mc_run in enumerate(mc_runs):
    if mc_run.testing_samples:
        testing_configs = np.array(mc_run.testing_samples)  # shape: (num_samples, num_particles, 2)
        testing_configs = testing_configs[:]
        
        # Compute well statistics
        avg_x_values, p_a_values, p_b_values, deltaF_normalized_values, runs = calculate_well_statistics(
            testing_configs, 0, HALF_BOX, R0
        )

        # Add this run's free energy data to the array
        free_energy_array.append(deltaF_normalized_values)

        # Plot well statistics
        run_folder = os.path.join(mc_runs_directory, f"run_{run_idx+1:03d}")
        well_stats_path_svg, well_stats_path_png = plot_well_statistics(
            avg_x_values, 
            p_a_values, 
            p_b_values, 
            deltaF_normalized_values, 
            runs, 
            HALF_BOX,
            run_folder
        )
        
        # Plot average x coordinate for individual particles
        avg_x_path_svg, avg_x_path_png, prob_above, prob_below = plot_avg_x_coordinate(
            testing_configs,
            run_folder,
            HALF_BOX,
            run_idx+1
        )
        print(f"MC Run {run_idx+1:03d}: Probability avg_x > HALF_BOX ({HALF_BOX:.2f}) = {prob_above:.4f}, Probability avg_x < HALF_BOX = {prob_below:.4f}")

# Plot summary of the first 10 MC runs if there are at least 10
if len(mc_runs) >= 10:
    multi_avg_x_path_svg, multi_avg_x_path_png = plot_multiple_avg_x_coordinates(
        mc_runs,
        directory
    )
    experiment_logger.info(f"Summary plot of first 10 MC runs saved to: {multi_avg_x_path_svg} and {multi_avg_x_path_png}")

# Plot average free energy across all runs
free_energy_plot_path_svg, free_energy_plot_path_png, final_mean, final_sem, final_std = plot_avg_free_energy(
    free_energy_array,
    directory,
    color='C11'
)
experiment_logger.info(f"Average free energy plot saved to: {free_energy_plot_path_svg} and {free_energy_plot_path_png}")
experiment_logger.info(f"Final mean delta F = {final_mean}")
experiment_logger.info(f"Final standard error delta F = {final_sem}")
experiment_logger.info(f"Final std delta F = {final_std}")

# -------------------------
# After the full experiment, save each MC run's sampled data to a CSV file and the configurations (sample[6] for each sample) are saved as an NPY file
# -------------------------
for i, mc_run in enumerate(mc_runs):
    run_folder = os.path.join(mc_runs_directory, f"run_{i+1:03d}")
    csv_filename = os.path.join(run_folder, 'sampled_data.csv')
    print(f"Saving sampled data to {csv_filename}")
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "cycle_number",
                "energy_per_particle",
                "density",
                "pressure",
                "box_size_x",
                "box_size_y",
                "particle_configuration"
            ])
            
            for data in mc_run.local_samples:
                # Expecting that each sample is structured as:
                # (cycle_number, energy_per_particle, density, pressure, box_size_x, box_size_y, particles)
                cycle_number, energy_per_particle, density, pressure, box_size_x, box_size_y, particles = data
                
                # Convert the particle configuration to a NumPy array and flatten it.
                particles_flat = np.array(particles).flatten()
                
                writer.writerow([
                    cycle_number,
                    energy_per_particle,
                    density,
                    pressure,
                    box_size_x,
                    box_size_y,
                    particles_flat.tolist()  # storing as a JSON-like list in the CSV
                ])
        experiment_logger.info(f"MC run {i+1} sampled data successfully saved to {csv_filename}")
    except Exception as e:
        experiment_logger.error(f"Error: Failed to save sampled data for MC run {i+1} to {csv_filename}. Error: {e}")

    # Extract and save local samples configurations
    configs = np.array([sample[6] for sample in mc_run.local_samples])
    configs_filepath = os.path.join(run_folder, "mc_run_configs.npy")
    np.save(configs_filepath, configs)
    experiment_logger.info(f"MC run {i+1} local config data saved to: {configs_filepath}")

    # Extract and save testing samples configurations
    testing_configs = np.array(mc_run.testing_samples)
    testing_configs_filepath = os.path.join(run_folder, "mc_run_testing_configs.npy") 
    np.save(testing_configs_filepath, testing_configs)
    experiment_logger.info(f"MC run {i+1} testing config data saved to: {testing_configs_filepath}")


