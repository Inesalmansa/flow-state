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
NUM_MC_RUNS = 100        # how many mc run running in 'parallel'
MASTER_SEED = 42        # master seed to give seeds for each mc run to run differently
NUM_PARTICLES = 3
NUM_DIM = 2
NUM_TRAINING_CYCLES = 0

# mc needed params
TEMP = 1.0
RHO = 0.03
ASPECT_RATIO = 1.0
VISUALISE = True
CHECKING = True
EQUILIBRATION_STEPS = 5000
NUM_WELLS = 2
V0_LIST = [-10.0,-10.0]              # depth of wells
R0 = 1.2                # radius of bottom of well
K_VAL = 15              # steepness of well slopes
HALF_BOX = ((NUM_PARTICLES/RHO)**(1/NUM_DIM))/2
INITIAL_MAX_DISPLACEMENT = 0.65
SAMPLING_FREQUENCY = 150
ADJUSTING_FREQUENCY = 5000

# nf needed params
INITIAL_TRAINING_NUM_SAMPLES = 102400
BATCH_SIZE = 512
K = 15
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 0
HIDDEN_UNITS = 256
NUM_BINS = 32
N_BLOCKS = 8

# post training runs parameters
TESTING = True
BIG_MOVE_ATTEMPTS = 1000
BIG_MOVE_INTERVAL = 1000
NUM_SAMPLES_FOR_ANALYSIS = BIG_MOVE_ATTEMPTS * NUM_MC_RUNS

# experiment_name = 'algo_1_premade_102400_samples_dV_0.0_samp_fr_150'

# result saving
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--experiment_id", type=str, required=True, help="Unique identifier for the experiment")
args, _ = parser.parse_known_args()
experiment_name = args.experiment_id

directory = os.path.join(get_project_root(), "results", experiment_name)
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
    "NUM_TRAINING_CYCLES": NUM_TRAINING_CYCLES,
    "INITIAL_TRAINING_NUM_SAMPLES": INITIAL_TRAINING_NUM_SAMPLES,
    "BATCH_SIZE": BATCH_SIZE,
    "K": K,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "HIDDEN_UNITS": HIDDEN_UNITS,
    "NUM_BINS": NUM_BINS,
    "N_BLOCKS": N_BLOCKS,
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
    "BIG_MOVE_INTERVAL": BIG_MOVE_INTERVAL,
    "NUM_SAMPLES_FOR_ANALYSIS": NUM_SAMPLES_FOR_ANALYSIS
}
json_file_path = os.path.join(directory, "params.json")
with open(json_file_path, "w") as f:
    json.dump(experiment_params, f, indent=4)
experiment_logger.info(f"Experiment parameters saved to {json_file_path}")

# initialisation of experiment - sets up the monte carlo runs and equilibrates and collects samples
mc_runs = []  # list to store each Monte Carlo simulation instance
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
        num_wells=NUM_WELLS,    # Pass number of wells
        V0_list=V0_LIST,                  # Pass well depth
        r0=R0,
        k=K_VAL,
        initial_max_displacement=INITIAL_MAX_DISPLACEMENT,  # Adjust as needed
        target_acceptance=0.5,
        timing=False,
        checking=CHECKING,
        logger=mc_logger,       # Each run now logs to its own file
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

# Initialize acceptance rate tracking at the very beginning
p_acc_history = [0.0]  # Initial acceptance rate is 0
mcmc_steps_history = [0]  # Start from MCMC step 0
big_move_attempts = 0
big_move_accepts = 0

# prepping initial samples as for dataloader
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
experiment_logger.info('device is: ' + str(device))

num_samples = 102400
data_path = "/home/n2401517d/my_workspace/flow_state/NF/data/samples_N3_rho_0.03.npz"

# IMPORT DATA
npz_data = np.load(data_path)
df = npz_data[npz_data.files[0]]
# Get unique samples
unique_data = np.unique(df, axis=0)
available_samples = unique_data.shape[0]
print("\nTotal unique samples available:", available_samples)

# Use the entire dataset if num_samples is not provided
if num_samples is None:
        num_samples = available_samples
        print(f"No num_samples specified. Using all available samples: {num_samples}")
elif num_samples > available_samples:
        raise ValueError(f"Error: Requested number of samples ({num_samples}) exceeds available unique samples ({available_samples}).")

# Randomly select the desired number of samples
indices = np.random.choice(available_samples, num_samples, replace=False)
global_samples_nf = unique_data[indices]
global_samples_nf = global_samples_nf.reshape(num_samples, NUM_PARTICLES * NUM_DIM)
print("Flattened global_samples_nf shape:", global_samples_nf.shape)
print('Data Loaded!')

unique_data = np.unique(global_samples_nf, axis=0)
experiment_logger.info("Total unique samples: " + str(unique_data.shape[0]))

dataloader = get_dataloader(global_samples_nf, NUM_PARTICLES, NUM_DIM, device, batch_size=BATCH_SIZE, shuffle=True)
experiment_logger.info("DataLoader details:")
experiment_logger.info("Dataset size: " + str(len(dataloader.dataset)))
experiment_logger.info("Number of batches: " + str(len(dataloader)))
first_batch = next(iter(dataloader))
experiment_logger.info("Shape of data in first batch: " + str(first_batch[0].shape))

# initialise the normalizing flow model
bound = HALF_BOX
base = NF.Energy.UniformParticle(NUM_PARTICLES, NUM_DIM, bound, device=device)
# target = NF.Energy.SimpleLJ(NUM_PARTICLES * NUM_DIM, NUM_PARTICLES, 1, bound)    # not used for now
# K = 15 - now defined in the argument of function
flow_layers = []
for i in range(K):
    flow_layers += [NF.flows.CircularCoupledRationalQuadraticSpline(NUM_PARTICLES * NUM_DIM, NUM_BINS, HIDDEN_UNITS,
                    range(NUM_PARTICLES * NUM_DIM), num_bins=NUM_BINS, tail_bound=bound)]
model = NF.NormalizingFlow(base, flow_layers).to(device)
experiment_logger.info(f'Model prepared with {NUM_PARTICLES} particles and {NUM_DIM} dimensions!')

# Create a dedicated folder and logger for the first training round of the normalizing flow.
nf_training_dir = os.path.join(training_rounds_directory, "initial_training_round")
os.makedirs(nf_training_dir, exist_ok=True)
nf_log_file = os.path.join(nf_training_dir, "nf_training_round.log")
nf_logger = setup_logger("NF_training_round", nf_log_file, file_level=logging.DEBUG, stream_level=logging.WARNING)

# train the normalizing flow on initial training batch
loss_hist = np.array([])
loss_epoch = []

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
experiment_logger.info(f'Optimizer prepared with learning rate: {LR}')
data_save = []

for it in trange(EPOCHS, desc="Training Progress"):
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        # Compute loss
        # loss,z = model.reverse_kld(num_samples)
        loss = model.forward_kld(batch[0].to(device))
        # Do backprop and optimizer step
        # if ~(torch.isnan(loss) | torch.isinf(loss)):
        if torch.isnan(loss) or torch.isinf(loss):
            experiment_logger.warning("Warning: Loss is NaN or Inf. Skipping this batch.")
        else:
            loss.backward()
            optimizer.step()
            # scheduler.step()
        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        epoch_loss += loss.item()
    loss_epoch.append(epoch_loss / len(dataloader))
    nf_logger.info(f'Epoch {it+1}/{EPOCHS}, Loss: {loss_epoch[-1]:.4f}')

# Plot loss
loss_plot_path_svg, loss_plot_path_png = plot_loss(loss_epoch, nf_training_dir)
nf_logger.info(f"Loss plot saved successfully to: {loss_plot_path_png}")

model_path = os.path.join(nf_training_dir, "initial_model_circularspline_res_dense.pth")
torch.save(model.state_dict(), model_path)
nf_logger.info(f"Model saved successfully to: {model_path}")

# change to eval mode to test the NF model
model.eval()
samples_for_mc = []
z = model.sample(NUM_MC_RUNS)
z_ = z.reshape(-1, NUM_PARTICLES, NUM_DIM)  # Reshape as required
z_ = z_.to('cpu').detach().numpy()  # Move to CPU and convert to NumPy
z_ = z_ + HALF_BOX  # Add HALF_BOX to each element in the samples
print(z_)

# Generate samples
a_ = generate_samples(model, NUM_PARTICLES, NUM_DIM, 
                     n_iterations=NUM_SAMPLES_FOR_ANALYSIS // 5000 + 1, 
                     samples_per_iteration=5000)
a_ = a_ + HALF_BOX  # Add HALF_BOX to each element in the samples
print(a_.shape)
samples_path = os.path.join(nf_training_dir, "samples.npy")
np.save(samples_path, a_)
final_model_samples = a_

# make heatmap
heatmap_path_svg, heatmap_path_png = plot_frequency_heatmap(a_- HALF_BOX, nf_training_dir, cmap_div, HALF_BOX)
nf_logger.info(f"Frequency heatmap saved successfully to: {heatmap_path_png}")

# Calculate and plot pair correlation function
r_vals, g_r = calculate_pair_correlation(a_ - HALF_BOX, NUM_PARTICLES, HALF_BOX, dr=HALF_BOX/50)
pair_corr_path_svg, pair_corr_path_png = plot_pair_correlation(
    r_vals, 
    g_r, 
    nf_training_dir + "/"
)
print(f"Pair correlation function plot saved successfully to: {pair_corr_path_svg} and {pair_corr_path_png}")

# Add another data point showing 0 acceptance at the end of training
p_acc_history.append(0.0)
mcmc_steps_history.append(total_mcmc_steps)

# -------------------------
# After the full experiment, running the mc_runs with the samples from the final trained model
# -------------------------
trained_nf_model = model  # this is your trained normalizing flow model

# For each Monte Carlo simulation instance, assign the trained model:
for mc_run in mc_runs:
    mc_run.set_nf_model(trained_nf_model)


if TESTING:
    test_configs = final_model_samples  # configurations produced by the final model for big move suggestions
    
    free_energy_array = []
    
    # Initialize arrays to store acceptance rates for each run
    run_acceptance_rates = []
    run_acceptance_histories = []  # To store acceptance rate history for each run
    run_mcmc_steps_histories = []  # To store MCMC steps history for each run

    # Testing phase: for each MC run, run displacement steps and suggest big moves
    for run_idx, mc_run in enumerate(mc_runs):
        run_big_move_accepts = 0
        run_acceptance_history = []
        run_mcmc_steps_history = []
        local_mcmc_steps = 0
        
        for attempt in range(BIG_MOVE_ATTEMPTS):
            # Run BIG_MOVE_INTERVAL displacement steps and collect testing samples
            for step in range(1, BIG_MOVE_INTERVAL + 1):
                mc_run.particle_displacement()
                local_mcmc_steps += 1
                if step % SAMPLING_FREQUENCY == 0:
                    sample = mc_run.sample(step)
                    mc_run.local_samples.append(sample)
                    mc_run.testing_samples.append(sample[6])
            
            # Each MC run receives a unique configuration for a big move suggestion
            test_config = test_configs[attempt * len(mc_runs) + run_idx]
            accepted = mc_run.nf_big_move(test_config)
            
            if accepted:
                run_big_move_accepts += 1
            
            # Calculate and store current acceptance rate for this run
            current_run_acc_rate = run_big_move_accepts / (attempt + 1)
            run_acceptance_history.append(current_run_acc_rate)
            run_mcmc_steps_history.append(local_mcmc_steps)
            
            experiment_logger.info(f"Run {run_idx+1}, Attempt {attempt+1}: "
                                 f"p_accept = {current_run_acc_rate:.4f} "
                                 f"({run_big_move_accepts}/{attempt + 1})")

        # Store final acceptance rate for this run
        final_run_acc_rate = run_big_move_accepts / BIG_MOVE_ATTEMPTS
        run_acceptance_rates.append(final_run_acc_rate)
        run_acceptance_histories.append(run_acceptance_history)
        run_mcmc_steps_histories.append(run_mcmc_steps_history)
        
        print(f"Testing Summary - Simulation run {run_idx+1}: "
              f"Final p_accept = {final_run_acc_rate:.4f} "
              f"({run_big_move_accepts}/{BIG_MOVE_ATTEMPTS} moves accepted)")

    # Calculate average acceptance rate and error across all runs
    mean_acceptance = np.mean(run_acceptance_rates)
    std_acceptance = np.std(run_acceptance_rates)
    sem_acceptance = std_acceptance / np.sqrt(len(run_acceptance_rates))
    
    experiment_logger.info(f"\nOverall acceptance rate statistics:")
    experiment_logger.info(f"Mean acceptance rate: {mean_acceptance:.4f}")
    experiment_logger.info(f"Standard deviation: {std_acceptance:.4f}")
    experiment_logger.info(f"Standard error of mean: {sem_acceptance:.4f}")

    # Plot individual run acceptance histories
    plt.figure(figsize=(10, 6))
    for run_idx, history in enumerate(run_acceptance_histories):
        plt.plot(run_mcmc_steps_histories[run_idx], 
                history, 
                alpha=0.3, 
                label=f'Run {run_idx+1}')
    
    # Plot mean acceptance rate with error bands
    mean_history = np.mean(run_acceptance_histories, axis=0)
    std_history = np.std(run_acceptance_histories, axis=0)
    mcmc_steps = run_mcmc_steps_histories[0]  # All runs have same MCMC steps
    
    plt.plot(mcmc_steps, mean_history, 'k-', label='Mean', linewidth=2)
    plt.fill_between(mcmc_steps, 
                     mean_history - std_history, 
                     mean_history + std_history, 
                     color='gray', 
                     alpha=0.2, 
                     label='±1 std')
    
    plt.xlabel('MCMC Steps')
    plt.ylabel('Acceptance Rate')
    plt.title('Big Move Acceptance Rate by Run')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Save the plot
    acc_plot_path_svg = os.path.join(directory, "acceptance_rates_by_run.svg")
    acc_plot_path_png = os.path.join(directory, "acceptance_rates_by_run.png")
    plt.savefig(acc_plot_path_svg, bbox_inches='tight')
    plt.savefig(acc_plot_path_png, bbox_inches='tight')
    plt.close()
    
    experiment_logger.info(f"Acceptance rate plots saved to: {acc_plot_path_png}")
    
    # Save the acceptance rate data to a CSV file
    acceptance_data_path = os.path.join(directory, "acceptance_rate_data.csv")
    with open(acceptance_data_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['MCMC_Steps'] + [f'Run_{i+1}' for i in range(len(mc_runs))] + ['Mean', 'Std'])
        for step_idx in range(len(mcmc_steps)):
            row = [mcmc_steps[step_idx]]
            row.extend([history[step_idx] for history in run_acceptance_histories])
            row.extend([mean_history[step_idx], std_history[step_idx]])
            writer.writerow(row)
    experiment_logger.info(f"Acceptance rate data saved to: {acceptance_data_path}")

# if TESTING:
#     test_configs = final_model_samples  # configurations produced by the final model for big move suggestions
    
#     free_energy_array = []

#     # Testing phase: for each MC run, run displacement steps and suggest a big move with a unique config for each attempt
#     for run_idx, mc_run in enumerate(mc_runs):
#         for attempt in range(BIG_MOVE_ATTEMPTS):
#             # Run BIG_MOVE_INTERVAL displacement steps and collect testing samples
#             for step in range(1, BIG_MOVE_INTERVAL + 1):
#                 mc_run.particle_displacement()
#                 total_mcmc_steps += 1
#                 if step % ADJUSTING_FREQUENCY == 0:
#                     mc_run.adjust_displacement()
#                 if step % SAMPLING_FREQUENCY == 0:
#                     sample = mc_run.sample(step)
#                     mc_run.local_samples.append(sample)
#                     mc_run.testing_samples.append(sample[6])  # store configuration (in MC box coordinates)
            
#             # Each MC run receives a unique configuration for a big move suggestion
#             test_config = test_configs[attempt * len(mc_runs) + run_idx]
#             # Assume nf_big_move returns a boolean indicating whether the big move was accepted
#             accepted = mc_run.nf_big_move(test_config)
#             big_move_attempts += 1
#             if accepted:
#                 big_move_accepts += 1
            
#             # Initialize testing move counters if they don't exist yet
#             if not hasattr(mc_run, 'test_moves_attempted'):
#                 mc_run.test_moves_attempted = 0
#                 mc_run.test_moves_accepted = 0
            
#             # Update testing counters for this simulation run
#             mc_run.test_moves_attempted += 1
#             if accepted:
#                 mc_run.test_moves_accepted += 1
            
#             # Calculate and print the acceptance probability for this simulation run so far
#             p_accept = mc_run.test_moves_accepted / mc_run.test_moves_attempted
            
#             # Track global acceptance rate after each big move attempt
#             current_global_p_acc = big_move_accepts / big_move_attempts
#             p_acc_history.append(current_global_p_acc)
#             mcmc_steps_history.append(total_mcmc_steps)
                
#             experiment_logger.info(f"Step {total_mcmc_steps}, Run {run_idx+1}, Attempt {attempt+1}: "
#                                     f"p_accept = {current_global_p_acc:.4f} ({big_move_accepts}/{big_move_attempts})")

#         # Print summary after all attempts for this run are complete
#         print(f"Testing Summary - Simulation run {run_idx+1}: Final p_accept = {p_accept:.4f} ({mc_run.test_moves_accepted}/{mc_run.test_moves_attempted} moves accepted)")

#     # Plot the acceptance rate history
#     acc_plot_path_svg, acc_plot_path_png = plot_acceptance_rate(
#         p_acc_history, 
#         directory, 
#         x_values=mcmc_steps_history, 
#         xlabel='MCMC Steps',
#         base_filename="nf_acceptance_rate",
#         color='C2'
#     )
#     experiment_logger.info(f"Acceptance rate plot saved to: {acc_plot_path_png}")
    
#     # Save the acceptance rate data to a CSV file
#     acceptance_data_path = os.path.join(directory, "acceptance_rate_data.csv")
#     with open(acceptance_data_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['MCMC_Steps', 'Acceptance_Rate'])
#         for steps, acc_rate in zip(mcmc_steps_history, p_acc_history):
#             writer.writerow([steps, acc_rate])
#     experiment_logger.info(f"Acceptance rate data saved to: {acceptance_data_path}")

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
        avg_x_path_svg, avg_x_path_png = plot_avg_x_coordinate(
            testing_configs,
            run_folder,
            run_idx+1
        )

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
    color='C2'
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