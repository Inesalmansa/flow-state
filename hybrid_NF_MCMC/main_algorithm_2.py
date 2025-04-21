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

# define experiment variables 
NUM_MC_RUNS = 100        # how many mc runs going on in 'parallel'
MASTER_SEED = 42        # master seed to give seeds for each mc run to run differently
NUM_PARTICLES = 3
NUM_DIM = 2
NUM_TRAINING_CYCLES = 1000
NF_TARGET_ACC = 0.5

# nf needed params
INITIAL_TRAINING_NUM_SAMPLES = 1000
BATCH_SIZE = 256
K = 23
EPOCHS = 1
LR = 0.000543510751759681
WEIGHT_DECAY = 9.5857178422352e-05
UPDATE_NUM_SAMPLES = 1000
CHECKPOINT_INTERVAL = 10
HIDDEN_UNITS = 128
NUM_BINS = 15
N_BLOCKS = 2
ALPHA = 1.0

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
SAMPLING_FREQUENCY = 10
ADJUSTING_FREQUENCY = 10000
CUMULATIVE_TRAINING_SAMPLES = False # Set to False to use only the new samples for each training round
PRODUCTION_SAMPLES = int((UPDATE_NUM_SAMPLES / (NUM_MC_RUNS / SAMPLING_FREQUENCY)) / SAMPLING_FREQUENCY) * NUM_TRAINING_CYCLES
print("number of production samples to be produced by each run:", PRODUCTION_SAMPLES)

# analysis parameters
NUM_SAMPLES_FOR_ANALYSIS = 50000
NUMBER_OF_SAMPLES_FOR_FREE_ENERGY = 5000
START_IDX = PRODUCTION_SAMPLES - NUMBER_OF_SAMPLES_FOR_FREE_ENERGY

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
training_directory = os.path.join(directory, "training")
os.makedirs(training_directory, exist_ok=True)

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
    "UPDATE_NUM_SAMPLES": UPDATE_NUM_SAMPLES,
    "HIDDEN_UNITS": HIDDEN_UNITS,
    "NUM_BINS": NUM_BINS,
    "N_BLOCKS": N_BLOCKS,
    "ALPHA": ALPHA,
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
    "CUMULATIVE_TRAINING_SAMPLES": CUMULATIVE_TRAINING_SAMPLES,
    "NUM_SAMPLES_FOR_ANALYSIS": NUM_SAMPLES_FOR_ANALYSIS
}
json_file_path = os.path.join(directory, "params.json")
with open(json_file_path, "w") as f:
    json.dump(experiment_params, f, indent=4)
experiment_logger.info(f"Experiment parameters saved to {json_file_path}")

# initialisation of experiment - sets up the monte carlo runs and equilibrates and collects samples
mc_runs = []  # list to store each Monte Carlo simulation instance

# Initialize MCMC step counter and acceptance tracking at the very beginning
total_mcmc_steps = 0
p_acc_history = [0.0]  # Initial acceptance rate is 0
mcmc_steps_history = [0]  # Start from MCMC step 0
big_move_attempts = 0
big_move_accepts = 0

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
            
# collect initial set of training samples
global_samples_mc = []  # storage of mc samples in mc box range of 0,10
production_runs = int(INITIAL_TRAINING_NUM_SAMPLES/(NUM_MC_RUNS / SAMPLING_FREQUENCY))
experiment_logger.info("production runs per cycle: " + str(production_runs))
for mc_run in mc_runs:
    for step in range(1, production_runs + 1):
        mc_run.particle_displacement()
        total_mcmc_steps += 1  
        if step % SAMPLING_FREQUENCY == 0:
            sample = mc_run.sample(step)
            mc_run.local_samples.append(sample)
            global_samples_mc.append(sample[6])  # only config being stored (in MC box bound)

global_samples_nf = np.array([np.array([particle - np.array([HALF_BOX, HALF_BOX]) for particle in config]) for config in global_samples_mc])

# Initialize cumulative training samples if enabled.
# Note that global_samples_nf were collected during the production phase of initial training.
if CUMULATIVE_TRAINING_SAMPLES:
    cumulative_training_samples_nf = global_samples_nf.copy()
else:
    # Even when not using cumulative samples for training, we still need to track all samples 
    # for the sliding window approach
    all_collected_samples_nf = global_samples_nf.copy()

# prepping initial samples as for dataloader
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
experiment_logger.info('device is: ' + str(device))

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
target = NF.Energy.DoubleWellLJ(NUM_PARTICLES * NUM_DIM, NUM_PARTICLES, TEMP, bound,
                               V0_list=V0_LIST, 
                               r0=R0,               
                               k=K_VAL)  
flow_layers = []
for i in range(K):
    flow_layers += [NF.flows.CircularCoupledRationalQuadraticSpline(
        NUM_PARTICLES * NUM_DIM, 
        N_BLOCKS, 
        HIDDEN_UNITS,
        range(NUM_PARTICLES * NUM_DIM), 
        num_bins=NUM_BINS, 
        tail_bound=bound)] 
model = NF.NormalizingFlow(base, flow_layers, target).to(device)
experiment_logger.info(f'Model prepared with {NUM_PARTICLES} particles and {NUM_DIM} dimensions!')

# Create a directory for training results instead of per-cycle directories
training_directory = os.path.join(directory, "training")
os.makedirs(training_directory, exist_ok=True)
training_log_file = os.path.join(training_directory, "nf_training.log")
training_logger = setup_logger("NF_training", training_log_file, file_level=logging.DEBUG, stream_level=logging.WARNING)

# Initialize arrays to store loss history across all cycles
cumulative_loss_history = []

# Initial training
loss_epoch = []

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
experiment_logger.info(f'Optimizer prepared with learning rate: {LR} and weight decay: {WEIGHT_DECAY}')
data_save = []

for it in trange(EPOCHS, desc="Training Progress"):
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        # Compute loss
        energy_loss,z = model.reverse_kld(BATCH_SIZE)
        sample_loss = model.forward_kld(batch[0].to(device))
        loss = ALPHA * sample_loss + (1 - ALPHA) * energy_loss
        # Do backprop and optimizer step
        # if ~(torch.isnan(loss) | torch.isinf(loss)):
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            # scheduler.step()
        # Log loss
        epoch_loss += loss.item()
    loss_epoch.append(epoch_loss / len(dataloader))
    training_logger.info(f'Epoch {it+1}/{EPOCHS}, Loss: {loss_epoch[-1]:.4f}')

# Update the loss_epoch array to store initial training losses
cumulative_loss_history.extend(loss_epoch)

# Plot loss using utility function
loss_plot_path_svg, loss_plot_path_png = plot_loss(loss_epoch, training_directory + "/")
training_logger.info(f"Loss plot saved successfully to: {loss_plot_path_svg} and {loss_plot_path_png}")

model_path = os.path.join(training_directory, "initial_model_circularspline_res_dense.pth")
torch.save(model.state_dict(), model_path)
training_logger.info(f"Model saved successfully to: {model_path}")

# change to eval mode to test the NF model
model.eval()
samples_for_mc = []
z = model.sample(NUM_MC_RUNS)
z_ = z.reshape(-1, NUM_PARTICLES, NUM_DIM)  # Reshape as required
z_ = z_.to('cpu').detach().numpy()  # Move to CPU and convert to NumPy
z_ = z_ + HALF_BOX  # Add HALF_BOX to each element in the samples
print(z_)

# Generate samples using the utility function
a_ = generate_samples(model, NUM_PARTICLES, NUM_DIM, 
                     n_iterations=NUM_SAMPLES_FOR_ANALYSIS // 5000 + 1, 
                     samples_per_iteration=5000)
a_ = a_ + HALF_BOX  # Add HALF_BOX to each element in the samples
print(a_.shape)
samples_path = os.path.join(training_directory, "samples.npy")
np.save(samples_path, a_)

# Create heatmap using utility function
heatmap_path_svg, heatmap_path_png = plot_frequency_heatmap(
    a_ - HALF_BOX,  # Convert back to centered coordinates for plotting
    training_directory + "/", 
    cmap_div, 
    HALF_BOX
)
print(f"Frequency heatmap saved successfully to: {heatmap_path_svg} and {heatmap_path_png}")

# Calculate and plot pair correlation function
r_vals, g_r = calculate_pair_correlation(a_ - HALF_BOX, NUM_PARTICLES, HALF_BOX, dr=HALF_BOX/50)
pair_corr_path_svg, pair_corr_path_png = plot_pair_correlation(
    r_vals, 
    g_r, 
    training_directory + "/"
)
print(f"Pair correlation function plot saved successfully to: {pair_corr_path_svg} and {pair_corr_path_png}")

# Initialise acceptance counters
nf_mc_accept = 0
nf_to_mc_attempts = 0

# Compute and log the acceptance probability
p_acc = nf_mc_accept / nf_to_mc_attempts if nf_to_mc_attempts > 0 else 0
p_acc_history.append(p_acc)
mcmc_steps_history.append(total_mcmc_steps)
training_logger.info(f"Monte Carlo acceptance: {nf_mc_accept} out of {nf_to_mc_attempts} attempts (p_accept = {p_acc:.4f})")

# -------------------------
# Begin update training cycles
# -------------------------
for cycle in range(NUM_TRAINING_CYCLES):
    experiment_logger.info(f"Starting update training cycle {cycle+1}/{NUM_TRAINING_CYCLES}")

    # -------------------------------------------------------------------
    # Production Phase: Generate new MC samples for NF update training
    # -------------------------------------------------------------------
    global_samples_mc_update = []  # reset samples
    # calculate number of production runs based on UPDATE_NUM_SAMPLES
    production_runs = int(UPDATE_NUM_SAMPLES / (NUM_MC_RUNS / SAMPLING_FREQUENCY))
    experiment_logger.info(f"Cycle {cycle+1}: production runs per MC run: {production_runs}")
    for mc_run in mc_runs:
        for step in range(1, production_runs + 1):
            mc_run.particle_displacement()
            total_mcmc_steps += 1  # Count each MCMC step
            if step % SAMPLING_FREQUENCY == 0:
                sample = mc_run.sample(step)
                mc_run.local_samples.append(sample)
                mc_run.testing_samples.append(sample[6]) 
                # store the configuration (in MC box coordinates)
                global_samples_mc_update.append(sample[6])
    
    # Convert collected samples into NF coordinates (centered at zero)
    global_samples_nf_update = np.array([
        np.array([particle - np.array([HALF_BOX, HALF_BOX]) for particle in config])
        for config in global_samples_mc_update
    ])

    # Use cumulative training samples if enabled; otherwise, use just the current cycle's samples
    if CUMULATIVE_TRAINING_SAMPLES:
        cumulative_training_samples_nf = np.concatenate((cumulative_training_samples_nf, global_samples_nf_update), axis=0)
        training_data = cumulative_training_samples_nf
    else:
        # Simply use the current cycle's samples directly - no need to track old samples
        training_data = global_samples_nf_update

    # Create a dataloader for the update training set
    unique_data = np.unique(training_data, axis=0)
    experiment_logger.info(f"Cycle {cycle+1}: Total unique update samples: {unique_data.shape[0]}")
    dataloader_update = get_dataloader(training_data, NUM_PARTICLES, NUM_DIM, device, batch_size=BATCH_SIZE, shuffle=True)
    experiment_logger.info(f"Cycle {cycle+1}: DataLoader size: {len(dataloader_update.dataset)}, batches: {len(dataloader_update)}")
    
    # -------------------------------------------------------------------
    # Training phase: Switch model to training mode and train
    # -------------------------------------------------------------------
    model.train()
    
    # Initialize the optimizer for this cycle
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Single epoch training
    cycle_loss = 0.0
    for batch in dataloader_update:
        optimizer.zero_grad()
        energy_loss,z = model.reverse_kld(BATCH_SIZE)
        sample_loss = model.forward_kld(batch[0].to(device))
        loss = ALPHA * sample_loss + (1 - ALPHA) * energy_loss
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        cycle_loss += loss.item()
    
    avg_cycle_loss = cycle_loss / len(dataloader_update)
    cumulative_loss_history.append(avg_cycle_loss)
    training_logger.info(f'Cycle {cycle+1}, Loss: {avg_cycle_loss:.4f}')
    
    # Plot loss every checkpoint interval cycles or on the final cycle
    if (cycle + 1) % CHECKPOINT_INTERVAL == 0 or cycle == NUM_TRAINING_CYCLES - 1:
        loss_plot_path_svg, loss_plot_path_png = plot_loss(
            cumulative_loss_history,
            training_directory + "/",
            base_filename=f"loss_function"
        )
        training_logger.info(f"Loss plot updated at cycle {cycle+1} and saved to: {loss_plot_path_svg} and {loss_plot_path_png}")
    
    # Save the model periodically
    if (cycle + 1) % CHECKPOINT_INTERVAL*2 == 0 or cycle == NUM_TRAINING_CYCLES - 1:
        model_path = os.path.join(training_directory, f"model_checkpoint_cycle_{cycle+1}.pth")
        torch.save(model.state_dict(), model_path)
        training_logger.info(f"Model checkpoint saved at cycle {cycle+1} to: {model_path}")
    
    # -------------------------------------------------------------------
    # Refeeding Phase: Evaluate model and track acceptance statistics
    # -------------------------------------------------------------------
    model.eval()
    
    # Generate samples for MC runs (keeping this for z_ generation consistency)
    z = model.sample(NUM_MC_RUNS)
    z_ = z.reshape(-1, NUM_PARTICLES, NUM_DIM)
    z_ = z_.to('cpu').detach().numpy()
    z_ = z_ + HALF_BOX
    
    # New method: Continue offering configurations until target percentage of MC runs have accepted
    if (cycle + 1) % CHECKPOINT_INTERVAL == 0 or cycle == NUM_TRAINING_CYCLES - 1:
        production_batch_size = 1000  # Fixed batch size for production
        n_iterations = (NUM_SAMPLES_FOR_ANALYSIS + production_batch_size - 1) // production_batch_size  # Ceiling division
        all_samples = []

        with torch.no_grad():
            for i in range(n_iterations):
                # Generate samples in smaller batches
                a = model.sample(production_batch_size)
                # Reshape as required
                a_ = a.reshape(-1, NUM_PARTICLES, NUM_DIM)
                # Move to CPU and convert to NumPy
                a_ = a_.to('cpu').detach().numpy()
                # Add HALF_BOX to each element in the samples
                a_ = a_ + HALF_BOX
                all_samples.append(a_)

        a_ = np.concatenate(all_samples, axis=0)
        
        samples_path = os.path.join(training_directory, "samples.npy")
        np.save(samples_path, a_)
        training_logger.info(f"Cycle {cycle+1}: Saved {NUM_SAMPLES_FOR_ANALYSIS} samples to {samples_path}")

        # Create heatmap
        heatmap_path_svg, heatmap_path_png = plot_frequency_heatmap(
            a_ - HALF_BOX,  # Convert back to centered coordinates for plotting
            training_directory + "/", 
            cmap_div, 
            HALF_BOX,
            base_filename=f"frequency_heatmap_cycle_{cycle+1}"
        )
        print(f"Frequency heatmap saved successfully to: {heatmap_path_svg} and {heatmap_path_png}")

        # Calculate and plot pair correlation function
        r_vals, g_r = calculate_pair_correlation(a_ - HALF_BOX, NUM_PARTICLES, HALF_BOX, dr=HALF_BOX/50)
        pair_corr_path_svg, pair_corr_path_png = plot_pair_correlation(
            r_vals, 
            g_r, 
            training_directory + "/", 
            base_filename=f"pair_correlation_function_{cycle+1}"
        )
        print(f"Pair correlation function plot saved successfully to: {pair_corr_path_svg} and {pair_corr_path_png}")

    # Track acceptance statistics
    nf_mc_accept_update = 0
    nf_to_mc_attempts_update = 0
    target_num_accepts = int(NF_TARGET_ACC * NUM_MC_RUNS)
    accepted_runs = set()

    for mc_run in mc_runs:
        mc_run.set_nf_model(model)
    
    for sample_idx in range(NUM_MC_RUNS):
        mc_run = mc_runs[sample_idx]
        
        # Test the configuration
        test_config = z_[sample_idx]
        nf_to_mc_attempts_update += 1
        accepted = mc_run.nf_big_move(test_config)
        
        if accepted:
            nf_mc_accept_update += 1
            accepted_runs.add(sample_idx)
            training_logger.info(f"Cycle {cycle+1}: MC run {sample_idx+1} accepted a configuration")
    
    # Calculate and log acceptance statistics
    p_acc_update = nf_mc_accept_update / nf_to_mc_attempts_update if nf_to_mc_attempts_update > 0 else 0
    p_acc_history.append(p_acc_update)
    mcmc_steps_history.append(total_mcmc_steps)
    
    # Update global counters
    big_move_attempts += nf_to_mc_attempts_update
    big_move_accepts += nf_mc_accept_update
    
    training_logger.info(f"Cycle {cycle+1}: {len(accepted_runs)}/{NUM_MC_RUNS} MC runs accepted a configuration")
    training_logger.info(f"Cycle {cycle+1}: Tested {nf_to_mc_attempts_update} configurations total")
    training_logger.info(f"Cycle {cycle+1}: Overall acceptance rate: {p_acc_update:.4f}")
    experiment_logger.info(f"Cycle {cycle+1}: Target reached after testing {nf_to_mc_attempts_update} configurations with p_accept = {p_acc_update:.4f}")
    print(f"Target reached: {len(accepted_runs)}/{NUM_MC_RUNS} MC runs accepted. p(accept) = {p_acc_update:.4f} after {nf_to_mc_attempts_update} attempts")
    
    # Plot acceptance rate every checkpoint cycles or on the final cycle
    if (cycle + 1) % CHECKPOINT_INTERVAL == 0 or cycle == NUM_TRAINING_CYCLES - 1:
        acc_plot_path_svg, acc_plot_path_png = plot_acceptance_rate(
            p_acc_history,
            training_directory + "/",
            x_values=mcmc_steps_history,
            xlabel='MCMC Steps',
            base_filename="nf_acceptance_rate_mcmc_steps",
            color='C4'
        )
        training_logger.info(f"Acceptance rate vs MCMC steps plot updated at cycle {cycle+1} and saved to: {acc_plot_path_svg} and {acc_plot_path_png}")

    experiment_logger.info(f"Finished update training cycle {cycle+1}/{NUM_TRAINING_CYCLES}")

final_model_samples = a_ 

# -------------------------
# After all update cycles: Plot p_acc_history as a function of total training samples
# -------------------------

# Compute the cumulative number of training samples used at each point.
# The initial model was trained on INITIAL_TRAINING_NUM_SAMPLES samples.
# Then, each update cycle adds UPDATE_NUM_SAMPLES samples.
total_samples_trained = [INITIAL_TRAINING_NUM_SAMPLES]
for i in range(1, NUM_TRAINING_CYCLES + 1):
    total_samples_trained.append(total_samples_trained[-1] + UPDATE_NUM_SAMPLES)

# Save the data used for plotting in a JSON file for future reference
plot_data = {
    "total_samples_trained": total_samples_trained,
    "p_acc_history": p_acc_history
}
data_save_path = os.path.join(directory, "p_acc_history_data.json")
with open(data_save_path, "w") as f:
    json.dump(plot_data, f, indent=4)
experiment_logger.info(f"Plot data saved successfully to: {data_save_path}")

# Plot the acceptance history with utility function
acc_vs_samples_path_svg, acc_vs_samples_path_png = plot_acceptance_rate(
    p_acc_history,
    directory + "/",
    x_values=total_samples_trained,
    xlabel='Training Samples',
    color='C4'
)
experiment_logger.info(f"Acceptance history plot saved successfully to: {acc_vs_samples_path_svg} and {acc_vs_samples_path_png}")

# -------------------------
# After the full experiment, running the mc_runs with the samples from the final trained model
# -------------------------
trained_nf_model = model  # this is your trained normalizing flow model
# Initialize free_energy_array outside the loop to collect data from all runs
free_energy_array = []

# After the testing phase, plot average X coordinate vs. step number using the stored testing configurations.
for run_idx, mc_run in enumerate(mc_runs):
    if mc_run.testing_samples:
        testing_configs = np.array(mc_run.testing_samples)  # shape: (num_samples, num_particles, 2)
        testing_configs = testing_configs[START_IDX:]
        
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
    color='C5'
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

# After all update cycles, plot the acceptance rate history
acc_plot_path_svg, acc_plot_path_png = plot_acceptance_rate(
    p_acc_history,
    directory + "/",
    x_values=mcmc_steps_history,
    xlabel='MCMC Steps',
    base_filename="nf_acceptance_rate_mcmc_steps",
    color='C5'
)
experiment_logger.info(f"Acceptance rate vs MCMC steps plot saved to: {acc_plot_path_svg} and {acc_plot_path_png}")

# Save the acceptance rate data to a CSV file
acceptance_data_path = os.path.join(directory, "acceptance_rate_mcmc_steps_data.csv")
with open(acceptance_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['MCMC_Steps', 'Acceptance_Rate'])
    for steps, acc_rate in zip(mcmc_steps_history, p_acc_history):
        writer.writerow([steps, acc_rate])
experiment_logger.info(f"Acceptance rate vs MCMC steps data saved to: {acceptance_data_path}")


