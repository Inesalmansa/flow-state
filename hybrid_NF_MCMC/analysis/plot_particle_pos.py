import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Set the experiment directory
experiment_dir = "/home/n2401517d/my_workspace/HMC_NF/results/N3_T_1.0_V_0.50KT_cumulative_102400_total"
mc_runs_dir = os.path.join(experiment_dir, "mc_runs")

# Create a figure for the plot
plt.figure(figsize=(12, 8))

# Loop through each run folder
run_folders = sorted(glob.glob(os.path.join(mc_runs_dir, "run_*")))
num_runs = len(run_folders)

# List to store average x trajectories for all runs
all_avg_x = []
max_length = 0

# First pass: Load data and determine the maximum trajectory length
for i, run_folder in enumerate(run_folders):
    config_file = os.path.join(run_folder, "mc_run_configs.npy")
    if os.path.exists(config_file):
        # Load configurations
        configs = np.load(config_file)
        # Calculate average x position for each configuration
        avg_x = [np.mean(config[:, 0]) for config in configs]
        all_avg_x.append(avg_x)
        max_length = max(max_length, len(avg_x))
    else:
        print(f"Warning: Could not find {config_file}")

# Create a common x-axis for all trajectories
steps = np.arange(1, max_length + 1)

# Plot all trajectories with different colors and calculate statistics
avg_trajectories = np.zeros((num_runs, max_length))
avg_trajectories[:] = np.nan  # Initialize with NaN

for i, avg_x in enumerate(all_avg_x):
    # Pad shorter trajectories with NaN
    padded_avg_x = np.full(max_length, np.nan)
    padded_avg_x[:len(avg_x)] = avg_x
    avg_trajectories[i, :] = padded_avg_x
    
    # Plot individual trajectory
    plt.plot(steps, padded_avg_x, alpha=0.5, linewidth=0.8, 
             label=f"Run {i+1:03d}")

# Create a 5x2 grid of subplots
fig, axes = plt.subplots(5, 2, figsize=(15, 20))

# Plot each run in its own subplot
for i in range(10):
    ax = axes[i // 2, i % 2]
    
    # Get trajectory for this run
    if i < len(avg_trajectories):
        traj = avg_trajectories[i]
        ax.plot(steps, traj, 'b-', linewidth=1.0)
    else:
        ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', 
                verticalalignment='center', transform=ax.transAxes)
    
    # Set labels and title for each subplot
    ax.set_xlabel("MC Step", fontsize=10)
    ax.set_ylabel(r"$\langle x \rangle$", fontsize=10)
    ax.set_title(f"Run {i+1:03d}", fontsize=12)

# Adjust layout and save
plt.tight_layout()
output_path = os.path.join(experiment_dir, "all_mc_runs_avg_x.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Calculate overall statistics for the final positions
final_positions = avg_trajectories[:, -1]
valid_finals = final_positions[~np.isnan(final_positions)]
final_mean = np.mean(valid_finals)
final_std = np.std(valid_finals)
final_sem = final_std / np.sqrt(len(valid_finals))

# Print statistics
print(f"Final average position: {final_mean:.4f} ± {final_sem:.4f} (SEM)")
print(f"Standard deviation: {final_std:.4f}")
print(f"Data from {len(valid_finals)} valid runs")

plt.close()
