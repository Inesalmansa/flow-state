# takes the name of an experiment, using that takes the most trained model and runs a long MC sim (using the MC sim params) and periodically asks the trained model for a configuration
# final plot is <x> as a function of steps

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from my_workspace.HMC_NF.analysis.utils_analysis import load_sampled_data

def run_mean_x_analysis(experiment_folder):
    """
    Reads the sampled_data CSV, computes the average x position for each configuration,
    and plots <x> as a function of simulation steps.
    """
    sampled_data = load_sampled_data(experiment_folder)
    
    # We assume the CSV contains at minimum "cycle_number" and "particle_configuration" columns.
    mean_x_values = []
    steps = []
    
    for idx, row in sampled_data.iterrows():
        cycle_number = row["cycle_number"]
        # Parse the particle configuration.
        # (This simplistic approach uses eval; in real code consider using json.loads)
        particles_flat = eval(row["particle_configuration"])
        # Reshape assuming a 2D simulation and a known number of particles.
        num_particles = int(row.get("num_particles", 64))
        particles_array = np.array(particles_flat).reshape((num_particles, 2))
        mean_x = np.mean(particles_array[:, 0])
        mean_x_values.append(mean_x)
        steps.append(cycle_number)
    
    plt.figure(figsize=(8, 6), dpi=250)
    plt.plot(steps, mean_x_values, marker='o')
    plt.xlabel("Cycle Number", fontsize=18)
    plt.ylabel("<x>", fontsize=18)
    plt.title("Average x Position over Simulation Steps", fontsize=20)
    # Save plot in the "analysis" folder within the experiment folder.
    output_dir = os.path.join(experiment_folder, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "mean_x_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Mean x plot saved to: {plot_path}")