import os
import numpy as np
import subprocess
import json
import csv
from datetime import datetime

def get_project_root():
    """
    Get the path to the flow_state project root directory.
    
    Returns:
        str: Path to the flow_state directory, or None if not found
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(current_dir) == "flow_state":
            return current_dir
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # Reached root directory
            return None
        current_dir = parent

def run_simulation(params, job_name, results_csv, log_dir, experiment_dir):
    """
    Run the simulation and append results.
    """
    specific_output_path = os.path.join(
        experiment_dir,  # Ensure it's within experiment_dir
        f"rho_{params['initial_rho']:.2f}_T_{params['temperature']:.3f}_AR_{params['aspect_ratio']:.3f}"
    )
    os.makedirs(specific_output_path, exist_ok=True)
    print(f"Created/Verified output directory: {specific_output_path}")

    # Define log file paths
    stdout_log = os.path.join(log_dir, f"{job_name}.out")
    stderr_log = os.path.join(log_dir, f"{job_name}.err")

    base_dir = get_project_root()
    if base_dir is None:
        print("Error: Could not find project root directory.")
        return

    # Path to main.py in project root
    main_script = os.path.join(base_dir, "MCMC", "main.py")

    # Construct the main simulation command
    simulation_cmd = [
        'python', main_script,
        '--temperature', str(params['temperature']),
        '--num_particles', str(params['num_particles']),
        '--initial_rho', str(params['initial_rho']),
        '--aspect_ratio', str(params['aspect_ratio']),
        '--equilibration_steps', str(params['equilibration_steps']),
        '--production_steps', str(params['production_steps']),
        '--sampling_frequency', str(params['sampling_frequency']),
        '--adjusting_frequency', str(params['checkpoint_frequency']),
        '--output_path', specific_output_path,
        '--experiment_id', params['experiment_id'],
        # **New Arguments for External Potential**
        '--num_wells', str(params['num_wells']),
        '--V0_list', *map(str, params['V0_list']),
        '--k', str(params['k']),
        '--r0', str(params['r0']),
        '--initialisation_type', params['initialisation_type'],
        '--seed', str(params['seed']),
        '--initial_max_displacement', str(params['initial_max_displacement'])
    ]

    # Include flags based on parameters
    if params.get('visualise', False):
        simulation_cmd.append('--visualise')
    if params.get('checking', False):
        simulation_cmd.append('--checking')
    if params.get('time_calc', False):
        simulation_cmd.append('--time_calc')

    # Path to append_results.py in scripts directory
    append_script = os.path.join(base_dir, "MCMC", "scripts", "append_results.py")

    # Construct the append results command
    append_cmd = [
        'python', append_script,
        results_csv,
        specific_output_path,
        str(params['temperature']),
        str(params['equilibration_steps'])
    ]

    # Open log files
    with open(stdout_log, 'w') as out, open(stderr_log, 'w') as err:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting job {job_name}")
        # Run the simulation
        sim_process = subprocess.run(simulation_cmd, stdout=out, stderr=err, cwd=base_dir)
        if sim_process.returncode == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Simulation {job_name} completed successfully.")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Simulation {job_name} failed. Check logs for details.")

        # Run the append results
        append_process = subprocess.run(append_cmd, stdout=out, stderr=err, cwd=base_dir)
        if append_process.returncode == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Results appended for job {job_name}.")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Appending results for job {job_name} failed. Check logs for details.")

def run_experiments():
    """
    Set up and run the experiments based on predefined parameters.
    """
    # Get project root directory
    base_dir = get_project_root()
    if base_dir is None:
        print("Error: Could not find project root directory.")
        return

    # Define experiment parameters
    params = {
        "num_particles": 3,
        "density_start": 0.03,        # Starting density
        "density_end": 0.03,          # Ending density
        "density_intervals": 1,      # Number of density intervals
        "equilibration_steps": 5000,
        "production_steps": 15360000,
        "sampling_frequency": 150,
        "checkpoint_frequency": 5000,  # Typically (equil + prod)/5 or user-chosen
        "output_path": os.path.join(base_dir, "MCMC", "data"),
        "experiment_id": "f_102400_samples_left_rho_0.03_gpu",
        "temp_start": 1.00,            # Starting temperature
        "temp_end": 1.00,              # Ending temperature
        "temp_intervals": 1,          # Number of temperature intervals
        "aspect_ratio_start": 1.0,
        "aspect_ratio_end": 1.0,
        "aspect_ratio_intervals": 1,
        "visualise": True,            # Set to True if you want visualization for individual runs
        "checking": True,             # Enable logging of density and box volume
        # **New Parameters for External Potential**
        "num_wells": 2,                # Number of external potential wells (0, 1, or 2)
        "V0_list": [-10.0,-10.0],                    # Depth of the potential wells
        "k": 15,                       # Width parameter for the Gaussian wells
        "r0": 1.2,                     # radius of bottom of well
        "initialisation_type": "left_half",  # Type of initialisation for particles: 'all', 'left_half', 'right_half'
        "seed": 42,
        "initial_max_displacement": 0.65  # Initial maximum displacement for Monte Carlo moves
    }

    if not os.path.isdir(base_dir):
        print(f"Error: The base directory '{base_dir}' does not exist.")
        return

    # Create the experiment directory
    experiment_dir = os.path.join(params['output_path'], params['experiment_id'])
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Created/Verified experiment directory: {experiment_dir}")

    # Save parameters to a JSON file
    params_file = os.path.join(experiment_dir, 'parameters.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Saved experiment parameters to {params_file}")

    # Create a log directory
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"Created/Verified log directory: {log_dir}")

    # Prepare CSV file for results
    results_csv = os.path.join(experiment_dir, "results.csv")
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Temperature", "Aspect_Ratio", "Density", "Average_Pressure"])
        print(f"Initialized results CSV at {results_csv}")

    # Generate density values using number of intervals
    if params['density_intervals'] < 1:
        print("Error: 'density_intervals' must be at least 1.")
        return
    densities = np.linspace(params['density_start'], params['density_end'], num=params['density_intervals'])
    densities = np.round(densities, decimals=4)  # Avoid floating point issues
    print(f"Generated densities: {densities}")

    # Generate temperature values using number of intervals
    if params['temp_intervals'] < 1:
        print("Error: 'temp_intervals' must be at least 1.")
        return
    temperatures = np.linspace(params['temp_start'], params['temp_end'], num=params['temp_intervals'])
    temperatures = np.round(temperatures, decimals=4)  # Avoid floating point issues
    print(f"Generated temperatures: {temperatures}")

    # Generate aspect ratio values
    if params['aspect_ratio_intervals'] < 1:
        print("Error: 'aspect_ratio_intervals' must be at least 1.")
        return
    aspect_ratios = np.linspace(params['aspect_ratio_start'], params['aspect_ratio_end'], 
                                params['aspect_ratio_intervals'])
    aspect_ratios = np.round(aspect_ratios, decimals=4)  # Avoid floating point issues
    print(f"Generated aspect ratios: {aspect_ratios}")

    # Iterate over density, temperature, and aspect ratio ranges
    for density in densities:
        for temperature in temperatures:
            for aspect_ratio in aspect_ratios:
                # Create job name with density first, followed by temperature and aspect ratio
                job_name = f"{params['experiment_id']}_rho{density:.2f}_T{temperature:.3f}_AR{aspect_ratio:.3f}"
                specific_output_path = os.path.join(
                    experiment_dir,
                    f"rho_{density:.2f}_T_{temperature:.3f}_AR_{aspect_ratio:.3f}"
                )
                os.makedirs(specific_output_path, exist_ok=True)
                print(f"Prepared output directory for job {job_name}: {specific_output_path}")

                # Update parameters for the current job
                job_params = params.copy()
                job_params.update({
                    'temperature': temperature,
                    'aspect_ratio': aspect_ratio,
                    'initial_rho': density  # Update density for the current job
                })

                print(f"Queueing job {job_name}...")

                # Run the simulation and handle logging
                run_simulation(job_params, job_name, results_csv, log_dir, experiment_dir)

                print(f"Job {job_name} has been processed.\n")

    print("All simulations have been completed.")

if __name__ == "__main__":
    run_experiments()
