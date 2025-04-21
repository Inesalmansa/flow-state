# takes in the name of an experiment and recreates the p(acc) plot but instead of only testing acceptance x NUM_MC_RUNS in each training iteration we load in each model, generate more samples
# from tested samples we get more precise p_acc plot

import os
import numpy as np
import matplotlib.pyplot as plt
from my_workspace.HMC_NF.analysis.utils_analysis import load_config

def run_p_acc_hr_analysis(experiment_folder, model_files, model_interval=1):
    """
    For each selected model, generate one Monte Carlo run, compute the reference average energy,
    and use the Metropolis criterion to evaluate acceptance of test configurations.
    """
    config = load_config(experiment_folder)
    selected_models = model_files[::model_interval]
    p_acc_values = []

    for mod_idx, model_path in enumerate(selected_models):
        print(f"Processing model {mod_idx+1}: {model_path}")
        # Generate test run samples and corresponding energies.
        test_samples, energies = generate_test_run(model_path, config)
        ref_energy = np.mean(energies)
        print(f"Reference energy: {ref_energy:.3f}")
        
        accepted = 0
        total = len(test_samples)
        for energy in energies:
            # Use a simple criterion (accept if energy is less than or equal to ref_energy).
            # You can enhance this using the Boltzmann factor.
            if energy <= ref_energy:
                accepted += 1
        p_acc = accepted / total if total > 0 else 0
        print(f"Model {mod_idx+1}: Accepted {accepted}/{total} samples, p_acc = {p_acc:.3f}")
        p_acc_values.append(p_acc)
    
    plt.figure(figsize=(8, 6), dpi=250)
    plt.plot(range(len(p_acc_values)), p_acc_values, marker='o')
    plt.xlabel("Model Index", fontsize=18)
    plt.ylabel("Acceptance Probability", fontsize=18)
    plt.title("Acceptance Probability vs Model", fontsize=20)
    # Save plot in the "analysis" folder within the experiment folder.
    output_dir = os.path.join(experiment_folder, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "p_acc_hr_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Acceptance probability plot saved to: {plot_path}")

def generate_test_run(model_path, config):
    """
    Placeholder: Replace this with the actual generation of a Monte Carlo run using the model.
    For now, we return dummy energies and configurations.
    """
    num_samples = 20
    energies = np.random.normal(loc=-1.0, scale=0.1, size=num_samples)
    test_samples = [np.random.rand(config.get("num_particles", 64), 2) * config.get('bound', 10)
                    for _ in range(num_samples)]
    return test_samples, energies