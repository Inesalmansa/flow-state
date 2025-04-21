import numpy as np
import matplotlib.pyplot as plt
import os
import sys

HALF_BOX = 5

print("hello")

def analyze_free_energy(base_dir, num_runs=10):
    all_free_energy_differences = []

    print(f"Starting analysis of {num_runs} runs...")
    for run_number in range(1, num_runs + 1):
        print(f"Processing run {run_number}/{num_runs}")
        run_dir = os.path.join(base_dir, f"run_{run_number:03}")
        config_path = os.path.join(run_dir, "mc_run_testing_configs.npy")
        
        print(f"Loading configurations from {config_path}")
        testing_configs = np.load(config_path)
        
        proportions_3_left = []
        proportions_3_right = []
        
        print(f"Analyzing {len(testing_configs)} configurations...")
        for i in range(1, len(testing_configs) + 1):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(testing_configs)} configurations")
            current_configs = testing_configs[:i]
            
            count_3_left = sum(1 for config in current_configs if sum(1 for particle in config if particle[0] < HALF_BOX) == 3)
            count_3_right = sum(1 for config in current_configs if sum(1 for particle in config if particle[0] >= HALF_BOX) == 3)
            
            total = len(current_configs)
            proportions_3_left.append(count_3_left / total)
            proportions_3_right.append(count_3_right / total)
        
        k_B_T = 1
        print("Calculating free energy differences...")
        free_energy_differences = [
            -np.log(p_left / p_right) * k_B_T if p_left > 0 and p_right > 0 else np.nan
            for p_left, p_right in zip(proportions_3_left, proportions_3_right)
        ]
        
        all_free_energy_differences.append(free_energy_differences)

    print("Computing statistics and generating plot...")
    all_free_energy_differences = np.array(all_free_energy_differences)
    average_free_energy_differences = np.nanmean(all_free_energy_differences, axis=0)
    std_free_energy_differences = np.nanstd(all_free_energy_differences, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sample_numbers = range(1, len(average_free_energy_differences) + 1)
    ax.plot(sample_numbers, average_free_energy_differences, 'b-', label='Average Free Energy Difference (3 Left vs 3 Right)')
    ax.fill_between(sample_numbers, 
                    average_free_energy_differences - std_free_energy_differences, 
                    average_free_energy_differences + std_free_energy_differences, 
                    color='b', alpha=0.2, label='Standard Deviation')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Free Energy Difference (k_B T)')
    ax.set_title('Average Free Energy Difference Between 3 Left and 3 Right vs Sample Number')
    ax.legend()

    plt.tight_layout()

    # Create replots directory if it doesn't exist
    replots_dir = os.path.join(os.path.dirname(base_dir), 'replots')
    os.makedirs(replots_dir, exist_ok=True)
    # Save the figure
    fig.savefig(os.path.join(replots_dir, 'free_energy_difference.png'), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_free_energy.py <base_directory>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    analyze_free_energy(base_dir)

# For direct execution without command line args
    # base_dir = "/home/n2401517d/my_workspace/HMC_NF/results/free_energy_acc_N3_0.00_KbT_N3_102400_samples_1000_big_moves_full/mc_runs"  # Uncomment and modify this line
    # analyze_free_energy(base_dir)  # Uncomment this line