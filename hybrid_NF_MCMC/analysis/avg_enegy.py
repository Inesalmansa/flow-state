import pandas as pd

# Read the CSV file
df = pd.read_csv("/home/n2401517d/my_workspace/HMC_NF/results/free_energy_N3_102400_0.25_KbT_samples_1000_big_moves/mc_runs/run_003/sampled_data.csv")

# Calculate average energy per particle
avg_energy_per_particle = df['energy_per_particle'].mean() 
print(avg_energy_per_particle)
avg_energy = avg_energy_per_particle *3
print(f"Average energy per particle: {avg_energy:.3f}")

