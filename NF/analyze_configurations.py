import numpy as np
from utils import classify_particles

# Load configurations
path = "/home/n2401517d/my_workspace/flow_state/NF/data/samples_N3_rho_0.03.npz"
configs = np.load(path)
positions = configs[list(configs.keys())[0]]  # Get the first key's data

# Parameters
halfbox = 5.0
r0 = 1.2

# Classify particles
classifications = classify_particles(positions, halfbox, r0)

# Count different configurations
total_samples = len(classifications)
counts = {
    'AAA': 0,
    'BBB': 0,
    'AAB': 0,
    'ABA': 0,
    'BAA': 0,
    'ABB': 0,
    'BAB': 0,
    'BBA': 0,
    'other': 0
}

for config in classifications:
    # Convert config list to string for easy pattern matching
    config_str = ''.join(config)
    
    if config_str == 'AAA':
        counts['AAA'] += 1
    elif config_str == 'BBB':
        counts['BBB'] += 1
    elif config_str == 'AAB':
        counts['AAB'] += 1
    elif config_str == 'ABA':
        counts['ABA'] += 1
    elif config_str == 'BAA':
        counts['BAA'] += 1
    elif config_str == 'ABB':
        counts['ABB'] += 1
    elif config_str == 'BAB':
        counts['BAB'] += 1
    elif config_str == 'BBA':
        counts['BBA'] += 1
    else:
        counts['other'] += 1

# Print statistics
print(f"Total number of samples: {total_samples}")
print("\nConfiguration counts:")
for config_type, count in counts.items():
    percentage = (count / total_samples) * 100
    print(f"{config_type}: {count} ({percentage:.2f}%)")

# Optional: Print grouped statistics
print("\nGrouped statistics:")
three_a = counts['AAA']
three_b = counts['BBB']
two_a_one_b = counts['AAB'] + counts['ABA'] + counts['BAA']
one_a_two_b = counts['ABB'] + counts['BAB'] + counts['BBA']
other = counts['other']

print(f"3A (AAA): {three_a} ({(three_a/total_samples)*100:.2f}%)")
print(f"3B (BBB): {three_b} ({(three_b/total_samples)*100:.2f}%)")
print(f"2A1B (AAB+ABA+BAA): {two_a_one_b} ({(two_a_one_b/total_samples)*100:.2f}%)")
print(f"1A2B (ABB+BAB+BBA): {one_a_two_b} ({(one_a_two_b/total_samples)*100:.2f}%)")
print(f"Other: {other} ({(other/total_samples)*100:.2f}%)") 