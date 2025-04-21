import numpy as np
from itertools import permutations
from utils import classify_particles

def get_distances(pos1, pos2, box_size):
    """Calculate distances between particles accounting for periodic boundary conditions"""
    diff = pos1 - pos2
    diff -= box_size * np.round(diff / box_size)
    return np.sqrt(np.sum(diff**2, axis=1))

def configurations_equal(conf1, conf2, class1, class2, box_size, tolerance=1e-6):
    """
    Check if two configurations are equivalent considering periodic boundaries,
    particle permutations, and their classifications
    """
    n_particles = len(conf1)
    particle_indices = list(range(n_particles))
    
    # Try all possible permutations of particles
    for perm in permutations(particle_indices):
        conf2_permuted = conf2[list(perm)]
        class2_permuted = [class2[i] for i in perm]
        
        # Check if classifications match
        if class1 != class2_permuted:
            continue
            
        # Check if positions match with periodic boundaries
        distances = get_distances(conf1, conf2_permuted, box_size)
        if np.all(distances < tolerance):
            return True
    return False

# Load configurations
path = "/home/n2401517d/my_workspace/flow_state/NF/data/samples_N3_rho_0.03.npz"
configs = np.load(path)
positions = configs[list(configs.keys())[0]]

# Parameters
halfbox = 5.0
box_size = 2 * halfbox
r0 = 1.2

# Get classifications for all configurations
classifications = classify_particles(positions, halfbox, r0)

# Find unique configurations
total_samples = len(positions)
unique_indices = []
duplicate_counts = {}  # To track how many times each unique configuration appears

for i in range(total_samples):
    is_unique = True
    for j in unique_indices:
        if configurations_equal(positions[i], positions[j], 
                              classifications[i], classifications[j], 
                              box_size):
            is_unique = False
            duplicate_counts[j] = duplicate_counts.get(j, 1) + 1
            break
    if is_unique:
        unique_indices.append(i)
        duplicate_counts[i] = 1

n_unique = len(unique_indices)

# Calculate effective sample size
# Using Kish's effective sample size formula
# ESS = (sum(weights))^2 / sum(weights^2)
weights = np.array(list(duplicate_counts.values()))
ess = np.sum(weights)**2 / np.sum(weights**2)

# Print statistics
print(f"Total number of samples: {total_samples}")
print(f"Number of unique configurations: {n_unique}")
print(f"Effective Sample Size (ESS): {ess:.2f}")

# Print distribution of duplicates
print("\nDuplicate distribution:")
duplicate_dist = {}
for count in duplicate_counts.values():
    duplicate_dist[count] = duplicate_dist.get(count, 0) + 1

for count, freq in sorted(duplicate_dist.items()):
    print(f"Configurations appearing {count} times: {freq}")

# Print unique configuration types
print("\nUnique configuration types:")
unique_types = {}
for idx in unique_indices:
    config_type = ''.join(sorted(classifications[idx]))  # Sort to group AAB and ABA together
    unique_types[config_type] = unique_types.get(config_type, 0) + 1

for config_type, count in sorted(unique_types.items()):
    print(f"Type {config_type}: {count} unique configurations")

# # Optional: Save unique configurations
# unique_configs = positions[unique_indices]
# np.savez(path.replace('.npz', '_unique.npz'), 
#          unique_configurations=unique_configs) 