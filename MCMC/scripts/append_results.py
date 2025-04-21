import fcntl
import csv
import sys
import os

def append_results(results_csv, output_path, temperature, equilibration_steps):
    """
    Append simulation results to the main CSV file.

    Parameters:
    - results_csv (str): Path to the main results CSV file.
    - output_path (str): Path to the simulation output directory.
    - temperature (float): Temperature of the simulation run.
    - equilibration_steps (int): Number of equilibration steps.

    The function reads 'sampled_data.csv' from the output_path, calculates the average
    pressure after equilibration steps, computes the average aspect ratio, and appends
    these along with temperature and density to the main results CSV.
    """
    # Path to the sampled_data.csv file
    sampled_data_file = os.path.join(output_path, 'sampled_data.csv')

    # Lists to store extracted values
    pressures = []
    aspect_ratios = []
    densities = []

    try:
        with open(sampled_data_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip the header
            for row in reader:
                cycle_number = int(row[0])
                if cycle_number > equilibration_steps:
                    try:
                        density = float(row[2])      # Assuming density is at index 2
                        pressure = float(row[3])     # Assuming pressure is at index 3
                        box_size_x = float(row[4])   # Assuming box_size_x is at index 4
                        box_size_y = float(row[5])   # Assuming box_size_y is at index 5
                        densities.append(density)
                        pressures.append(pressure)
                        # Compute aspect ratio for each sample and store
                        if box_size_y != 0:
                            aspect_ratio = box_size_x / box_size_y
                            aspect_ratios.append(aspect_ratio)
                        else:
                            print(f"Warning: box_size_y is zero at cycle {cycle_number}. Skipping aspect ratio computation.")
                    except ValueError as ve:
                        print(f"Warning: Could not convert data types in row {row}. Error: {ve}. Skipping this row.")
    except FileNotFoundError:
        print(f"Error: The file '{sampled_data_file}' does not exist.")
        return
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading '{sampled_data_file}'. Error: {e}")
        return

    if not pressures:
        print("Error: No production data found after equilibration steps.")
        return

    # Calculate average pressure and aspect ratio
    average_pressure = sum(pressures) / len(pressures)
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    average_density = sum(densities) / len(densities)  # Should be constant in NVT

    # Debugging information (optional)
    # print(f"Average Pressure: {average_pressure}")
    # print(f"Average Aspect Ratio: {average_aspect_ratio}")
    # print(f"Average Density: {average_density}")

    # Append the results to the main CSV file with file locking
    try:
        with open(results_csv, 'a', newline='') as csvfile:
            fcntl.flock(csvfile, fcntl.LOCK_EX)  # Acquire an exclusive lock
            writer = csv.writer(csvfile)
            writer.writerow([temperature, f"{average_aspect_ratio:.3f}", f"{average_density:.3f}", f"{average_pressure:.3f}"])
            fcntl.flock(csvfile, fcntl.LOCK_UN)  # Release the lock
        print(f"Results appended successfully for Temperature: {temperature}")
    except Exception as e:
        print(f"Error: Failed to append results to '{results_csv}'. Error: {e}")
        return

    print(f"Average Density: {average_density:.3f}")
    print(f"Average Pressure: {average_pressure:.3f}")
    print(f"Average Aspect Ratio: {average_aspect_ratio:.3f}")

if __name__ == "__main__":
    # Expecting arguments: results_csv, output_path, temperature, equilibration_steps
    if len(sys.argv) != 5:
        print("Usage: append_results.py <results_csv> <output_path> <temperature> <equilibration_steps>")
        sys.exit(1)

    results_csv = sys.argv[1]
    output_path = sys.argv[2]
    try:
        temperature = float(sys.argv[3])
    except ValueError:
        print("Error: Temperature must be a float.")
        sys.exit(1)
    try:
        equilibration_steps = int(sys.argv[4])
    except ValueError:
        print("Error: Equilibration steps must be an integer.")
        sys.exit(1)

    append_results(results_csv, output_path, temperature, equilibration_steps)
