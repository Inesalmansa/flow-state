import argparse
import numpy as np
import os
from simulation_box import SimulationBox
from monte_carlo import MonteCarlo
from initialise import (
    initialise_fcc,
    initialise_fcc_left_half,
    initialise_fcc_right_half,
    initialise_low_left,
    initialise_low_right
)
from visualise import visualise_simulation, plot_potential
import csv

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NVT Monte Carlo simulation")

    # Simulation parameters
    parser.add_argument("--temperature", type=float, required=True, help="Temperature of the simulation")
    parser.add_argument("--num_particles", type=int, default=64, help="Number of particles in the simulation")
    parser.add_argument("--initial_rho", type=float, required=True, help="Initial density of the system")
    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="Aspect ratio (box_size_x / box_size_y)")
    parser.add_argument("--visualise", action='store_true', help="Enable visualization of the simulation")
    parser.add_argument("--checking", action='store_true', help="Enable checking for density and box volume consistency")
    parser.add_argument("--equilibration_steps", type=int, required=True, help="Number of equilibration steps")
    parser.add_argument("--production_steps", type=int, required=True, help="Number of production steps")
    parser.add_argument("--sampling_frequency", type=int, required=True, help="Frequency of sampling during the simulation")
    parser.add_argument("--adjusting_frequency", type=int, required=True, help="Frequency of adjusting displacements")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output files")
    parser.add_argument("--experiment_id", type=str, required=True, help="Unique identifier for the experiment")
    parser.add_argument("--time_calc", action='store_true', help="Enable timing calculations") 

    # **New Arguments for External Potential**
    parser.add_argument("--num_wells", type=int, choices=[0, 1, 2], default=0,
                        help="Number of external potential wells (0, 1, or 2). Default is 0 (no wells).")
    parser.add_argument("--V0_list", type=float, nargs='+', default=[-0.5, -0.5],
                        help="List of potential well depths. Default is [-0.5, -0.5].")
    parser.add_argument("--k", type=float, default=10,
                        help="Width parameter for the Gaussian wells. Default is 10.")
    parser.add_argument("--r0", type=float, default=1.0,
                        help="radius of bottom of well")
    
    parser.add_argument("--initialisation_type", type=str, choices=['all', 'left_half', 'right_half'], default='all',
                        help="Type of initialisation for particles: 'all' for full FCC, 'left_half' for left half FCC, 'right_half' for right half FCC. Default is 'all'.")
    parser.add_argument("--seed", type=int, required=True, help="set the random numpy seed")
    parser.add_argument("--initial_max_displacement", type=float, default=0.1, help="Initial maximum displacement for Monte Carlo moves")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    print(f"Random seed set to {args.seed}.")

    # Initialize particles based on the number of particles and the initialisation type.
    #
    # If there are between 2 and 12 particles, we use the "low" initialisation functions.
    # Otherwise, we use the FCC-based functions.
    if 2 <= args.num_particles <= 12:
        print(f"Using low particle initialization for {args.num_particles} particles.")
        if args.initialisation_type == 'left_half':
            particles, sim_box = initialise_low_left(
                num_particles=args.num_particles,
                rho=args.initial_rho,
                aspect_ratio=args.aspect_ratio,
                visualise=args.visualise,
                checking=args.checking
            )
        elif args.initialisation_type == 'right_half':
            particles, sim_box = initialise_low_right(
                num_particles=args.num_particles,
                rho=args.initial_rho,
                aspect_ratio=args.aspect_ratio,
                visualise=args.visualise,
                checking=args.checking
            )
        else:
            raise ValueError(f"Unknown initialisation type: {args.initialisation_type}")
    else:
        # Use FCC-based initialization for more than 12 particles
        if args.initialisation_type == 'all':
            particles, sim_box = initialise_fcc(
                num_particles=args.num_particles,
                rho=args.initial_rho,
                aspect_ratio=args.aspect_ratio,
                visualise=args.visualise,
                checking=args.checking
            )
        elif args.initialisation_type == 'left_half':
            particles, sim_box = initialise_fcc_left_half(
                num_particles=args.num_particles,
                rho=args.initial_rho,
                aspect_ratio=args.aspect_ratio,
                visualise=args.visualise,
                checking=args.checking
            )
        elif args.initialisation_type == 'right_half':
            particles, sim_box = initialise_fcc_right_half(
                num_particles=args.num_particles,
                rho=args.initial_rho,
                aspect_ratio=args.aspect_ratio,
                visualise=args.visualise,
                checking=args.checking
            )
        else:
            raise ValueError(f"Unknown initialisation type: {args.initialisation_type}")
        
    print(f"Initialized simulation box with sizes: {sim_box.box_size_x:.3f} x {sim_box.box_size_y:.3f}")
    print(f"Initialized {len(particles)} particles.")
    
    # **Plot the Double Well Potential**
    plot_potential(
        box_siz_x=sim_box.box_size_x,
        box_size_y=sim_box.box_size_y,
        V0_list=args.V0_list,
        r0=args.r0,
        k=args.k,
        num_wells=args.num_wells,
        output_path=args.output_path
    )

    # Initialize MonteCarlo
    mc = MonteCarlo(
        particles=particles,
        sim_box=sim_box,
        temperature=args.temperature,
        num_particles=args.num_particles,
        num_wells=args.num_wells,    # Pass number of wells
        V0_list=args.V0_list,                  # Pass well depth
        r0=args.r0,
        k=args.k,
        initial_max_displacement=args.initial_max_displacement,
        target_acceptance=0.5,
        timing=args.time_calc,
        checking=True
    )
    print("Monte Carlo simulation initialized.")

    # **Check for Valid Number of Wells**
    if args.num_wells > 0:
        print(f"External Potential: {args.num_wells} well(s), V0_list={args.V0_list}, k={args.k}, r0={args.r0}")
    else:
        print("No external potential wells applied.")

    # Sample initial state
    sampled_data = []
    initial_sample = mc.sample(0)
    sampled_data.append(initial_sample)
    print("Initial state sampled.")

    # Equilibration phase
    print(f"Starting equilibration phase: {args.equilibration_steps} steps")
    for step in range(1, args.equilibration_steps + 1):
        mc.particle_displacement()
        if step % args.adjusting_frequency == 0:
            mc.adjust_displacement()
        if step % args.sampling_frequency == 0:
            sample = mc.sample(step)
            sampled_data.append(sample)

    print("Equilibration phase completed.")

    # Production phase
    print(f"Starting production phase: {args.production_steps} steps")
    for step in range(1, args.production_steps + 1):
        mc.particle_displacement()
        current_step = args.equilibration_steps + step
        if current_step % args.adjusting_frequency == 0:
            mc.adjust_displacement()
        if step % args.sampling_frequency == 0:
            sample = mc.sample(current_step)
            sampled_data.append(sample)

    print("Production phase completed.")

    # Extract production configurations from the sampled data (only those with cycle_number > equilibration steps)
    production_configs = [sample[6] for sample in sampled_data if sample[0] > args.equilibration_steps]
    # Determine the translation vector using the box dimensions from the first production sample
    first_prod_sample = next(sample for sample in sampled_data if sample[0] > args.equilibration_steps)
    translation_vector = np.array([-first_prod_sample[4] / 2, -first_prod_sample[5] / 2])
    # Translate each particle in each production configuration individually
    production_configs = [np.array([particle + translation_vector for particle in config]) for config in production_configs]

    # Save the production configurations into an NPZ file
    npz_filename = os.path.join(args.output_path, 'production_configurations.npz')
    np.savez(npz_filename, configurations=production_configs)
    print(f"Production configurations saved to {npz_filename}")

    production_samples = [sample for sample in sampled_data if sample[0] > args.equilibration_steps]
    if production_samples:
        total_energies = [sample[1] * args.num_particles for sample in production_samples]
        avg_total_energy = sum(total_energies) / len(total_energies)
        print(f"Average total energy during production: {avg_total_energy}")
    else:
        print("No production samples found to calculate average energy.")

    # Save sampled data to CSV
    csv_filename = os.path.join(args.output_path, 'sampled_data.csv')
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
            for data in sampled_data:
                cycle_number, energy_per_particle, density, pressure, box_size_x, box_size_y, particles = data
                # Flatten the particle positions for CSV
                particles_flat = particles.flatten()
                writer.writerow([
                    cycle_number,
                    energy_per_particle,
                    density,
                    pressure,
                    box_size_x,
                    box_size_y,
                    particles_flat.tolist()  # Convert to list for JSON-like storage
                ])
        print(f"Sampled data successfully saved to {csv_filename}")
    except Exception as e:
        print(f"Error: Failed to save sampled data to {csv_filename}. Error: {e}")

    # Visualize if requested
    if args.visualise:
        # Choose samples for visualization
        num_samples = len(sampled_data)
        if num_samples >= 4:
            chosen_samples = [
                sampled_data[0],  # Initial configuration
                sampled_data[args.equilibration_steps // args.sampling_frequency],  # Point after equilibration
                sampled_data[(args.equilibration_steps + (num_samples - args.equilibration_steps) // 2) // args.sampling_frequency],  # Mid production
                sampled_data[-1]  # Final production
            ]
            print("Selected samples for visualization.")
        else:
            print("Not enough samples collected for visualization. Using all available samples.")
            chosen_samples = sampled_data  # Fallback to whatever samples are available

        visualization_path = os.path.join(args.output_path, 'simulation_visualisation.png')
        try:
            visualise_simulation(
                chosen_samples,
                num_subplots=len(chosen_samples),
                filename=visualization_path
            )
            print(f"Simulation visualization saved to {visualization_path}")
        except Exception as e:
            print(f"Error: Failed to save simulation visualization. Error: {e}")

    # Log density and box volume if checking is enabled
    if args.checking:
        final_density = sampled_data[-1][2]  # Assuming density is the third element
        final_box_size_x = sampled_data[-1][4]  # Fifth element
        final_box_size_y = sampled_data[-1][5]  # Sixth element
        print(f"Final Density: {final_density:.3f}")
        print(f"Final Box Size: {final_box_size_x:.3f} x {final_box_size_y:.3f}")

    # Summary of Monte Carlo moves
    try:
        acceptance_ratio = mc.accepted_displacement / mc.attempts_displacement if mc.attempts_displacement > 0 else 0
        print(f"Displacement moves: Accepted = {mc.accepted_displacement}, "
              f"Attempted = {mc.attempts_displacement}, Ratio = {acceptance_ratio:.2f}")
    except AttributeError:
        print("Warning: Monte Carlo move attributes not found.")

if __name__ == "__main__":
    main()
