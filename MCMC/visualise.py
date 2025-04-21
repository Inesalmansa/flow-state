# visualise.py

import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
import os
from utils import set_icl_color_cycle, get_icl_heatmap_cmap

set_icl_color_cycle()

cmap_seq = get_icl_heatmap_cmap("sequential")
cmap_div = get_icl_heatmap_cmap("diverging")
cmap_multi = get_icl_heatmap_cmap("multistep")

# Existing function to visualize simulation samples
def visualise_simulation(chosen_samples, num_subplots=6, filename='simulation_visualisation.png'):
    """
    Visualize the simulation at specified samples.

    :param chosen_samples: List of samples, each containing
                           [cycle_number, energy_per_particle, density, pressure, box_size_x, box_size_y, particle_configuration].
    :param num_subplots: Number of subplots to create (default is 6).
    :param filename: Filename to save the plot.
    """
    if len(chosen_samples) != num_subplots:
        raise ValueError(f"Number of chosen samples ({len(chosen_samples)}) must match the number of subplots ({num_subplots}).")

    # Determine the grid size for subplots
    rows = ceil(sqrt(num_subplots))
    cols = ceil(num_subplots / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle('Simulation Stages', fontsize=16)

    for i, sample in enumerate(chosen_samples):
        # Unpack the sample
        try:
            cycle_number, energy_per_particle, density, pressure, box_size_x, box_size_y, particles = sample
        except ValueError:
            raise ValueError(f"Each sample must contain 7 elements: [cycle_number, energy_per_particle, density, pressure, box_size_x, box_size_y, particles]. Found {len(sample)} elements in sample {i}.")

        # Determine the current subplot
        ax = axes[i % rows, i // rows] if rows > 1 else axes[i // rows]
        # ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        ax.scatter(particles[:, 0], particles[:, 1], s=350)
        ax.set_title(f'Cycle {cycle_number}')
        ax.set_xlim(0, box_size_x)
        ax.set_ylim(0, box_size_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        # Add a text box with the current state information
        textstr = '\n'.join((
            f'Cycle: {cycle_number}',
            f'Energy/Particle: {energy_per_particle:.2f}',
            f'Pressure: {pressure:.2f}',
            f'Density: {density:.2f}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    # Hide any unused subplots
    total_subplots = rows * cols
    if total_subplots > num_subplots:
        for j in range(num_subplots, total_subplots):
            fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust to make room for the main title
    plt.savefig(filename, dpi=300)
    plt.close()  # Close the figure to free memory

# ------------------------------------------------------------------------
# New Function to Plot Concave Double Well Potential (Heatmap and Cross-Section)
# ------------------------------------------------------------------------
def plot_potential(box_siz_x, box_size_y, V0_list, r0, k, num_wells, output_path):
    """
    Plot a heatmap of the concave double well potential and a cross-section along the x-axis.

    The concave potential is modeled such that within a radius r0 of each well center the potential is nearly constant (V0_list)
    and then continuously transitions to 0 via a hyperbolic tangent function.

    Parameters:
    - box_siz_x: Width of the simulation box.
    - box_size_y: Height of the simulation box.
    - V0_list: Depth of the potential wells (typically a negative number for an attractive well).
    - r0: Radius for the flat-bottom region of the well.
    - k: Steepness parameter for the transition.
    - num_wells: Number of potential wells (1 or 2).
    - output_path: Directory to save the plots.
    """
    def double_well_concave(position, box_size_x, box_size_y, V0_list=[-0.5,-0.8], r0=1.0, k=20.0, num_wells=2):
        """
        Compute the concave double well potential using a smooth, step-like function.

        The potential is nearly constant (V0_list) within a radius r0 of each well center and decays smoothly to 0 outside r0
        via a hyperbolic tangent transition.

        Parameters:
        - position : array-like, with shape (N, 2) for multiple positions or (2,) for a single position.
        - box_size_x : float, simulation box width.
        - box_size_y : float, simulation box height.
        - V0_list : list, depths of the potential wels.
        - r0 : float, radius defining the flat bottom region.
        - k : float, steepness of the transition.
        - num_wells : int, number of wells (1 or 2).

        Returns:
        - V : Potential energy value(s) at the provided position(s).
        """
        if V0_list is None:
            V0_list = [-4.0] * num_wells  # Default depths for each well

        position = np.atleast_2d(position)  # Ensure position is at least 2D

        x = position[:, 0]
        y = position[:, 1]

        # Define the centers of the wells based on num_wells
        centers = []
        if num_wells >= 1:
            centers.append([box_size_x / 4, box_size_y / 2])
        if num_wells == 2:
            centers.append([3 * box_size_x / 4, box_size_y / 2])
        centers = np.array(centers)

        V = np.zeros_like(x, dtype=np.float64)

        # Convert V0_list to numpy array of floats if it's not already
        V0_list = np.array(V0_list, dtype=np.float64)
        print(V0_list)

        for i, center in enumerate(centers):
            dx = x - center[0]
            dy = y - center[1]

            # Apply minimum image convention for periodic boundary conditions
            dx -= box_size_x * np.round(dx / box_size_x)
            dy -= box_size_y * np.round(dy / box_size_y)

            # Compute the radial distance from the well center
            r = np.sqrt(dx**2 + dy**2)

            # Define a smooth step function:
            transition = 0.5 * (1 + np.tanh(k * (r - r0)))

            V += V0_list[i] * (1 - transition)

        if V.shape[0] == 1:
            return V[0]  # Return scalar if input was a single position
        
        return V


    # --- Section 1: Contour Plot (Heatmap) of the Concave Potential ---
    num_points = 500  # Number of grid points in each dimension
    x_vals = np.linspace(0, box_siz_x, num_points)
    y_vals = np.linspace(0, box_size_y, num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    V_vals = double_well_concave(grid_points, box_siz_x, box_size_y, V0_list, r0, k, num_wells)
    V_vals = V_vals.reshape(X.shape)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, V_vals, levels=100, cmap=cmap_seq)
    cbar = plt.colorbar(contour, label=r'Potential Energy / $k_BT$')
    min_tick = np.floor(np.min(V_vals))
    max_tick = np.ceil(np.max(V_vals))
    cbar.set_ticks(np.arange(min_tick, max_tick + 1, 1))
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    heatmap_filename = os.path.join(output_path, 'double_well_concave_potential_heatmap.png')
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()
    print(f"Concave potential heatmap saved to {heatmap_filename}")

    # --- Section 2: Potential Energy vs. x-Coordinate at y = box_size_y/2 ---
    y_fixed = box_size_y / 2
    x_slice = x_vals  # All x positions in the grid
    points_line = np.column_stack((x_slice, np.full_like(x_slice, y_fixed)))
    V_slice = double_well_concave(points_line, box_siz_x, box_size_y, V0_list, r0, k, num_wells)

    # Optionally, print the potential energy at x = 0 and at x = box_siz_x
    V_at_0 = double_well_concave(np.array([[0, y_fixed]]), box_siz_x, box_size_y, V0_list, r0, k, num_wells)
    V_at_max = double_well_concave(np.array([[box_siz_x, y_fixed]]), box_siz_x, box_size_y, V0_list, r0, k, num_wells)
    print(f"Potential at x = 0: {V_at_0}")
    print(f"Potential at x = {box_siz_x}: {V_at_max}")

    plt.figure(figsize=(8, 4))
    plt.plot(x_slice, V_slice, linewidth=2, label=f'$y={box_size_y/2:.1f}$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'Potential Energy / $k_BT$')
    plt.legend(loc='lower right')
    plt.tight_layout()
    cross_section_filename = os.path.join(output_path, 'double_well_concave_potential_cross_section.png')
    plt.savefig(cross_section_filename, dpi=300)
    plt.close()
    print(f"Concave potential cross-section plot saved to {cross_section_filename}")

    # --- Create a more complex layout with uneven heights ---
    from matplotlib import gridspec
    
    fig = plt.figure(figsize=(11, 5))
    # Create a custom grid with 3 rows and 2 columns
    gs = gridspec.GridSpec(3, 2, width_ratios=[0.95, 1.05], height_ratios=[0.15, 0.7, 0.15])
    
    # Create the cross-section plot in the middle-left cell
    ax0 = fig.add_subplot(gs[1, 0])  # Middle row, left column
    
    # Create the heatmap in the right column, spanning all rows
    ax1 = fig.add_subplot(gs[:, 1])  # All rows, right column
    
    # --- Section 1: Contour Plot (Heatmap) of the Concave Potential ---
    num_points = 500  # Number of grid points in each dimension
    x_vals = np.linspace(0, box_siz_x, num_points)
    y_vals = np.linspace(0, box_size_y, num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    V_vals = double_well_concave(grid_points, box_siz_x, box_size_y, V0_list, r0, k, num_wells)
    V_vals = V_vals.reshape(X.shape)
    
    # Add heatmap to right subplot (spans full height)
    contour = ax1.contourf(X, Y, V_vals, levels=100, cmap=cmap_seq)
    cbar = fig.colorbar(contour, ax=ax1, label=r'Potential Energy / $k_BT$', 
                        shrink=1.0,       # Make colorbar 80% of the axis height
                        fraction=0.046,   # Make colorbar thinner
                        pad=0.04)   
    cbar.set_label(r'Potential Energy / $k_BT$', fontsize=12)  # Set fontsize for the label
    min_tick = np.floor(np.min(V_vals))
    max_tick = np.ceil(np.max(V_vals))
    cbar.set_ticks(np.arange(min_tick, max_tick + 1, 1))
    
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r'$x$',fontsize=12)
    ax1.set_ylabel(r'$y$',fontsize=12)

    # --- Section 2: Potential Energy vs. x-Coordinate at y = box_size_y/2 ---
    y_fixed = box_size_y / 2
    x_slice = x_vals  # All x positions in the grid
    points_line = np.column_stack((x_slice, np.full_like(x_slice, y_fixed)))
    V_slice = double_well_concave(points_line, box_siz_x, box_size_y, V0_list, r0, k, num_wells)

    # Add cross-section to left subplot (only middle portion of height)
    ax0.plot(x_slice, V_slice, linewidth=2, label=f'$y={box_size_y/2:.1f}$')
    ax0.set_xlim(0,box_siz_x)
    ax0.set_xlabel(r'$x$',fontsize=12)
    ax0.set_ylabel(r'Potential Energy / $k_BT$',fontsize=12)
    ax0.legend(fontsize="small")
    
    # Add labels 'A' and 'B' to wells in both plots
    if num_wells >= 1:
        well_x1 = box_siz_x / 4
        well_y = box_size_y / 2
        # Label for well A in the cross-section plot (color C1)
        ax0.text(well_x1, V_slice[np.abs(x_slice - well_x1).argmin()] + 1.5, '$\mathbf{A}$', 
                 color='C0', fontsize=16, ha='center', va='top', fontweight='bold')
        # Label for well A in the heatmap (white)
        ax1.text(well_x1, well_y, '$\mathbf{A}$', color='white', fontsize=16, 
                 ha='center', va='center', fontweight='bold')
    
    if num_wells == 2:
        well_x2 = 3 * box_siz_x / 4
        # Label for well B in the cross-section plot (color C1)
        ax0.text(well_x2, V_slice[np.abs(x_slice - well_x2).argmin()] + 1.5, '$\mathbf{B}$', 
                 color='C0', fontsize=16, ha='center', va='top', fontweight='bold')
        # Label for well B in the heatmap (white)
        ax1.text(well_x2, well_y, '$\mathbf{B}$', color='white', fontsize=16, 
                 ha='center', va='center', fontweight='bold')
    
    # Save the combined figure
    plt.tight_layout()
    combined_filename_png = os.path.join(output_path, 'double_well_concave_potential_combined.png')
    combined_filename_svg = os.path.join(output_path, 'double_well_concave_potential_combined.svg')
    fig.savefig(combined_filename_png, dpi=300, format='png')
    fig.savefig(combined_filename_svg, format='svg')
    plt.close()


    # **Old Function to Plot Potential Heatmap and Cross-Section**
def plot_potential_old(w, h, V0_list, a, num_wells, output_path):
    """
    Plot a heatmap of the double well potential and a cross-section in the x-direction.

    :param w: Width of the simulation box.
    :param h: Height of the simulation box.
    :param V0_list: Depth of the potential wells.
    :param a: Width parameter for the Gaussian wells.
    :param num_wells: Number of potential wells (1 or 2).
    :param output_path: Directory to save the plots.
    """
    # Define the double well potential function with optional number of wells
    def double_well_potential(x, y, w, h, V0_list=1.0, a=10.0, num_wells=2):
        """
        Calculate the double well potential at position (x, y) within a periodic box.
        Allows for either one or two wells.

        Parameters:
        - x, y: Coordinates of the particle (can be scalars or numpy arrays).
        - w, h: Width and height of the simulation box.
        - V0_list: Depth of the potential wells.
        - a: Controls the width of the Gaussian wells.
        - num_wells: Number of wells (1 or 2).

        Returns:
        - V: Potential energy at (x, y).
        """
        # Define the centers of the wells based on num_wells
        centers = []
        if num_wells >= 1:
            centers.append([w / 4, h / 2])
        if num_wells == 2:
            centers.append([3 * w / 4, h / 2])

        centers = np.array(centers)

        # Initialize potential energy array
        V = np.zeros_like(x, dtype=np.float64)

        for center in centers:
            dx = x - center[0]
            dy = y - center[1]

            # Apply minimum image convention for periodic boundary conditions
            dx -= w * np.round(dx / w)
            dy -= h * np.round(dy / h)

            r_squared = dx**2 + dy**2

            # Gaussian well
            V += V0_list * np.exp(-a * r_squared)

        return V

    # --- Section 1: Contour Plot of the Double Well Potential ---

    # Create a grid of points
    num_points = 500  # Number of points in each dimension
    x = np.linspace(0, w, num_points)
    y = np.linspace(0, h, num_points)
    X, Y = np.meshgrid(x, y)

    # Calculate the potential at each grid point
    V = double_well_potential(X, Y, w, h, V0_list, a, num_wells=num_wells)

    # Plot the potential using a contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, V, levels=100, cmap='viridis')
    plt.colorbar(contour, label='Potential Energy / $k_BT$')

    # Plot the centers of the wells
    centers = []
    if num_wells >= 1:
        centers.append([w / 4, h / 2])
    if num_wells == 2:
        centers.append([3 * w / 4, h / 2])
    centers = np.array(centers)
    plt.scatter(centers[:,0], centers[:,1], color='red', marker='o', label='Potential Minima')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Double Well Potential in 2D with {num_wells} Well(s)')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.tight_layout()
    plt.legend()
    plt.tight_layout()
    heatmap_filename = os.path.join(output_path, f'double_well_potential_heatmap.png')
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()
    print(f"Potential heatmap saved to {heatmap_filename}")

    # --- Section 2: Potential Energy vs. x-Coordinate at y = h/2 ---

    # Define the y-coordinate at which to evaluate the potential
    y_fixed = h / 2

    # Extract the x-coordinates and corresponding potential energies
    x_slice = x  # All x positions
    y_slice = np.full_like(x_slice, y_fixed)  # y = h/2 for all points
    V_slice = double_well_potential(x_slice, y_slice, w, h, V0_list, a, num_wells=num_wells)

    # Print the potential at x = 0 and x = w (assuming w is the box width)
    V_at_0 = double_well_potential(0, y_fixed, w, h, V0_list, a, num_wells=num_wells)
    V_at_w = double_well_potential(w, y_fixed, w, h, V0_list, a, num_wells=num_wells)
    print(f"Potential at x = 0: {V_at_0}")
    print(f"Potential at x = {w}: {V_at_w}")

    # Plot the potential energy vs. x-coordinate
    plt.figure(figsize=(8, 4))
    plt.plot(x_slice, V_slice, color='blue', linewidth=2, label='Potential Energy')

    # Highlight the minima
    for i, center in enumerate(centers):
        V_min = double_well_potential(center[0], y_fixed, w, h, V0_list, a, num_wells=num_wells)
        plt.scatter(center[0], V_min, color='red', zorder=5, label='Potential Minimum' if i == 0 else "")

    plt.title(f'Double Well Potential along y = {y_fixed} with {num_wells} Well(s)')
    plt.xlabel(r'$x$')
    plt.ylabel('Potential Energy / $k_BT$')
    plt.legend()
    plt.tight_layout()
    cross_section_filename = os.path.join(output_path, f'double_well_potential_cross_section.png')
    plt.savefig(cross_section_filename, dpi=300)
    plt.close()
    print(f"Potential cross-section plot saved to {cross_section_filename}")