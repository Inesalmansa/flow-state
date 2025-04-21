import numpy as np
import matplotlib.pyplot as plt
from simulation_box import SimulationBox
import math
from utils import set_icl_color_cycle
set_icl_color_cycle()

def initialise_fcc(num_particles=48, rho=0.5, aspect_ratio=1.5, visualise=False, checking=False):
    """
    Initialize particles in a 2D system into an FCC-like lattice structure with a given density
    and aspect ratio.

    This function generates a candidate lattice with two sublattices per grid cell (one at the
    integer grid position and one offset by half a grid spacing). If the total candidate number is 
    larger than the desired particle count, the function picks the particles that lie closest to the 
    center of the simulation box so that the final configuration is well–centered and evenly spread.

    :param num_particles: Number of particles to initialize.
    :param rho: 2D density of the system.
    :param aspect_ratio: Ratio box_size_x / box_size_y.
    :param visualise: Boolean flag to visualize the initial configuration.
    :param checking: Boolean flag to print debug information.
    :return: (particles, sim_box) where 'particles' is an (N x 2) numpy array of positions,
             and 'sim_box' is the SimulationBox instance.
    """
    # Calculate the simulation box dimensions from density and number of particles.
    # area = num_particles / rho.
    area = num_particles / rho
    box_size_x = np.sqrt(area * aspect_ratio)
    box_size_y = np.sqrt(area / aspect_ratio)
    
    sim_box = SimulationBox(box_size_x, box_size_y)
    
    # ---- Determine the grid parameters for candidate sites ----
    #
    # We wish to generate candidates arranged in two interpenetrating (FCC–like) sublattices.
    # In each rectangular grid cell we will have two candidate positions:
    #   - One at (i*dx, j*dy)
    #   - One at ((i+0.5)*dx, (j+0.5)*dy)
    #
    # We choose the number of grid cells in the x– and y–directions such that
    #   2 * nx * ny >= num_particles.
    # A natural choice is to use the aspect ratio of the simulation box.
    #
    # We start by estimating nx from the aspect ratio. (Since roughly half the particles come from each sublattice.)
    
    # Estimate nx from aspect ratio; note: num_cells_total ~ num_particles/2.
    nx = math.ceil(np.sqrt(num_particles/2 * aspect_ratio))
    ny = math.ceil(num_particles/(2 * nx))
    
    # Ensure we have enough candidate points.
    candidate_count = 2 * nx * ny
    if checking:
        print(f"Candidate grid: nx = {nx}, ny = {ny}, total candidates = {candidate_count}")
    
    # Choose grid spacing so that the candidate points cover the entire simulation box.
    # For the (offset) points, the maximum x coordinate is (nx - 0.5)*dx and we wish that to equal box_size_x.
    dx = box_size_x / (nx - 0.5)
    dy = box_size_y / (ny - 0.5)
    
    if checking:
        print(f"Grid spacing: dx = {dx:.3f}, dy = {dy:.3f}")
        print(f"Box size: x = {box_size_x:.3f}, y = {box_size_y:.3f}")
        print(f"Target density: {num_particles / (box_size_x * box_size_y):.3f}")
    
    # ---- Generate candidate particle positions on an FCC-like (two-sublattice) grid ----
    candidate_positions = []
    for i in range(nx):
        for j in range(ny):
            # Sublattice A: at (i*dx, j*dy)
            pos_A = np.array([i * dx, j * dy])
            pos_A = sim_box.apply_pbc(pos_A)
            candidate_positions.append(pos_A)
            
            # Sublattice B: offset by half grid spacing
            pos_B = np.array([(i + 0.5) * dx, (j + 0.5) * dy])
            pos_B = sim_box.apply_pbc(pos_B)
            candidate_positions.append(pos_B)
    
    candidate_positions = np.array(candidate_positions)
    
    if checking:
        print(f"Generated {len(candidate_positions)} candidate positions.")
    
    # ---- Select the best num_particles candidates (center-out selection) ----
    # Compute the center of the box:
    center = np.array([box_size_x/2, box_size_y/2])
    
    # Compute squared distances to center for each candidate.
    distances2 = np.sum((candidate_positions - center)**2, axis=1)
    
    # Get the indices that would sort the positions by distance from center.
    sort_indices = np.argsort(distances2)
    
    # Select the first num_particles candidate positions.
    particles = candidate_positions[sort_indices[:num_particles]]
    
    if checking:
        # Print average distance from center for the selected particles.
        avg_dist = np.mean(np.sqrt(np.sum((particles - center)**2, axis=1)))
        print(f"Selected {num_particles} particles with average distance from center: {avg_dist:.3f}")
    
    # # ---- Visualization (optional) ----
    # if visualise:
    #     plt.figure(figsize=(6, 4))
    #     plt.scatter(particles[:, 0], particles[:, 1], s=40, edgecolors='k')
    #     plt.title(f"FCC-like Initialization\nDensity: {rho}, Aspect Ratio: {aspect_ratio}")
    #     plt.xlim(0, box_size_x)
    #     plt.ylim(0, box_size_y)
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.grid(True)
    #     plt.show()
    
    return particles, sim_box

def initialise_low_left(num_particles=2, rho=0.5, aspect_ratio=1.0, visualise=False, checking=False):
    """
    Initialize a small set (1 to 12) of particles positioned on the left side of the simulation box.
    The particles are arranged in a grid formation centered at (box_size_x/4, box_size_y/2) with sensible spacing.
    For num_particles=1, the particle is placed in the middle of the left well.

    Parameters:
      num_particles: int
          Number of particles to initialize (must be between 1 and 12).
      rho: float
          The 2D density of the system.
      aspect_ratio: float
          Ratio box_size_x / box_size_y.
      visualise: bool
          If True, a scatter plot of the initial positions is displayed.
      checking: bool
          If True, prints debug information.

    Returns:
      particles: (N,2) ndarray
          Array of particle positions.
      sim_box: SimulationBox instance
          The simulation box corresponding to the computed box dimensions.
    """
    if num_particles < 1 or num_particles > 12:
        raise ValueError("Number of particles for low initialization must be between 1 and 12.")

    # Compute box dimensions based on the specified density and aspect ratio.
    area = num_particles / rho
    box_size_x = np.sqrt(area * aspect_ratio)
    box_size_y = np.sqrt(area / aspect_ratio)
    sim_box = SimulationBox(box_size_x, box_size_y)

    if num_particles == 1:
        # For single particle, place it in the middle of the left well
        particles = np.array([[box_size_x / 4, box_size_y / 2]])
    else:
        # Define the group center on the left side.
        group_center = np.array([box_size_x / 4, box_size_y / 2])

        # Determine grid dimensions for arranging the particles.
        grid_cols = int(np.ceil(np.sqrt(num_particles)))
        grid_rows = int(np.ceil(num_particles / grid_cols))

        # Compute maximum allowable spacing so that the grid remains within the designated area.
        if grid_cols > 1:
            max_sep_x = box_size_x / (2 * (grid_cols - 1))
        else:
            max_sep_x = float('inf')
        if grid_rows > 1:
            max_sep_y = box_size_y / (grid_rows - 1)
        else:
            max_sep_y = float('inf')

        # default_sep = 0.2 * min(box_size_x, box_size_y)
        # Set default particle spacing to 1.5 and adjust if necessary to fit within the grid constraints.
        default_sep = 1.5
        spacing = min(default_sep, max_sep_x, max_sep_y)
        total_width = (grid_cols - 1) * spacing
        total_height = (grid_rows - 1) * spacing

        particles = []
        count = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if count >= num_particles:
                    break
                x = group_center[0] - total_width / 2 + col * spacing
                y = group_center[1] - total_height / 2 + row * spacing
                pos = np.array([x, y])
                pos = sim_box.apply_pbc(pos)
                particles.append(pos)
                count += 1
            if count >= num_particles:
                break

        particles = np.array(particles)

    if checking:
        print(f"Initialised low left with {num_particles} particles.")
    if visualise:
        plt.figure(figsize=(6, 4))
        plt.scatter(particles[:, 0], particles[:, 1], s=40, edgecolors='k')
        plt.title(f"Low Left Initialization: {num_particles} Particles")
        plt.xlim(0, box_size_x)
        plt.ylim(0, box_size_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    return particles, sim_box


def initialise_low_right(num_particles=2, rho=0.5, aspect_ratio=1.0, visualise=False, checking=False):
    """
    Initialize a small set (1 to 12) of particles positioned on the right side of the simulation box.
    The particles are arranged in a grid formation centered at (3*box_size_x/4, box_size_y/2) with sensible spacing.
    For num_particles=1, the particle is placed in the middle of the right well.

    Parameters:
      num_particles: int
          Number of particles to initialize (must be between 1 and 12).
      rho: float
          The 2D density of the system.
      aspect_ratio: float
          Ratio box_size_x / box_size_y.
      visualise: bool
          If True, a scatter plot of the initial positions is displayed.
      checking: bool
          If True, prints debug information.

    Returns:
      particles: (N,2) ndarray
          Array of particle positions.
      sim_box: SimulationBox instance
          The simulation box corresponding to the computed box dimensions.
    """
    if num_particles < 1 or num_particles > 12:
        raise ValueError("Number of particles for low initialization must be between 1 and 12.")

    # Compute box dimensions based on the specified density and aspect ratio.
    area = num_particles / rho
    box_size_x = np.sqrt(area * aspect_ratio)
    box_size_y = np.sqrt(area / aspect_ratio)
    sim_box = SimulationBox(box_size_x, box_size_y)

    if num_particles == 1:
        # For single particle, place it in the middle of the right well
        particles = np.array([[3 * box_size_x / 4, box_size_y / 2]])
    else:
        # Define the group center on the right side.
        group_center = np.array([3 * box_size_x / 4, box_size_y / 2])

        # Determine grid dimensions for arranging the particles.
        grid_cols = int(np.ceil(np.sqrt(num_particles)))
        grid_rows = int(np.ceil(num_particles / grid_cols))

        # Compute maximum allowable spacing so that the grid remains within the designated area.
        if grid_cols > 1:
            max_sep_x = box_size_x / (2 * (grid_cols - 1))
        else:
            max_sep_x = float('inf')
        if grid_rows > 1:
            max_sep_y = box_size_y / (grid_rows - 1)
        else:
            max_sep_y = float('inf')

        # Use a default spacing as a fraction of the box size and ensure the grid fits within the right half.
        # default_sep = 0.2 * min(box_size_x, box_size_y)
        default_sep = 1.5
        spacing = min(default_sep, max_sep_x, max_sep_y)
        total_width = (grid_cols - 1) * spacing
        total_height = (grid_rows - 1) * spacing

        particles = []
        count = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if count >= num_particles:
                    break
                x = group_center[0] - total_width / 2 + col * spacing
                y = group_center[1] - total_height / 2 + row * spacing
                pos = np.array([x, y])
                pos = sim_box.apply_pbc(pos)
                particles.append(pos)
                count += 1
            if count >= num_particles:
                break

        particles = np.array(particles)

    if checking:
        print(f"Initialised low right with {num_particles} particles.")
    if visualise:
        plt.figure(figsize=(6, 4))
        plt.scatter(particles[:, 0], particles[:, 1], s=40, edgecolors='k')
        plt.title(f"Low Right Initialization: {num_particles} Particles")
        plt.xlim(0, box_size_x)
        plt.ylim(0, box_size_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    return particles, sim_box


def initialise_fcc_old(num_particles=48, rho=0.5, aspect_ratio=1.5, visualise=False, checking=False):
    """
    Initialize particles in a 2D system into an FCC-like lattice structure with a given density
    and aspect ratio.

    :param num_particles: Number of particles to initialize.
    :param rho: 2D density of the system.
    :param aspect_ratio: Ratio box_size_x / box_size_y. Must be 1.5 for perfect particle numbers.
    :param visualise: Boolean flag to visualize the initial configuration.
    :param checking: Boolean flag to print debug information.
    :return: (particles, sim_box) where 'particles' is an Nx2 array of positions,
             and 'sim_box' is the SimulationBox instance.
    """
    # Calculate the number of unit cells based on aspect ratio and desired particle count
    n = int(np.sqrt(num_particles / 12))  # Since num_particles = 12n^2 for aspect_ratio=1.5
    if n < 1:
        raise ValueError("Number of particles is too small for the given aspect ratio.")
    
    # Derived number of unit cells per side
    num_per_side_x = 3 * n
    num_per_side_y = 2 * n
    
    if checking:
        print(f"Initializing FCC lattice with {num_particles} particles:")
        print(f"  Number of unit cells - X: {num_per_side_x}, Y: {num_per_side_y}")
        print(f"  Aspect Ratio: {aspect_ratio}")
    
    # Compute box dimensions
    area = num_particles / rho
    box_size_x = np.sqrt(area * aspect_ratio)
    box_size_y = np.sqrt(area / aspect_ratio)
    
    sim_box = SimulationBox(box_size_x, box_size_y)
    
    # Lattice spacing
    lattice_spacing_x = box_size_x / num_per_side_x
    lattice_spacing_y = box_size_y / num_per_side_y
    
    if checking:
        print(f"Lattice spacing - X: {lattice_spacing_x:.3f}, Y: {lattice_spacing_y:.3f}")
        print(f"Box size - X: {box_size_x:.3f}, Y: {box_size_y:.3f}")
        print(f"Calculated Density: {num_particles / (box_size_x * box_size_y):.3f}")
    
    # Initialize particle positions
    particles = []
    for i in range(num_per_side_x):
        for j in range(num_per_side_y):
            if len(particles) < num_particles:
                # Main lattice point
                position = np.array([i * lattice_spacing_x, j * lattice_spacing_y])
                position = sim_box.apply_pbc(position)
                particles.append(position)
                
                # FCC-like offset point
                if len(particles) < num_particles:
                    position = np.array([
                        (i + 0.5) * lattice_spacing_x,
                        (j + 0.5) * lattice_spacing_y
                    ])
                    position = sim_box.apply_pbc(position)
                    particles.append(position)
    
    particles = np.array(particles)
    
    # Optional consistency checks
    if checking:
        final_density = num_particles / (box_size_x * box_size_y)
        print(f"Final Density: {final_density:.3f}")
        print(f"Box Dimensions: {box_size_x:.3f} x {box_size_y:.3f}")
    
    # Visualization
    if visualise:
        plt.figure(figsize=(6, 4))
        plt.scatter(particles[:, 0], particles[:, 1], s=20, edgecolors='k')
        plt.title(f"FCC-like Initialization\nDensity: {rho}, Aspect Ratio: {aspect_ratio}")
        plt.xlim(0, box_size_x)
        plt.ylim(0, box_size_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
    
    return particles, sim_box

def initialise_fcc_left_half(num_particles=48, rho=0.5, aspect_ratio=1.5, visualise=False, checking=False):
    """
    Initialize particles in a 2D system into an FCC-like lattice structure with a given density
    and aspect ratio.

    :param num_particles: Number of particles to initialize.
    :param rho: 2D density of the system.
    :param aspect_ratio: Ratio box_size_x / box_size_y. Must be 1.5 for perfect particle numbers.
    :param visualise: Boolean flag to visualize the initial configuration.
    :param checking: Boolean flag to print debug information.
    :return: (particles, sim_box) where 'particles' is an Nx2 array of positions,
             and 'sim_box' is the SimulationBox instance.
    """
    # Calculate the number of unit cells based on aspect ratio and desired particle count
    n = int(np.sqrt(num_particles / 12))  # Since num_particles = 12n^2 for aspect_ratio=1.5
    if n < 1:
        raise ValueError("Number of particles is too small for the given aspect ratio.")
    
    # Derived number of unit cells per side
    num_per_side_x = 3 * n
    num_per_side_y = 2 * n
    
    if checking:
        print(f"Initializing FCC lattice with {num_particles} particles:")
        print(f"  Number of unit cells - X: {num_per_side_x}, Y: {num_per_side_y}")
        print(f"  Aspect Ratio: {aspect_ratio}")
    
    # Compute box dimensions
    area = num_particles / rho
    box_size_x = np.sqrt(area * aspect_ratio)
    box_size_y = np.sqrt(area / aspect_ratio)
    
    sim_box = SimulationBox(box_size_x, box_size_y)

    a_x = 0.15
    a_y = 0.5
    
    # Lattice spacing
    lattice_spacing_x = (box_size_x / num_per_side_x)*a_x
    lattice_spacing_y = (box_size_y / num_per_side_y)*a_y
    
    if checking:
        print(f"Lattice spacing - X: {lattice_spacing_x:.3f}, Y: {lattice_spacing_y:.3f}")
        print(f"Box size - X: {box_size_x:.3f}, Y: {box_size_y:.3f}")
        print(f"Calculated Density: {num_particles / (box_size_x * box_size_y):.3f}")
    
    # Initialize particle positions
    particles = []
    for i in range(num_per_side_x+2):
        for j in range(num_per_side_y):
            if len(particles) < num_particles:
                # Main lattice point
                position = np.array([box_size_x*(a_x) + (i * lattice_spacing_x),(box_size_y*(a_y*0.6)) + (j * lattice_spacing_y)])
                position = sim_box.apply_pbc(position)
                particles.append(position)
                
                # FCC-like offset point
                if len(particles) < num_particles:
                    position = np.array([
                        box_size_x*(a_x) + ((i + 0.5) * lattice_spacing_x),
                        box_size_y*(a_y*0.6) + ((j + 0.5) * lattice_spacing_y)
                    ])
                    position = sim_box.apply_pbc(position)
                    particles.append(position)
    
    particles = np.array(particles)


def initialise_fcc_right_half(num_particles=48, rho=0.5, aspect_ratio=1.5, visualise=False, checking=False):
    """
    Initialize particles in a 2D system into an FCC-like lattice structure with a given density
    and aspect ratio, positioned in the right half of the simulation box.

    :param num_particles: Number of particles to initialize.
    :param rho: 2D density of the system.
    :param aspect_ratio: Ratio box_size_x / box_size_y. Must be 1.5 for perfect particle numbers.
    :param visualise: Boolean flag to visualize the initial configuration.
    :param checking: Boolean flag to print debug information.
    :return: (particles, sim_box) where 'particles' is an Nx2 array of positions,
             and 'sim_box' is the SimulationBox instance.
    """
    # Calculate the number of unit cells based on aspect ratio and desired particle count
    n = int(np.sqrt(num_particles / 12))  # Since num_particles = 12n^2 for aspect_ratio=1.5
    if n < 1:
        raise ValueError("Number of particles is too small for the given aspect ratio.")
    
    # Derived number of unit cells per side
    num_per_side_x = 3 * n
    num_per_side_y = 2 * n
    
    if checking:
        print(f"Initializing FCC lattice with {num_particles} particles:")
        print(f"  Number of unit cells - X: {num_per_side_x}, Y: {num_per_side_y}")
        print(f"  Aspect Ratio: {aspect_ratio}")
    
    # Compute box dimensions
    area = num_particles / rho
    box_size_x = np.sqrt(area * aspect_ratio)
    box_size_y = np.sqrt(area / aspect_ratio)
    
    sim_box = SimulationBox(box_size_x, box_size_y)

    a_x = 0.15
    a_y = 0.5
    
    # Lattice spacing
    lattice_spacing_x = (box_size_x / num_per_side_x)*a_x
    lattice_spacing_y = (box_size_y / num_per_side_y)*a_y
    
    if checking:
        print(f"Lattice spacing - X: {lattice_spacing_x:.3f}, Y: {lattice_spacing_y:.3f}")
        print(f"Box size - X: {box_size_x:.3f}, Y: {box_size_y:.3f}")
        print(f"Calculated Density: {num_particles / (box_size_x * box_size_y):.3f}")
    
    # Initialize particle positions
    particles = []
    for i in range(num_per_side_x+2):
        for j in range(num_per_side_y):
            if len(particles) < num_particles:
                # Main lattice point
                position = np.array([box_size_x*(1-a_x) - (i * lattice_spacing_x),(box_size_y*(a_y*0.6)) + (j * lattice_spacing_y)])
                position = sim_box.apply_pbc(position)
                particles.append(position)
                
                # FCC-like offset point
                if len(particles) < num_particles:
                    position = np.array([
                        box_size_x*(1-a_x) - ((i + 0.5) * lattice_spacing_x),
                        box_size_y*(a_y*0.6) + ((j + 0.5) * lattice_spacing_y)
                    ])
                    position = sim_box.apply_pbc(position)
                    particles.append(position)
    
    particles = np.array(particles)
    
    # Optional consistency checks
    if checking:
        final_density = num_particles / (box_size_x * box_size_y)
        print(f"Final Density: {final_density:.3f}")
        print(f"Box Dimensions: {box_size_x:.3f} x {box_size_y:.3f}")
    
    # Visualization
    if visualise:
        plt.figure(figsize=(6, 4))
        plt.scatter(particles[:, 0], particles[:, 1], s=20, edgecolors='k')
        plt.title(f"FCC-like Initialization\nDensity: {rho}, Aspect Ratio: {aspect_ratio}")
        plt.xlim(0, box_size_x)
        plt.ylim(0, box_size_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()
    
    return particles, sim_box


