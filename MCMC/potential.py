import numpy as np

def lennard_jones_energy_virial(r, epsilon=1.0, sigma=1.0, cutoff_constant=2.5, shift=True):
    """Calculate Lennard-Jones potential energy with optional potential shift."""
    r = np.asarray(r)
    r_cut = cutoff_constant
    #r_cut = cutoff_constant * sigma     ---- put back in if I decide to use sigma that is not 1.0
    energy = np.zeros_like(r)
    virial = np.zeros_like(r)
    
    mask = r <= r_cut

    sr6 = (sigma / r[mask])**6
    #print("sr6 = ", sr6)
    sr12 = sr6 * sr6
    #print("sr12 = ", sr12)

    energy[mask] = 4.0 * epsilon * (sr12 - sr6)
    virial[mask] = 48.0 * epsilon * (sr12 - 0.5 * sr6)

    if shift:
        # Shift potential to zero at cutoff
        sr6_cut = (sigma / r_cut)**6
        sr12_cut = sr6_cut * sr6_cut
        energy_cut = 4.0 * epsilon * (sr12_cut - sr6_cut)
        #print("energy_cut = ", energy_cut)
        energy[mask] -= energy_cut

    return energy, virial

def tail_correction_energy_2d(rho, N, r_cut, epsilon=1.0, sigma=1.0):
    """Calculate the tail correction for the potential energy in 2D."""
    U_tail = (8 * np.pi * epsilon * rho * N) * (
        (sigma**12) / (10 * r_cut**10) - (sigma**6) / (4 * r_cut**4)
    )
    return U_tail

def lennard_jones_force(r, epsilon=1.0, sigma=1.0, cutoff_constant=2.5):
    r = np.asarray(r)
    r_cut = cutoff_constant * sigma
    force = np.zeros_like(r)
    mask = (r > 0) & (r <= r_cut)
    sr6 = (sigma / r[mask])**6
    sr12 = sr6 * sr6
    force[mask] = 24.0 * epsilon * (2.0 * sr12 - sr6) / r[mask]
    return force

def tail_correction_pressure_2d(rho, r_cut, epsilon=1.0, sigma=1.0):
    """Calculate the tail correction for the pressure in 2D."""
    P_tail = (24 * np.pi * epsilon * rho**2) * (
        (sigma**12) / (5 * r_cut**10) - (sigma**6) / (4 * r_cut**4)
    )
    return P_tail

def double_well_potential(position, box_size_x, box_size_y, V0_list=None, r0=1.0, k=10.0, num_wells=2):
    """
    Compute a double well potential with variable depths for each well.
    
    Parameters:
        position : array-like
            Array of positions with shape (N, 2) for multiple positions or shape (2,) for a single position.
        box_size_x : float
            The size of the simulation box in the x direction.
        box_size_y : float
            The size of the simulation box in the y direction.
        V0_list : list of floats, optional
            The depths of the potential wells (typically negative). Default is [-4.0, -4.0].
        r0 : float, optional
            The radius of the flat-bottom region for each well. Default is 1.0.
        k : float, optional
            The steepness of the transition from the flat bottom to zero potential. Default is 10.0.
        num_wells : int, optional
            The number of wells; if 1, one well is used, and if 2, two wells are used. Default is 2.
    
    Returns:
        float or np.ndarray
            The computed potential at the given position(s). If a single position is provided, a scalar is returned;
            otherwise, an array of potentials is returned.
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



def double_well_potential_equal(position, box_size_x, box_size_y, V0=-2.0, r0=1.0, k=10.0, num_wells=2):
    """
    Compute a double well potential with a concave (flat-bottomed) shape using a continuous, step-like function.
    
    The potential is modeled such that within a radius r0 of each well center the potential is nearly constant (V0),
    and then it continuously transitions to 0 outside of r0. The hyperbolic tangent is used to provide a smooth
    yet steep step transition. Adjust r0 to increase the width of the bottom of the well and k to control the
    steepness of the transition.
    
    Parameters:
        position : array-like
            Array of positions with shape (N, 2) for multiple positions or shape (2,) for a single position.
        box_size_x : float
            The size of the simulation box in the x direction.
        box_size_y : float
            The size of the simulation box in the y direction.
        V0 : float, optional
            The depth of the potential well (typically negative). Default is -0.5.
        r0 : float, optional
            The radius of the flat-bottom region for each well. Default is 1.0.
        k : float, optional
            The steepness of the transition from the flat bottom to zero potential. Default is 10.0.
        num_wells : int, optional
            The number of wells; if 1, one well is used, and if 2, two wells are used. Default is 2.
    
    Returns:
        float or np.ndarray
            The computed potential at the given position(s). If a single position is provided, a scalar is returned;
            otherwise, an array of potentials is returned.
    """
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

    for center in centers:
        dx = x - center[0]
        dy = y - center[1]

        # Apply minimum image convention for periodic boundary conditions
        dx -= box_size_x * np.round(dx / box_size_x)
        dy -= box_size_y * np.round(dy / box_size_y)

        # Compute the radial distance from the well center
        r = np.sqrt(dx**2 + dy**2)

        # Define a smooth step function:
        # - For r << r0, tanh(k*(r - r0)) ~ -1, so transition ~ 0 and V ≈ V0.
        # - For r >> r0, tanh(k*(r - r0)) ~ 1, so transition ~ 1 and V ≈ 0.
        transition = 0.5 * (1 + np.tanh(k * (r - r0)))

        V += V0 * (1 - transition)

    if V.shape[0] == 1:
        return V[0]  # Return scalar if input was a single position
    return V

def double_well_potential_old(position, box_size_x, box_size_y, V0=-0.5, a=5.0, num_wells=2):
    """
    """

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

    # Initialize potential energy array
    V = np.zeros_like(x, dtype=np.float64)

    for center in centers:
        dx = x - center[0]
        dy = y - center[1]

        # Apply minimum image convention for periodic boundary conditions
        dx -= box_size_x * np.round(dx / box_size_x)
        dy -= box_size_y * np.round(dy / box_size_y)

        r_squared = dx**2 + dy**2

        # Gaussian well
        V += V0 * np.exp(-a * r_squared)

    if V.shape[0] == 1:
        return V[0]  # Return scalar if input was a single position
    return V

