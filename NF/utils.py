import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from tqdm import trange
import torch
import numpy as np
import pandas as pd
import json
import os

def get_project_root():
    """
    Get the path to the flow_state project root directory.
    
    Returns:
        str: Path to the flow_state directory, or None if not found
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(current_dir) == "flow_state":
            return current_dir
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # Reached root directory
            return None
        current_dir = parent

def set_icl_color_cycle():
    """
    Sets the Matplotlib color cycle to a custom set of ICL colors
    in an order aimed at being clearer for color-blind users,
    and also updates the rcParams based on the server type.
    
    Custom Color Order:
        1. Imperial Blue    : "#0000CD"
        2. Crimson          : "#DC143C"
        3. Teal             : "#008080"
        4. Orange Red       : "#FF4500"
        5. Yellow           : "#FFFF00"
        6. Medium Violet Red: "#C71585"
        7. Dark Green       : "#006400"
        8. Indigo           : "#4B0082"
        9. Saddle Brown     : "#8B4513"
       10. Navy Blue        : "#000080"
       11. Slate Gray       : "#708090"
       12. Dark (near-black): "#232323"
    """
    # Define the custom color cycle
    icl_cycle = [
        "#0000CD",  # Imperial Blue
        "#DC143C",  # Crimson
        "#008080",  # Teal
        "#FF4500",  # Orange Red
        "#FFFF00",  # Yellow
        "#C71585",  # Medium Violet Red
        "#006400",  # Dark Green
        "#4B0082",  # Indigo
        "#8B4513",  # Saddle Brown
        "#000080",  # Navy Blue
        "#708090",  # Slate Gray
        "#232323",  # Dark (near-black)
    ]
    
    # Update the Matplotlib color cycle
    mpl.rcParams['axes.prop_cycle'] = cycler(color=icl_cycle)
    
    def is_tex_available():
        """Check if TeX is available by actually trying to render something."""
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Save current rcParams
        original_rcParams = mpl.rcParams.copy()
        
        try:
            # Temporarily set usetex to True
            with mpl.rc_context({'text.usetex': True}):
                # Create a small figure and try to render TeX
                fig = plt.figure(figsize=(1, 1))
                canvas = FigureCanvasAgg(fig)
                
                # Try to render a simple TeX expression
                plt.text(0.5, 0.5, r'$\alpha$')
                
                # Force drawing to trigger TeX compilation
                canvas.draw()
                
                # If we got here without error, TeX works
                plt.close(fig)
                return True
        except Exception as e:
            # Any exception means TeX isn't working properly
            plt.close('all')
            print(f"TeX rendering failed: {str(e)}")
            return False
        finally:
            # Restore original settings
            mpl.rcParams.update(original_rcParams)

    # Update matplotlib settings, using TeX only if available
    tex_available = is_tex_available()
    mpl.rcParams.update({
        "text.usetex": tex_available,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman","DejaVu Serif","Times New Roman",  "Bitstream Vera Serif"],
    })  

    # Optional: inform the user of the TeX status
    if tex_available:
        print("TeX rendering is available and enabled.")
    else:
        import warnings
        warnings.warn("TeX not available, falling back to Matplotlib's mathtext.")

    # Set default DPI for figures and savefig
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'svg'

def get_icl_heatmap_cmap(cmap_type="sequential"):
    """
    Returns a Matplotlib colormap suitable for heatmaps based on the ICL palette.
    
    Parameters:
        cmap_type (str): The type of colormap to return. Options are:
            "sequential" - A sequential colormap from Imperial Blue to Yellow.
            "diverging"  - A diverging colormap from Imperial Blue through white to Crimson.
            "multistep"  - A multistep colormap using Imperial Blue, Teal, Orange Red, and Yellow.
    
    Returns:
        A Matplotlib LinearSegmentedColormap.
    
    Examples:
        cmap = get_icl_heatmap_cmap("sequential")
        plt.imshow(data, cmap=cmap)
    """
    if cmap_type == "sequential":
        # Suggestion 1: Sequential colormap from Imperial Blue to Yellow.
        return LinearSegmentedColormap.from_list("ICL_Sequential", ["#000080", "#FFFF00"])
    elif cmap_type == "diverging":
        # Suggestion 2: Diverging colormap from Imperial Blue to white to Crimson.
        return LinearSegmentedColormap.from_list("ICL_Diverging", ["#000080", "#FFFFFF", "#DC143C"])
    elif cmap_type == "multistep":
        # Suggestion 3: Multistep colormap using Imperial Blue, Teal, Orange Red, and Yellow.
        return LinearSegmentedColormap.from_list("ICL_MultiStep", ["#000080", "#008080", "#FF4500", "#FFFF00"])
    else:
        raise ValueError("Invalid cmap_type. Choose from 'sequential', 'diverging', or 'multistep'.")

def classify_particles(positions, halfbox, r0):
    box_size_x = halfbox*2
    box_size_y = halfbox*2

    # Define circle centers and radius
    left_center = [box_size_x / 4, box_size_y / 2]
    right_center = [3 * box_size_x / 4, box_size_y / 2]
    radius = r0 * 1.2  # 10% larger than r0

    # Function to check if point is in circle with periodic boundaries
    def point_in_circle(point, center, radius, box_x, box_y):
        dx = point[0] - center[0]
        dy = point[1] - center[1]

        # Apply minimum image convention
        dx -= box_x * np.round(dx / box_x)
        dy -= box_y * np.round(dy / box_y)

        return (dx ** 2 + dy ** 2) <= radius ** 2

    # Classify each particle
    classifications = []
    for config in positions:
        config_classifications = []
        for particle in config:
            in_left = point_in_circle(particle, left_center, radius, box_size_x, box_size_y)
            in_right = point_in_circle(particle, right_center, radius, box_size_x, box_size_y)

            if in_left:
                config_classifications.append('A')
            elif in_right:
                config_classifications.append('B')
            else:
                config_classifications.append('Outside')
        classifications.append(config_classifications)

    classifications = np.array(classifications)
    return classifications

def plot_loss(loss_epoch=None, directory=None, base_filename="loss_function", data_path=None):
    """
    Plot and save the training loss history in both SVG and PNG formats.
    
    Args:
        loss_epoch (list or numpy.ndarray, optional): Loss values for each epoch
        directory (str, optional): Directory to save the plot
        base_filename (str): Base filename for the saved plots without extension
        data_path (str, optional): Path to saved loss data. If provided, other args are ignored.
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
        loss_epoch = np.array(data['loss_epoch'])
        directory = data.get('directory', './')
        base_filename = data.get('base_filename', 'loss_function')
    else:
        # Save data for future use
        data = {
            'loss_epoch': np.array(loss_epoch).tolist(),
            'directory': directory,
            'base_filename': base_filename
        }
        data_save_path = f'{directory}{base_filename}_data.json'
        with open(data_save_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.array(loss_epoch))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    loss_plot_path_svg = f'{directory}{base_filename}.svg'
    loss_plot_path_png = f'{directory}{base_filename}.png'
    
    fig.savefig(loss_plot_path_svg, bbox_inches='tight')
    fig.savefig(loss_plot_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return loss_plot_path_svg, loss_plot_path_png

def generate_samples(model, n_particles, n_dimension, n_iterations=100, samples_per_iteration=5000):
    """
    Generate samples from the trained model using a batch approach.
    
    Args:
        model (torch.nn.Module): Trained model
        n_particles (int): Number of particles
        n_dimension (int): Number of dimensions
        n_iterations (int): Number of batches to generate
        samples_per_iteration (int): Number of samples per batch
        
    Returns:
        numpy.ndarray: Generated samples
    """
    model.eval()
    all_samples = []

    with torch.no_grad():
        for i in range(n_iterations):
            # Generate samples in smaller batches
            z = model.sample(samples_per_iteration)
            # Reshape as required
            z_ = z.reshape(-1, n_particles, n_dimension)
            # Move to CPU and convert to NumPy
            z_ = z_.to('cpu').detach().numpy()
            all_samples.append(z_)

    final_samples = np.concatenate(all_samples, axis=0)
    return final_samples

def plot_frequency_heatmap(final_samples=None, directory=None, cmap_div=None, bound=None, 
                          base_filename="frequency_heatmap", bins=100, data_path=None):
    """
    Generate and save a frequency heatmap of particle positions in both SVG and PNG formats.
    
    Args:
        final_samples (numpy.ndarray, optional): Samples from the model
        directory (str, optional): Directory to save the plot
        cmap_div (matplotlib.colors.Colormap, optional): Colormap for the heatmap
        bound (float, optional): Boundary value for the plot
        base_filename (str): Base filename for the saved plots without extension
        bins (int): Number of bins for the histogram
        data_path (str, optional): Path to saved heatmap data. If provided, other args are ignored.
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
        hist_relative = np.array(data['hist_relative'])
        x_edges = np.array(data['x_edges'])
        y_edges = np.array(data['y_edges'])
        directory = data.get('directory', './')
        base_filename = data.get('base_filename', 'frequency_heatmap')
        cmap_name = data.get('cmap_name', 'viridis')
        
        # Handle ICL custom colormaps
        if cmap_name.startswith('ICL_'):
            cmap_type = cmap_name.split('_')[1].lower()  # Extract 'diverging', 'sequential', etc.
            cmap_div = get_icl_heatmap_cmap(cmap_type)
        else:
            cmap_div = plt.get_cmap(cmap_name)
    else:
        # Calculate histogram data
        coordinates = final_samples.reshape(-1, 2)
        x_edges = np.linspace(-bound, bound, bins)
        y_edges = np.linspace(-bound, bound, bins)
        
        hist, x_edges, y_edges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], 
                                               bins=[x_edges, y_edges])
        # Normalize the histogram
        hist_relative = hist / hist.sum()
        
        # Save data for future use
        data = {
            'hist_relative': hist_relative.tolist(),
            'x_edges': x_edges.tolist(),
            'y_edges': y_edges.tolist(),
            'directory': directory,
            'base_filename': base_filename,
            'cmap_name': cmap_div.name if hasattr(cmap_div, 'name') else 'viridis'
        }
        data_save_path = f'{directory}{base_filename}_data.json'
        with open(data_save_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    # Plot the heatmap
    fig = plt.figure(figsize=(6, 4.8))
    plt.imshow(hist_relative.T, origin='lower', 
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
              aspect='auto', cmap=cmap_div)
    plt.colorbar(label='Relative Frequency')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    
    heatmap_path_svg = f'{directory}{base_filename}.svg'
    heatmap_path_png = f'{directory}{base_filename}.png'
    
    fig.savefig(heatmap_path_svg, bbox_inches='tight')
    fig.savefig(heatmap_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return heatmap_path_svg, heatmap_path_png

def calculate_pair_correlation(final_samples, n_particles, bound, dr=None):
    """
    Calculate the pair correlation function (RDF) correctly.
    
    Args:
        final_samples (numpy.ndarray): Samples from the model (centered at 0)
        n_particles (int): Number of particles
        bound (float): Half box length (coordinates from -bound to +bound)
        dr (float): Bin width for histogram (default: bound/50)
        
    Returns:
        tuple: (r values, g(r) values)
    """
    if dr is None:
        dr = bound/50
        
    result_BG = []
    
    for i_f in trange(0, len(final_samples), desc="Calculating pair correlation"):
        particle_location = final_samples[i_f]
        
        # FIXED: Don't add an artificial particle at zero
        # This distorts the RDF and messes up the normalization
        
        expanded_tensor = particle_location[:, np.newaxis, :]
        diff = expanded_tensor - expanded_tensor.transpose(1, 0, 2)
        diff = diff - (2*bound) * np.round(diff/(2*bound))
        distance_matrix = np.linalg.norm(diff, axis=-1)
        
        # Flatten and remove self-distances (zeros on diagonal)
        distance_matrix_all = distance_matrix.flatten()
        distance_matrix_all = distance_matrix_all[distance_matrix_all != 0]
        
        N, _ = np.histogram(distance_matrix_all, np.arange(0, bound+dr, dr))
        
        # FIXED: Use correct normalization
        # Each particle has n_particles-1 pairs
        norm = n_particles * (n_particles - 1) / 2
        
        # FIXED: Use correct density calculation
        # For a box from -bound to +bound, area is (2*bound)^2
        rou = n_particles/(4*bound*bound)
        
        i_vals = np.arange(0, bound, dr)
        area = np.pi*((i_vals+dr)**2-i_vals**2)
        result_r = N/(norm*rou*area)
        result_BG.append(result_r)

    result_BG = np.array(result_BG)
    g_r = pd.DataFrame(result_BG).apply(np.mean, axis=0)
    r_vals = np.arange(0, bound, dr)
    
    return r_vals, g_r

def plot_pair_correlation(r_vals=None, g_r=None, directory=None, base_filename="pair_correlation_function", data_path=None):
    """
    Plot and save the pair correlation function in both SVG and PNG formats.
    
    Args:
        r_vals (numpy.ndarray, optional): Distance values
        g_r (pandas.Series or numpy.ndarray, optional): Pair correlation values
        directory (str, optional): Directory to save the plot
        base_filename (str): Base filename for the saved plots without extension
        data_path (str, optional): Path to saved pair correlation data. If provided, other args are ignored.
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
        r_vals = np.array(data['r'])
        g_r = np.array(data['g_r'])
        directory = data.get('directory', './')
        base_filename = data.get('base_filename', 'pair_correlation_function')
    else:
        # Save data for future use
        data = {
            'r': r_vals.tolist(),
            'g_r': g_r.tolist() if isinstance(g_r, np.ndarray) else g_r.values.tolist(),
            'directory': directory,
            'base_filename': base_filename
        }
        data_save_path = f'{directory}{base_filename}_data.json'
        with open(data_save_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    fig = plt.figure(figsize=[8, 5])
    plt.plot(r_vals, g_r)
    plt.xlabel(r'$r$ / $\sigma$')
    plt.ylabel(r'$g(r)$')
    
    pair_corr_path_svg = f'{directory}{base_filename}.svg'
    pair_corr_path_png = f'{directory}{base_filename}.png'
    
    fig.savefig(pair_corr_path_svg, bbox_inches='tight')
    fig.savefig(pair_corr_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return pair_corr_path_svg, pair_corr_path_png

def save_rdf_data(r_vals, g_r, directory, experiment_id):
    """
    Save RDF data as JSON for later combined plotting.
    
    Args:
        r_vals (numpy.ndarray): Distance values
        g_r (pandas.Series or numpy.ndarray): Pair correlation values
        directory (str): Directory to save the data
        experiment_id (str): Experiment identifier
        
    Returns:
        str: Path to the saved JSON file
    """
    rdf_data = {
        "r": r_vals.tolist(),
        "g_r": g_r.tolist() if isinstance(g_r, np.ndarray) else g_r.values.tolist()
    }
    
    rdf_json_path = f'{directory}rdf_data_{experiment_id}.json'
    with open(rdf_json_path, 'w') as f:
        json.dump(rdf_data, f, indent=4)
    
    return rdf_json_path