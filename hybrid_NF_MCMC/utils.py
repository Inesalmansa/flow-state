import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from tqdm import trange
import torch
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

def setup_logger(logger_name, log_file, file_level=logging.DEBUG, stream_level=logging.WARNING):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(file_level)
    ch = logging.StreamHandler()
    ch.setLevel(stream_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)    
    return logger

def get_dataloader(data, num_particles, dimension, device, batch_size, shuffle=True):
    # Reshape the data to have shape (N, num_particles * dimension) e.g., (510, 6)
    data = data.reshape(data.shape[0], num_particles * dimension)
    print(data.shape)
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32).to(device)
    # Create a TensorDataset
    dataset = TensorDataset(data)
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def calculate_well_statistics(configurations, start_idx, half_box, r0=1.2):
    count_left = 0
    count_right = 0
    avg_x_values = []
    p_a_values = []
    p_b_values = []
    deltaF_normalized_values = []
    runs = []
    
    # Get classifications for all configurations
    classifications = classify_particles(configurations, half_box, r0)
    
    for i, config_class in enumerate(classifications[start_idx:], start=1):
        # Calculate average x position
        avg_x = np.mean(configurations[start_idx + i - 1][:,0])
        avg_x_values.append(avg_x)

        # Check if all particles are in well A or well B
        all_in_A = np.all(config_class == 'A')  # True if all particles are in A
        all_in_B = np.all(config_class == 'B')  # True if all particles are in B
        
        if all_in_A:
            count_left += 1
        elif all_in_B:
            count_right += 1
        
        # Calculate probabilities (cumulative up to this point)
        p_a = count_left / i
        p_b = count_right / i
        p_a_values.append(p_a)
        p_b_values.append(p_b)
        
        # Calculate free energy difference if both probabilities are non-zero
        if p_a > 0 and p_b > 0:
            deltaF = np.log(p_b / p_a)
        else:
            deltaF = 0
        deltaF_normalized_values.append(deltaF)
        runs.append(i)
        
    return avg_x_values, p_a_values, p_b_values, deltaF_normalized_values, runs


def classify_particles(positions, halfbox, r0):
    box_size_x = halfbox*2
    box_size_y = halfbox*2

    # Define circle centers and radius
    left_center = [box_size_x / 4, box_size_y / 2]
    right_center = [3 * box_size_x / 4, box_size_y / 2]
    radius = r0 * 1.1  # 10% larger than r0

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


def plot_state_histogram(classifications=None, directory=None, base_filename="state_histogram", data_path=None):
    """
    Create a histogram plot showing the distribution of system states from particle classifications.
    
    Args:
        classifications (numpy.ndarray, optional): Numpy array of particle classifications ('A', 'B', or 'Outside')
        directory (str, optional): Directory to save the plot
        base_filename (str): Base filename for the saved plots without extension
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved plots (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            state_counts = data['state_counts']
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        # Count configurations in each state
        state_counts = {'All A': 0, '1A2B': 0, '2A1B': 0, 'All B': 0, 'Outside': 0}
        for config in classifications:
            num_A = np.sum(config == 'A') 
            num_B = np.sum(config == 'B')
            num_outside = np.sum(config == 'Outside')
            
            # Classify the configuration
            if num_outside > 0:
                state_counts['Outside'] += 1
            elif num_A == 3:
                state_counts['All A'] += 1
            elif num_B == 3:
                state_counts['All B'] += 1
            elif num_A == 1 and num_B == 2:
                state_counts['1A2B'] += 1
            elif num_A == 2 and num_B == 1:
                state_counts['2A1B'] += 1
        
        # Save data for future use
        data = {
            'state_counts': state_counts,
            'directory': directory
        }
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)

    set_icl_color_cycle()

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['All in A', '1 in A, 2 in B', '2 in A, 1 in B', 'All in B', 'Any Outside']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    total_configs = sum(state_counts.values())
    states = ['All A', '1A2B', '2A1B', 'All B', 'Outside']
    
    for i, state in enumerate(states):
        count = state_counts[state]
        percentage = (count / total_configs) * 100 if total_configs > 0 else 0
        ax.bar(i, percentage, alpha=0.7, label=labels[i], color=colors[i])

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Percentage of Configurations / %')
    ax.set_title('Distribution of System States')
    ax.legend()
    plt.tight_layout()
    
    histogram_path_svg = f'{directory}/{base_filename}.svg'
    histogram_path_png = f'{directory}/{base_filename}.png'
    
    fig.savefig(histogram_path_svg, bbox_inches='tight')
    fig.savefig(histogram_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return histogram_path_svg, histogram_path_png

def calculate_well_statistics_old(configurations, start_idx, half_box):
    count_left = 0
    count_right = 0
    avg_x_values = []
    p_a_values = []
    p_b_values = []
    deltaF_normalized_values = []
    runs = []
    
    for i, config in enumerate(configurations[start_idx:], start=1):
        # Calculate average x position
        avg_x = np.mean(config[:,0])
        avg_x_values.append(avg_x)

        # Count particles in left and right wells
        if avg_x <= half_box:  # Using HALF_BOX = 5.0 from context
            count_left += 1
        else:  # avg_x > 5.0
            count_right += 1
        
        # Calculate probabilities
        p_a = count_left / len(avg_x_values)
        p_b = count_right / len(avg_x_values)
        p_a_values.append(p_a)
        p_b_values.append(p_b)
        
        # Calculate free energy difference if both probabilities are non-zero
        if p_a > 0 and p_b > 0:
            deltaF = np.log(p_b / p_a)
        else:
            deltaF = 0
        deltaF_normalized_values.append(deltaF)
        runs.append(i)
        
    return avg_x_values, p_a_values, p_b_values, deltaF_normalized_values, runs

def set_icl_color_cycle():
    """
    Sets the Matplotlib color cycle to a custom set of ICL colors
    in an order aimed at being clearer for color-blind users,
    and also updates the rcParams to use LaTeX for text rendering
    with the Computer Modern Roman font.
    
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
            "sequential" - A sequential colormap from Navy Blue to Yellow.
            "diverging"  - A diverging colormap from Imperial Blue through white to Crimson.
            "multistep"  - A multistep colormap using Imperial Blue, Teal, Orange Red, and Yellow.
    
    Returns:
        A Matplotlib LinearSegmentedColormap.
    
    Examples:
        cmap = get_icl_heatmap_cmap("sequential")
        plt.imshow(data, cmap=cmap)
    """
    if cmap_type == "sequential":
        # Suggestion 1: Sequential colormap from Navy Blue to Yellow.
        return LinearSegmentedColormap.from_list("ICL_Sequential", ["#000080", "#FFFF00"])
    elif cmap_type == "diverging":
        # Suggestion 2: Diverging colormap from Imperial Blue to white to Crimson.
        return LinearSegmentedColormap.from_list("ICL_Diverging", ["#000080", "#FFFFFF", "#DC143C"])
    elif cmap_type == "multistep":
        # Suggestion 3: Multistep colormap using Imperial Blue, Teal, Orange Red, and Yellow.
        return LinearSegmentedColormap.from_list("ICL_MultiStep", ["#0000CD", "#008080", "#FF4500", "#FFFF00"])
    else:
        raise ValueError("Invalid cmap_type. Choose from 'sequential', 'diverging', or 'multistep'.")
    
def plot_loss(loss_epoch=None, directory=None, base_filename="loss_function", data_path=None):
    """
    Plot and save the training loss history in both SVG and PNG formats.
    
    Args:
        loss_epoch (list or numpy.ndarray, optional): Loss values for each epoch
        directory (str, optional): Directory to save the plot
        base_filename (str): Base filename for the saved plots without extension
        data_path (str, optional): Path to load saved data from instead of using loss_epoch
        
    Returns:
        tuple: Paths to the saved plots (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            loss_epoch = data['loss_epoch']
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        # Save data for future use
        data = {'loss_epoch': loss_epoch if isinstance(loss_epoch, list) else loss_epoch.tolist()}
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.array(loss_epoch))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    loss_plot_path_svg = f'{directory}/{base_filename}.svg'
    loss_plot_path_png = f'{directory}/{base_filename}.png'
    
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
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved heatmaps (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            hist_relative = np.array(data['hist_relative'])
            x_edges = np.array(data['x_edges'])
            y_edges = np.array(data['y_edges'])
            directory = data.get('directory', os.path.dirname(data_path))
            
            # Get colormap from data if not provided
            if cmap_div is None and 'cmap_name' in data:
                cmap_name = data['cmap_name']
                # Handle ICL custom colormaps
                if cmap_name and cmap_name.startswith('ICL_'):
                    cmap_type = cmap_name.split('_')[1].lower()  # Extract 'diverging', 'sequential', etc.
                    cmap_div = get_icl_heatmap_cmap(cmap_type)
                else:
                    cmap_div = plt.get_cmap(cmap_name) if cmap_name else plt.get_cmap('viridis')
    else:
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
            'cmap_name': cmap_div.name if hasattr(cmap_div, 'name') else None
        }
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    # Use default colormap if none is specified
    if cmap_div is None:
        cmap_div = plt.get_cmap('viridis')
        
    # Plot the heatmap
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(hist_relative.T, origin='lower', 
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
              aspect='auto', cmap=cmap_div)
    plt.colorbar(label='Relative Frequency')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    
    heatmap_path_svg = f'{directory}/{base_filename}.svg'
    heatmap_path_png = f'{directory}/{base_filename}.png'
    
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
        
        expanded_tensor = particle_location[:, np.newaxis, :]
        diff = expanded_tensor - expanded_tensor.transpose(1, 0, 2)
        diff = diff - (2*bound) * np.round(diff/(2*bound))
        distance_matrix = np.linalg.norm(diff, axis=-1)
        
        distance_matrix_all = distance_matrix.flatten()
        distance_matrix_all = distance_matrix_all[distance_matrix_all != 0]
        
        N, _ = np.histogram(distance_matrix_all, np.arange(0, bound+dr, dr))
        
        norm = n_particles * (n_particles - 1) / 2
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
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved plots (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            r_vals = np.array(data['r'])
            g_r = np.array(data['g_r'])
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        # Save data for future use
        data = {
            "r": r_vals.tolist(),
            "g_r": g_r.tolist() if isinstance(g_r, np.ndarray) else g_r.values.tolist(),
            "directory": directory
        }
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    fig = plt.figure(figsize=[8, 5])
    plt.plot(r_vals, g_r)
    plt.xlabel(r'$r$')
    plt.ylabel(r'$g(r)$')
    
    pair_corr_path_svg = f'{directory}/{base_filename}.svg'
    pair_corr_path_png = f'{directory}/{base_filename}.png'
    
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

def plot_acceptance_rate(p_acc_history=None, directory=None, x_values=None, xlabel='Cycle', 
                        base_filename="p_acc_history", color=None, data_path=None):
    """
    Plot and save the acceptance rate history in both SVG and PNG formats.
    
    Args:
        p_acc_history (list or numpy.ndarray, optional): Acceptance probability values
        directory (str, optional): Directory to save the plot
        x_values (list or numpy.ndarray, optional): X-axis values (e.g., cycles or total samples)
        xlabel (str): Label for the x-axis
        base_filename (str): Base filename for the saved plots without extension
        color (str, optional): Specific color for the plot line (e.g., 'C2', 'C4')
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved plots (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            p_acc_history = np.array(data['p_acc_history'])
            x_values = np.array(data['x_values']) if 'x_values' in data else None
            xlabel = data.get('xlabel', xlabel)
            if color is None:
                color = data.get('color', color)
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        # Save data for future use
        data = {
            'p_acc_history': p_acc_history if isinstance(p_acc_history, list) else p_acc_history.tolist(),
            'xlabel': xlabel,
            'color': color,
            'directory': directory
        }
        if x_values is not None:
            data['x_values'] = x_values if isinstance(x_values, list) else x_values.tolist()
        
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    fig = plt.figure(figsize=(8, 6))
    
    # Use the specified color if provided, otherwise use default 'blue' = 'C0'
    plot_color = color if color is not None else 'C0'
    
    if x_values is not None:
        plt.plot(x_values, p_acc_history, marker='', linestyle='-', color=plot_color)
    else:
        plt.plot(np.arange(len(p_acc_history)), p_acc_history, marker='', linestyle='-', color=plot_color)
    
    plt.xlabel(xlabel)
    plt.ylabel(r'$P(acc)$')
    plt.ylim(-0.01, 1.01)
    plt.grid(False)
    
    acc_plot_path_svg = f'{directory}/{base_filename}.svg'
    acc_plot_path_png = f'{directory}/{base_filename}.png'
    
    fig.savefig(acc_plot_path_svg, bbox_inches='tight')
    fig.savefig(acc_plot_path_png, bbox_inches='tight', format='png')
    plt.close(fig)
    
    return acc_plot_path_svg, acc_plot_path_png

def plot_avg_free_energy(free_energy_array=None, directory=None, base_filename="avg_free_energy", color=None, data_path=None):
    """
    Plot and save the average free energy with error bands in both SVG and PNG formats.
    
    Args:
        free_energy_array (list or numpy.ndarray, optional): List of free energy arrays from different runs
        directory (str, optional): Directory to save the plot
        base_filename (str): Base filename for the saved plots without extension
        color (str, optional): Specific color for the mean line (e.g., 'C2', 'C4')
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved plots (svg, png), final mean, final sem, and final std values
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            mean_deltaF = np.array(data['mean_deltaF'])
            sem_deltaF = np.array(data['sem_deltaF'])
            final_mean = data['final_mean']
            final_sem = data['final_sem']
            if color is None:
                color = data.get('color', color)
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        # Convert to numpy array if needed
        all_deltaF = np.array(free_energy_array)
        
        # Calculate mean and standard error across runs
        mean_deltaF = np.nanmean(all_deltaF, axis=0)
        sem_deltaF = np.nanstd(all_deltaF, axis=0) / np.sqrt(all_deltaF.shape[0])
        
        # Get final values
        final_mean = float(mean_deltaF[-1])
        final_sem = float(sem_deltaF[-1])
        
        # Save data for future use
        data = {
            'mean_deltaF': mean_deltaF.tolist(),
            'sem_deltaF': sem_deltaF.tolist(),
            'final_mean': final_mean,
            'final_sem': final_sem,
            'color': color,
            'directory': directory
        }
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    # Calculate standard deviation (sem * sqrt(n))
    final_std = final_sem * np.sqrt(all_deltaF.shape[0]) if 'all_deltaF' in locals() else None
    
    # Plot the average free energy with error shading
    steps = np.arange(1, len(mean_deltaF) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use the specified color if provided, otherwise use default 'black' = C11
    plot_color = color if color is not None else 'C11'
    
    ax.plot(steps, mean_deltaF, color=plot_color, lw=2, label='Average $\Delta F$')
    ax.fill_between(steps, mean_deltaF - sem_deltaF, mean_deltaF + sem_deltaF, 
                    color=plot_color, alpha=0.3, label='±1 SEM')
    
    # Add text annotations for final values
    ax.text(0.98, 0.05, f'Final $\Delta F$ = {final_mean:.3f} ± {final_sem:.3f} $k_B T$', 
            transform=ax.transAxes, ha='right', va='top')
    # ax.text(0.98, 0.07, f'SEM = {final_sem:.3f} $k_B T$',
    #         transform=ax.transAxes, ha='right', va='top')

    ax.set_xlabel('Sample Number')
    ax.set_ylabel(r'$\Delta F$ / $k_B T$')
    ax.legend()
    plt.tight_layout()
    
    free_energy_plot_path_svg = f'{directory}/{base_filename}.svg'
    free_energy_plot_path_png = f'{directory}/{base_filename}.png'
    
    fig.savefig(free_energy_plot_path_svg, bbox_inches='tight')
    fig.savefig(free_energy_plot_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return free_energy_plot_path_svg, free_energy_plot_path_png, final_mean, final_sem, final_std

def plot_well_statistics(avg_x_values=None, p_a_values=None, p_b_values=None, deltaF_normalized_values=None, 
                        runs=None, half_box=None, directory=None, base_filename="well_stats", data_path=None):
    """
    Create and save a three-panel plot showing well statistics in both SVG and PNG formats.
    
    Args:
        avg_x_values (list or numpy.ndarray, optional): Average x position values
        p_a_values (list or numpy.ndarray, optional): Probability of being in well A
        p_b_values (list or numpy.ndarray, optional): Probability of being in well B
        deltaF_normalized_values (list or numpy.ndarray, optional): Free energy difference values
        runs (list or numpy.ndarray, optional): Sample numbers or steps
        half_box (float, optional): Half the box size for y-axis scaling
        directory (str, optional): Directory to save the plot
        base_filename (str): Base filename for the saved plots without extension
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved plots (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            avg_x_values = np.array(data['avg_x_values'])
            p_a_values = np.array(data['p_a_values'])
            p_b_values = np.array(data['p_b_values'])
            deltaF_normalized_values = np.array(data['deltaF_normalized_values'])
            runs = np.array(data['runs'])
            half_box = data['half_box']
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        # Save data for future use
        data = {
            'avg_x_values': avg_x_values if isinstance(avg_x_values, list) else avg_x_values.tolist(),
            'p_a_values': p_a_values if isinstance(p_a_values, list) else p_a_values.tolist(),
            'p_b_values': p_b_values if isinstance(p_b_values, list) else p_b_values.tolist(),
            'deltaF_normalized_values': deltaF_normalized_values if isinstance(deltaF_normalized_values, list) else deltaF_normalized_values.tolist(),
            'runs': runs if isinstance(runs, list) else runs.tolist(),
            'half_box': half_box,
            'directory': directory
        }
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    # Plot individual particle positions and average x position
    ax1.plot(runs, avg_x_values, 'C0', alpha=1.0)
    ax1.set_ylabel(r'$\langle x \rangle$')
    ax1.set_ylim(0, half_box * 2)

    # Plot probabilities over time
    ax2.plot(runs, p_a_values, 'C1', label='$p(A)$')
    ax2.plot(runs, p_b_values, 'C6', label='$p(B)$') 
    ax2.set_ylabel('Probability')
    ax2.legend()

    # Plot free energy difference over time
    ax3.plot(runs, deltaF_normalized_values, 'C11')
    ax3.set_xlabel('Sample number')
    ax3.set_ylabel(r'$\Delta F\, / k_B T$')

    # Add text label with final free energy value on the third subplot
    final_deltaF = deltaF_normalized_values[-1]
    ax3.text(0.98, 0.1, f'$\Delta F$ = {final_deltaF:.3f} $k_B T$', 
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax3.transAxes)
  
    ax1.text(0.01, 0.90, '$\mathbf{a.}$', transform=ax1.transAxes, fontsize=12, weight='bold', style='italic')
    ax2.text(0.01, 0.90, '$\mathbf{b.}$', transform=ax2.transAxes, fontsize=12, weight='bold', style='italic')
    ax3.text(0.01, 0.90, '$\mathbf{c.}$', transform=ax3.transAxes, fontsize=12, weight='bold', style='italic')

    plt.tight_layout()

    well_stats_path_svg = f'{directory}/{base_filename}.svg'
    well_stats_path_png = f'{directory}/{base_filename}.png'

    fig.savefig(well_stats_path_svg, bbox_inches='tight')
    fig.savefig(well_stats_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return well_stats_path_svg, well_stats_path_png


def plot_avg_x_coordinate(testing_configs=None, directory=None, half_box=None, run_idx=None, 
                          base_filename="avg_x_configuration", data_path=None):
    """
    Plot and save the average x coordinate over time in both SVG and PNG formats.
    
    Args:
        testing_configs (numpy.ndarray, optional): Configuration data with shape (num_samples, num_particles, dimensions)
        directory (str, optional): Directory to save the plot
        half_box (float, optional): Half the box size for context
        run_idx (int, optional): Run index for the title
        base_filename (str): Base filename for the saved plots without extension
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved plots (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            particle_trajectories = np.array(data['particle_trajectories'])
            avg_x_vals = np.array(data['avg_x_vals'])
            run_idx = data.get('run_idx', run_idx)
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        # Calculate average x position for each configuration
        avg_x_vals = [np.mean(config[:, 0]) for config in testing_configs]
        
        # Extract particle trajectories
        particle_trajectories = []
        for i in range(testing_configs.shape[1]):
            particle_trajectories.append([config[i, 0] for config in testing_configs])
        
        # Save data for future use
        data = {
            'particle_trajectories': particle_trajectories,
            'avg_x_vals': avg_x_vals,
            'run_idx': run_idx,
            'directory': directory
        }
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    steps = np.arange(1, len(avg_x_vals) + 1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot individual particle trajectories with low opacity
    for i, particle_x in enumerate(particle_trajectories):
        ax.plot(steps, particle_x, '-', alpha=0.3, label=f'Particle {i+1}' if i == 0 else None)
    
    # Plot average x position with higher opacity
    ax.plot(steps, avg_x_vals, marker='', linestyle='-', color='blue', alpha=1.0, label='Average')
    
    # Set labels and title
    ax.set_xlabel('Sample Number', fontsize=14)
    ax.set_ylabel(r'$\langle x \rangle$', fontsize=14)
    
    if run_idx is not None:
        ax.set_title(f'MC Run {run_idx}: Average X Coordinate', fontsize=16)
    else:
        ax.set_title('Average X Coordinate vs Sample Number', fontsize=16)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Save the plots
    avg_x_path_svg = f'{directory}/{base_filename}.svg'
    avg_x_path_png = f'{directory}/{base_filename}.png'
    
    fig.savefig(avg_x_path_svg, bbox_inches='tight')
    fig.savefig(avg_x_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return avg_x_path_svg, avg_x_path_png

def plot_multiple_avg_x_coordinates(mc_runs=None, directory=None, num_runs=10, 
                                   base_filename="first10_mc_runs_avg_x", data_path=None):
    """
    Create a multi-panel plot showing average x coordinates for multiple MC runs.
    
    Args:
        mc_runs (list, optional): List of MC run objects, each with testing_samples attribute
        directory (str, optional): Directory to save the plot
        num_runs (int): Number of runs to include in the plot (default: 10)
        base_filename (str): Base filename for the saved plots without extension
        data_path (str, optional): Path to load saved data from
        
    Returns:
        tuple: Paths to the saved plots (svg, png)
    """
    if data_path is not None:
        # Load data from file
        with open(data_path, 'r') as f:
            data = json.load(f)
            avg_x_by_run = data['avg_x_by_run']
            num_runs = len(avg_x_by_run)
            directory = data.get('directory', os.path.dirname(data_path))
    else:
        num_runs = min(num_runs, len(mc_runs))
        avg_x_by_run = []
        
        for j in range(num_runs):
            testing_configs = np.array(mc_runs[j].testing_samples) if hasattr(mc_runs[j], 'testing_samples') and mc_runs[j].testing_samples else np.array([])
            
            if testing_configs.size > 0:
                avg_x = [np.mean(config[:, 0]) for config in testing_configs]
                avg_x_by_run.append(avg_x)
            else:
                avg_x_by_run.append([])
        
        # Save data for future use
        data = {
            'avg_x_by_run': avg_x_by_run,
            'directory': directory
        }
        data_path = f'{directory}/{base_filename}_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f)
    
    rows = (num_runs + 1) // 2  # Calculate rows needed
    
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
    axes = axes.flatten() if num_runs > 2 else [axes] if num_runs == 1 else axes
    
    for j in range(num_runs):
        ax = axes[j]
        avg_x = avg_x_by_run[j]
        
        if avg_x:
            steps = np.arange(1, len(avg_x) + 1)
            ax.plot(steps, avg_x, marker='', linestyle='-', color='blue')
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
        ax.set_title(f'MC Run {j+1:03d}', fontsize=12)
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel(r'$\langle x \rangle$', fontsize=10)
        ax.set_ylim(0,10)
    
    # Hide any unused subplots
    for j in range(num_runs, len(axes)):
        axes[j].set_visible(False)
    
    fig.tight_layout()
    
    summary_plot_path_svg = f'{directory}/{base_filename}.svg'
    summary_plot_path_png = f'{directory}/{base_filename}.png'
    
    fig.savefig(summary_plot_path_svg, bbox_inches='tight')
    fig.savefig(summary_plot_path_png, bbox_inches='tight')
    plt.close(fig)
    
    return summary_plot_path_svg, summary_plot_path_png
