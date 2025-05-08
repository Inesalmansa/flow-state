import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
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

def set_icl_color_cycle(server_type=None):
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
        return LinearSegmentedColormap.from_list("ICL_Diverging", ["#0000CD", "#FFFFFF", "#DC143C"])
    elif cmap_type == "multistep":
        # Suggestion 3: Multistep colormap using Imperial Blue, Teal, Orange Red, and Yellow.
        return LinearSegmentedColormap.from_list("ICL_MultiStep", ["#0000CD", "#008080", "#FF4500", "#FFFF00"])
    else:
        raise ValueError("Invalid cmap_type. Choose from 'sequential', 'diverging', or 'multistep'.")