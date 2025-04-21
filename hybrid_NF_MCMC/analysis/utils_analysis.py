import os
import json
import glob
import pandas as pd
import numpy as np

def load_config(experiment_folder):
    """Load simulation configuration from a JSON file."""
    config_path = os.path.join(experiment_folder, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def list_model_files(experiment_folder, extension='pth'):
    """List all model-file paths in the experiment's models folder."""
    models_folder = os.path.join(experiment_folder, 'models')
    return sorted(glob.glob(os.path.join(models_folder, f'*.{extension}')))

def load_sampled_data(experiment_folder):
    """Load the sampled data CSV file."""
    csv_path = os.path.join(experiment_folder, 'data', 'sampled_data.csv')
    return pd.read_csv(csv_path)

def load_production_configurations(experiment_folder):
    """Load production configurations from the NPZ file."""
    npz_path = os.path.join(experiment_folder, 'data', 'production_configurations.npz')
    return np.load(npz_path, allow_pickle=True)['configurations']