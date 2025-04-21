import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from scipy.spatial.distance import squareform, cdist
import pandas as pd
import logging
import sys
import os
import json
import argparse
import csv

# importing the NF and MC modules
path = "/home/n2401517d/my_workspace/normalizing-flows-master_template"
sys.path.append(path)

import normflows as NF

def generate_configs(experiment_folder, config, model_files, num_configs, config_gen_batch_size=1000):
    configs_array=[]

    # Extract the required parameters from the configuration dictionary
    num_particles = config.get("NUM_PARTICLES")
    num_dim = config.get("NUM_DIM")
    batch_size = config.get("BATCH_SIZE")
    k = config.get("K")
    half_box = config.get("HALF_BOX")
    
    # Optionally, bundle the extracted parameters into a dictionary for later use
    extracted_params = {
        "NUM_PARTICLES": num_particles,
        "NUM_DIM": num_dim,
        "BATCH_SIZE": batch_size,
        "K": k,
        "HALF_BOX": half_box
    }
    
    # For demonstration, print the extracted parameters
    print("Extracted configuration parameters:")
    for param, value in extracted_params.items():
        print(f"{param}: {value}")

    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print('device is: ' + str(device))

    # Use bracket notation to access the dictionary key
    print("can extract half_box: ", extracted_params["HALF_BOX"])

    # initialise the normalizing flow model
    bound = extracted_params["HALF_BOX"]
    base = NF.Energy.UniformParticle(extracted_params["NUM_PARTICLES"], extracted_params["NUM_DIM"], bound, device=device)
    # target = NF.Energy.SimpleLJ(NUM_PARTICLES * NUM_DIM, NUM_PARTICLES, 1, bound)    # not used for now
    # K = 15 - now defined in the argument of function
    flow_layers = []
    for i in range(extracted_params["K"]):
        flow_layers += [
            NF.flows.CircularCoupledRationalQuadraticSpline(
                extracted_params["NUM_PARTICLES"] * extracted_params["NUM_DIM"],
                8,
                256,
                range(extracted_params["NUM_PARTICLES"] * extracted_params["NUM_DIM"]),
                num_bins=32,
                tail_bound=bound
            )
        ]
    model = NF.NormalizingFlow(base, flow_layers).to(device)
    print(f'Model prepared with {extracted_params["NUM_PARTICLES"]} particles and {extracted_params["NUM_DIM"]} dimensions!')

    for model_path in model_files:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  

        n_iterations = int(num_configs / config_gen_batch_size)
        config_gen_batch_size = 1000

        # List to store samples
        all_samples = []

        # Generate samples in batches
        for i in range(n_iterations):
            z = model.sample(config_gen_batch_size)  # Generate samples
            z_ = z.reshape(-1, extracted_params["NUM_PARTICLES"], extracted_params["NUM_DIM"])  # Reshape as required
            z_ = z_.to('cpu').detach().numpy()  # Move to CPU and convert to NumPy
            z_ = z_ + 5
            all_samples.append(z_)  # Collect the batch

        # Concatenate all batches into a single array
        final_samples = np.concatenate(all_samples, axis=0)
        print(np.shape(final_samples))

        configs_array.append(final_samples)
    
    return configs_array




