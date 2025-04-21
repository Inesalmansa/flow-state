import time
import argparse
import torch
import numpy as np
import sys
import json
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import squareform, cdist
import pandas as pd
import os
from utils import (get_project_root, set_icl_color_cycle, get_icl_heatmap_cmap, 
                  plot_loss, generate_samples, plot_frequency_heatmap, 
                  calculate_pair_correlation, plot_pair_correlation, save_rdf_data)

set_icl_color_cycle()
cmap_div = get_icl_heatmap_cmap("diverging")

# Add the custom path for normflows
project_root = get_project_root()
nf_path = os.path.join(project_root, "NF")
sys.path.append(nf_path)
import normflows as nf

def main(experiment_id, data_path, batch_size, epochs, K, n_particles, n_dimension, lr, half_box, num_samples, n_blocks, hidden_units, num_bins, weight_decay):
    start_time = time.time()  # Record the start time for execution timing.
    print('Imports done!')

    project_root = get_project_root()
    directory = os.path.join(project_root, "results", experiment_id)
    if not os.path.exists(directory):
         os.makedirs(directory)

    # Set GPU if possible
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f'device is: {device}')

    # IMPORT DATA
    npz_data = np.load(data_path)
    df = npz_data[npz_data.files[0]]
    # Get unique samples
    unique_data = np.unique(df, axis=0)
    available_samples = unique_data.shape[0]
    print("\nTotal unique samples available:", available_samples)
    
    # Use the entire dataset if num_samples is not provided
    if num_samples is None:
         num_samples = available_samples
         print(f"No num_samples specified. Using all available samples: {num_samples}")
    elif num_samples > available_samples:
         raise ValueError(f"Error: Requested number of samples ({num_samples}) exceeds available unique samples ({available_samples}).")
    
    # Randomly select the desired number of samples
    indices = np.random.choice(available_samples, num_samples, replace=False)
    unique_df = unique_data[indices]
    unique_df = unique_df.reshape(num_samples, n_particles * n_dimension)
    print("Flattened unique_df shape:", unique_df.shape)
    print('Data Loaded!')

    def get_dataloader(data, batch_size, shuffle=True):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32).to(device)
        # Create a TensorDataset
        dataset = TensorDataset(data)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    dataloader = get_dataloader(unique_df, batch_size=batch_size)
    print('Data prepared!')

    '''
    Model preparation
    '''
    bound = half_box
    base = nf.Energy.UniformParticle(n_particles, n_dimension, bound, device=device)
    target = nf.Energy.SimpleLJ(n_particles * n_dimension, n_particles, 1, bound)
    flow_layers = []
    for i in range(K):
        flow_layers += [nf.flows.CircularCoupledRationalQuadraticSpline(n_particles * n_dimension, n_blocks, hidden_units, 
                      range(n_particles * n_dimension), num_bins=num_bins, tail_bound=bound)]
    model = nf.NormalizingFlow(base, flow_layers).to(device)
    print(f'Model prepared with {n_particles} particles and {n_dimension} dimensions!')
    setup = time.time() - start_time 
    print(f"Time taken for setup: {setup:.2f} seconds")

    '''
    Training
    '''
    loss_hist = np.array([])
    loss_epoch = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f'Optimizer prepared with learning rate: {lr} and weight decay: {weight_decay}')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iteration)
    data_save = []

    for it in trange(epochs, desc="Training Progress"):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            # Compute loss
            # loss,z = model.reverse_kld(num_samples)
            loss = model.forward_kld(batch[0].to(device))
            # Do backprop and optimizer step
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                # scheduler.step()
            # Log loss
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            epoch_loss += loss.item()
        loss_epoch.append(epoch_loss / len(dataloader))
        print(f'Epoch {it+1}/{epochs}, Loss: {loss_epoch[-1]:.4f}')

    # Plot loss
    loss_plot_path_svg, loss_plot_path_png = plot_loss(loss_epoch, directory)
    print(f"Loss plot saved successfully to: {loss_plot_path_png}")
    
    model_path = os.path.join(directory, 'LJ_T1_P3_circularspline_res_dense.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to: {model_path}")

    '''
    Test samples preparation!
    '''
    final_samples = generate_samples(model, n_particles, n_dimension, n_iterations=100, samples_per_iteration=5000)
    print(f'Test samples are prepared here: {np.shape(final_samples)}')

    '''
    Frequency samples
    '''
    heatmap_path_svg, heatmap_path_png = plot_frequency_heatmap(final_samples, directory, cmap_div, bound)
    print(f"Frequency heatmap saved successfully to: {heatmap_path_png}")

    '''
    Pair correlation function (RDF)
    '''
    r_vals, g_r = calculate_pair_correlation(final_samples, n_particles, bound)
    pair_corr_path_svg, pair_corr_path_png = plot_pair_correlation(r_vals, g_r, directory)
    print(f"Pair correlation function plot saved successfully to: {pair_corr_path_png}")

    # Save RDF data as JSON for later combined plotting
    rdf_json_path = save_rdf_data(r_vals, g_r, directory, experiment_id)
    print(f"RDF data saved successfully to: {rdf_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalizing Flow on GPU')
    parser.add_argument('--experiment_id', type=str, required=True, help='Experiment ID for saving results')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data', 
                        default=os.path.join(get_project_root(), 'data/rho_0.10_T_0.400_AR_2.000.npz'))
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--K', type=int, default=15, help='number of layers')
    parser.add_argument('--n_particles', type=int, default=2, help='Number of particles')
    parser.add_argument('--n_dimension', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--half_box', type=float, default=8.0, help='Half width of the sim box (aka to set the bound)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to randomly select from the dataset. If not provided, the entire dataset will be used.')
    parser.add_argument('--n_blocks', type=int, default=8, help='Number of blocks in spline flows')
    parser.add_argument('--hidden_units', type=int, default=256, help='Number of hidden units in spline flows')
    parser.add_argument('--num_bins', type=int, default=32, help='Number of bins in spline flows')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay regularization parameter')
    args = parser.parse_args()
    main(args.experiment_id, args.data_path, args.batch_size, args.epochs, args.K, args.n_particles,
         args.n_dimension, args.lr, args.half_box, args.num_samples, args.n_blocks, args.hidden_units, args.num_bins, args.weight_decay)
