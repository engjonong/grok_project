import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict

def create_model(depth, width, initialization_scale, device='cpu'):
    """Recreate the MLP model based on depth and width."""
    layers = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(784, width))
            layers.append(nn.ReLU())  # Assuming ReLU, adjust if different
        elif i == depth - 1:
            layers.append(nn.Linear(width, 10))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
    mlp = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = initialization_scale * p.data
    return mlp

def compute_mean_pairwise_dot_product(weight_matrix, threshold=0.0):
    """Compute mean and std dev of pairwise dot products after normalizing rows to unit norm.
    
    Only includes dot products where both row norms are above the threshold.
    
    Args:
        weight_matrix: Shape (out_features, in_features)
        threshold: Minimum row norm to include in calculation (default 0.0)
    
    Returns:
        Tuple of (mean_dot, std_dev_dot) - mean and standard deviation of pairwise dot products
    """
    # weight_matrix shape: (out_features, in_features)
    row_norms = torch.norm(weight_matrix, dim=1)
    
    # Find indices of rows with norm above threshold
    valid_indices = torch.where(row_norms > threshold)[0]
    
    if len(valid_indices) < 2:
        # Need at least 2 valid rows to compute pairwise dot products
        return 0.0, 0.0
    
    # Extract valid rows and their norms
    valid_weights = weight_matrix[valid_indices]
    valid_norms = row_norms[valid_indices].unsqueeze(1)
    
    # Normalize valid rows to unit norm
    w_norm = valid_weights / valid_norms
    
    # Compute gram matrix for valid rows
    gram = torch.mm(w_norm, w_norm.t())  # shape: (num_valid, num_valid)
    
    # Extract pairwise dot products (excluding self-products on diagonal)
    num_valid = len(valid_indices)
    dot_products = []
    for i in range(num_valid):
        for j in range(i + 1, num_valid):
            dot_products.append(gram[i, j].item())
    
    dot_products = np.array(dot_products)
    
    # Compute mean and std dev of absolute dot products
    abs_dot_products = np.abs(dot_products)
    mean_dot = np.mean(abs_dot_products)
    std_dev_dot = np.std(abs_dot_products)
    
    return mean_dot, std_dev_dot

def compute_mean_weight_row_norms(weight_matrix):
    """Compute mean of the L2 norms of weight matrix rows."""
    # weight_matrix shape: (out_features, in_features)
    row_norms = torch.norm(weight_matrix, dim=1)  # L2 norm of each row
    mean_row_norm = row_norms.mean().item()
    return mean_row_norm

def load_intrinsic_dims(checkpoint_step, depth, width, scale):
    """Load intrinsic dimensionality data from layer_outputs directory."""
    layer_outputs_dir = Path("layer_outputs")
    pattern = r"mlp_test_layer_outputs_step(\d+)_depth(\d+)_width(\d+)_scale([\d.]+)\.npz"
    
    intrinsic_dims_by_layer = {}
    
    if not layer_outputs_dir.exists():
        return intrinsic_dims_by_layer
    
    # Find the matching file
    for npz_path in layer_outputs_dir.glob("*.npz"):
        match = re.match(pattern, npz_path.name)
        if not match:
            continue
        
        steps = int(match.group(1))
        d = int(match.group(2))
        w = int(match.group(3))
        s = float(match.group(4))
        
        if steps == checkpoint_step and d == depth and w == width and s == scale:
            # Load the npz file
            data = np.load(npz_path)
            for key in data.files:
                if key == "intrinsic_dims":
                    intrinsic_dims = data[key]
                    for i, dim in enumerate(intrinsic_dims):
                        intrinsic_dims_by_layer[f"layer_{i}"] = dim
            break
    
    return intrinsic_dims_by_layer

def main():
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("No checkpoints directory found.")
        return

    # Dictionary to store data: layer_name -> list of (step, mean_dot, std_dot, intrinsic_dim)
    layer_data = defaultdict(list)
    # Dictionary to store mean weight row norms: layer_name -> list of (step, mean_row_norm)
    mean_row_norms_data = defaultdict(list)

    # Regex to parse filename
    pattern = r"mlp_checkpoint_step(\d+)_depth(\d+)_width(\d+)_scale([\d.]+)\.pt"

    for ckpt_path in checkpoint_dir.glob("*.pt"):
        match = re.match(pattern, ckpt_path.name)
        if not match:
            continue

        steps = int(match.group(1))
        depth = int(match.group(2))
        width = int(match.group(3))
        scale = float(match.group(4))

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']

        # Create model
        mlp = create_model(depth, width, scale, device='cpu')
        mlp.load_state_dict(model_state_dict)

        # Load intrinsic dimensions for this checkpoint
        intrinsic_dims_by_layer = load_intrinsic_dims(steps, depth, width, scale)

        # Iterate through layers to find Linear layers
        layer_idx = 0
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data  # shape: (out_features, in_features)
                mean_dot, std_dot = compute_mean_pairwise_dot_product(weight)
                mean_row_norm = compute_mean_weight_row_norms(weight)
                layer_name = f"layer_{layer_idx}"
                intrinsic_dim = intrinsic_dims_by_layer.get(layer_name, None)
                layer_data[layer_name].append((steps, mean_dot, std_dot, intrinsic_dim))
                mean_row_norms_data[layer_name].append((steps, mean_row_norm))
                layer_idx += 1

    # Sort the data by steps for each layer
    for layer_name in layer_data:
        layer_data[layer_name].sort(key=lambda x: x[0])
        mean_row_norms_data[layer_name].sort(key=lambda x: x[0])

    # Plot line plot of mean pairwise dot products with shaded uncertainty bands
    plt.figure(figsize=(10, 6))
    for layer_name, data in layer_data.items():
        if data:
            steps, mean_dots, std_dots, _ = zip(*data)
            steps = np.array(steps)
            mean_dots = np.array(mean_dots)
            std_dots = np.array(std_dots)
            
            # Plot mean line
            plt.plot(steps, mean_dots, label=layer_name, marker='o')
            # Fill between mean - std and mean + std
            plt.fill_between(steps, mean_dots - std_dots, mean_dots + std_dots, alpha=0.2)

    plt.xlabel('Step Number')
    plt.ylabel('Mean Pairwise Dot Product')
    plt.title('Mean Pairwise Dot Products of Normalized Weight Rows per Layer')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('mean_pairwise_dot_products.png')
    plt.show()

    # Plot mean weight row norms
    plt.figure(figsize=(10, 6))
    for layer_name, data in mean_row_norms_data.items():
        if data:
            steps, mean_row_norms = zip(*data)
            plt.plot(steps, mean_row_norms, label=layer_name, marker='o')

    plt.xlabel('Step Number')
    plt.ylabel('Mean Weight Row Norm')
    plt.title('Mean Weight Row Norms per Layer')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('mean_weight_row_norms.png')
    plt.show()

    # Create scatter plots for each layer: intrinsic dimensionality vs mean pairwise dot product
    # Collect valid data for each layer
    valid_layers = []
    layer_plot_data = {}
    
    for layer_name, data in layer_data.items():
        if data:
            steps, mean_dots, std_dots, intrinsic_dims = zip(*data)
            # Filter out None values for intrinsic_dims
            valid_indices = [i for i, dim in enumerate(intrinsic_dims) if dim is not None]
            
            if valid_indices:
                valid_mean_dots = [mean_dots[i] for i in valid_indices]
                valid_intrinsic_dims = [intrinsic_dims[i] for i in valid_indices]
                valid_layers.append(layer_name)
                layer_plot_data[layer_name] = (valid_intrinsic_dims, valid_mean_dots)
    
    # Create grid of subplots
    if valid_layers:
        num_layers = len(valid_layers)
        num_cols = 3
        num_rows = (num_layers + num_cols - 1) // num_cols  # Ceiling division
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        for idx, layer_name in enumerate(valid_layers):
            valid_intrinsic_dims, valid_mean_dots = layer_plot_data[layer_name]
            axes[idx].scatter(valid_intrinsic_dims, valid_mean_dots, s=100, alpha=0.7, edgecolors='k')
            axes[idx].set_xlabel('Intrinsic Dimensionality')
            axes[idx].set_ylabel('Mean Pairwise Dot Product')
            axes[idx].set_title(f'{layer_name}')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(num_layers, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('intrinsic_dim_vs_dotproduct_grid.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    main()#</content>
#<parameter name="filePath">/home/engjon/work2/grok_project/src/analyze_checkpoints.py