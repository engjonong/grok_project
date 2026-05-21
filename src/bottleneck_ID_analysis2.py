import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib
# use a non-GUI backend when running in environments without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import os
from itertools import islice
from tools import transformed_nn_ratios, PredDimFromRatio, MakeDimCurves

# UMAP for weight visualization
import umap

def l2n2_dim_est(points):
    use_slope =  0.904825
    use_intercept = 0.65670
    all_ratios = transformed_nn_ratios( points, [[0,1]] )
      
    # get the predictions from all the curves
    pred_id = PredDimFromRatio( all_ratios[0], use_slope, use_intercept )
    return pred_id

def compute_l2n2_intrinsic_dims(layer_outputs, sample_limit=10000):
    """Compute intrinsic dimensionality for each layer output using l2n2_dim_est.

    Returns a NumPy array of shape (num_layers,) with the estimated intrinsic dimension
    for each layer. Layers that fail estimation will be assigned np.nan.
    """
    dims = []
    for arr in layer_outputs:
        # Convert torch.Tensor inputs to NumPy arrays
        if isinstance(arr, torch.Tensor):
            np_arr = arr.cpu().numpy()
        else:
            np_arr = np.asarray(arr)

        # np_arr is (num_samples, features)
        n_samples = np_arr.shape[0]
        if n_samples < 3:
            dims.append(np.nan)
            continue

        # subsample if dataset is large
        if n_samples > sample_limit:
            idx = np.random.choice(n_samples, sample_limit, replace=False)
            X = np_arr[idx].astype(np.float64)
        else:
            X = np_arr.astype(np.float64)

        try:
            d = l2n2_dim_est(X)
        except Exception:
            d = np.nan
        dims.append(d)
    return np.array(dims)


def compute_layer_ID(model, dataset, device, batch_size=50):
    """Compute intrinsic dimensionality for layers 1,3,5 of `model` on `dataset`.

    Runs the dataset through the network, captures the pre-activation outputs of
    the three linear layers (indices 1,3,5 in the sequential model), and then
    passes those arrays to :func:`compute_l2n2_intrinsic_dims`.  The returned
    NumPy array has one entry per layer.
    """
    # same indices used elsewhere in this file
    linear_indices = [1, 3, 5]
    collected = [[] for _ in linear_indices]

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model[0](x)  # flatten
            for i, lin_idx in enumerate(linear_indices):
                preact = model[lin_idx](out)
                collected[i].append(preact.cpu())
                # forward through activation if present
                if lin_idx + 1 < len(model):
                    out = model[lin_idx + 1](preact)

    # concatenate and convert to numpy
    layer_outputs = [torch.cat(lst, dim=0) for lst in collected]
    # intrinsic dimension computation handles tensors or arrays
    ids = compute_l2n2_intrinsic_dims(layer_outputs)
    return ids

def compute_accuracy(network, dataset, device, N=10000, batch_size=50):
    """Computes accuracy of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()

def generate_fgsm_adversarial_examples(model, dataset, epsilon, device, batch_size=50):
    """
    Generate adversarial examples from the dataset using Fast Gradient Sign Method (FGSM).
    
    Args:
        model: The neural network model.
        dataset: The dataset to generate adversarial examples from (e.g., test dataset).
        epsilon: The perturbation magnitude.
        device: The device to run on (e.g., 'cuda' or 'cpu').
        batch_size: Batch size for processing.
    
    Returns:
        adv_examples: Tensor of adversarial examples.
        labels: Tensor of corresponding labels.
    """
    model.eval()
    adv_examples = []
    labels = []
    
    # Use CrossEntropyLoss for adversarial attack, as it's standard for classification
    loss_fn = nn.CrossEntropyLoss()
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # Enable gradient computation for input
        x.requires_grad = True
        
        # Forward pass
        outputs = model(x)
        loss = loss_fn(outputs, y)
        
        # Zero gradients
        model.zero_grad()
        
        # Backward pass to compute gradients w.r.t. input
        loss.backward()
        
        # Get the gradient
        grad = x.grad.data
        
        # Generate adversarial example
        adv_x = x + epsilon * grad.sign()
        
        # Clamp to valid range (assuming input is in [0, 1])
        adv_x = torch.clamp(adv_x, 0, 1)
        
        adv_examples.append(adv_x.detach().cpu())
        labels.append(y.detach().cpu())
    
    adv_examples = torch.cat(adv_examples, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return adv_examples, labels

# Parameters matching the training script
depth = 3
width = 200
#width = 300
activation = 'ReLU'
initialization_scale = 8.0
#initialization_scale = 1.0   # the standard deviation of the normal distribution used for weight initialization; higher means more unconstrained optimization, lower means more constrained
download_directory = "."

# Toggle visualization of first 16 adversarial examples (4x4 grid)
SHOW_ADV_GRID = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
torch.set_default_dtype(dtype)

# Load datasets
train = torchvision.datasets.MNIST(root=download_directory, train=True, 
    transform=torchvision.transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root=download_directory, train=False, 
    transform=torchvision.transforms.ToTensor(), download=True)
#train = torch.utils.data.Subset(train, range(1000))  # same as in script

# Activation dict
activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}
activation_fn = activation_dict[activation]

# Create model architecture
layers = [nn.Flatten()]
for i in range(depth):
    if i == 0:
        layers.append(nn.Linear(784, width))
        layers.append(activation_fn())
    elif i == depth - 1:
        layers.append(nn.Linear(width, 10))
    else:
        layers.append(nn.Linear(width, width))
        layers.append(activation_fn())
mlp_template = nn.Sequential(*layers).to(device)

# Function to analyze hyperplanes
def ID_bottleneck_upperbound(model, dataset, device, batch_size=50):
    """
    For each linear layer, count how many hyperplanes each example falls on the positive side.
    Returns list of lists: counts_per_layer[layer_idx][sample_idx] = count
    """
    linear_indices = [1, 3, 5]  # indices of Linear layers in the sequential model
    counts_per_layer = [[] for _ in range(len(linear_indices))]
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # array of sets for storing unique activation binary patterns for each layer
    activ_pat_set = [set() for _ in linear_indices]
    num_examples = len(dataset)

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model[0](x)  # Flatten
            
            for i, lin_idx in enumerate(linear_indices):
                preact = model[lin_idx](out)  # W @ out + b
                
                preact_binary = (preact >= 0).cpu().numpy().astype(int)  # binary pattern of activations
                # add preact_binary patterns to the set for this layer
                for pattern in preact_binary:
                    activ_pat_set[i].add(tuple(pattern))  # convert to tuple for hashability

                count = (preact >= 0).sum(dim=1).cpu().numpy()  # number of positive per sample
                counts_per_layer[i].extend(count)
                
                # Apply activation for next layer
                if lin_idx + 1 < len(model):
                    out = model[lin_idx + 1](preact)
    
    num_unique_patterns = [len(s) for s in activ_pat_set]
    #normalise num_unique_patterns by num_examples to get a ratio
    unique_pattern_ratios = [num / num_examples for num in num_unique_patterns]
    print(f"Unique activation pattern ratios per layer: {unique_pattern_ratios}")

    # for each sample, set the count for a layer to be the min of all previous layers (bottleneck)
    for i in range(1, len(counts_per_layer)):
        counts_per_layer[i] = [min(counts_per_layer[i][j], counts_per_layer[i-1][j]) for j in range(len(counts_per_layer[i]))]
    
    # compute the expected counts for each layer
    expected_counts = []
    for i in range(len(counts_per_layer)):
        expected_count = np.mean(counts_per_layer[i])
        expected_counts.append(expected_count)
    
    return counts_per_layer, expected_counts, unique_pattern_ratios

# Checkpoint paths
checkpoint_dir = Path("checkpoints")  # Adjust if necessary
#checkpoint_dir = Path("../grok_data/checkpoints")  # Adjust if necessary
checkpoints = {
    'first': checkpoint_dir / f"mlp_checkpoint_step20000_depth{depth}_width{width}_scale{initialization_scale}.pt",
    'middle': checkpoint_dir / f"mlp_checkpoint_step100000_depth{depth}_width{width}_scale{initialization_scale}.pt",
    'last': checkpoint_dir / f"mlp_checkpoint_step190000_depth{depth}_width{width}_scale{initialization_scale}.pt"
}

#all_steps = range(10000, 190001, 10000)
all_steps = range(5000, 175001, 5000)
use_checkpoints_files = [f"mlp_checkpoint_step{i}_depth{depth}_width{width}_scale{initialization_scale}.pt" for i in all_steps]
# Run analysis on test data for each checkpoint
results = {}
acc_dict = {}
results_adv = {}

all_expected_counts = []
all_intrinsic_dims = []
all_test_accuracies = []
all_steps = []
all_unique_pattern_ratios = []

#for name, ckpt_path in checkpoints.items():
for ckpt_file in use_checkpoints_files:
    ckpt_path = checkpoint_dir / ckpt_file
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found, skipping.")
        continue
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    mlp = mlp_template
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval()
    
    # Get accuracies
    step = checkpoint['step']
    test_log_steps = checkpoint['test_log_steps']
    test_accuracies = checkpoint['test_accuracies']
    idx = test_log_steps.index(step)
    test_acc = test_accuracies[idx]
    train_accuracies = checkpoint['train_accuracies']
    #train_acc = train_accuracies[idx]
    train_acc = []
    #acc_dict[name] = {'test': test_acc, 'train': train_acc}
    
    # Analyze
    counts_per_layer, expected_counts, unique_pattern_ratios = ID_bottleneck_upperbound(mlp, test, device)
    print(f"Expected counts for checkpoint {test_log_steps[idx]}: {expected_counts}")
    
    #results[name] = counts_per_layer #(counts_per_layer, expected_counts)
    print(f"Analyzed checkpoint {test_log_steps[idx]}: step {checkpoint['step']}")

    # compute the intrinsic dimensions of the layers for this checkpoint
    ids = compute_layer_ID(mlp, test, device)

    all_expected_counts.append(expected_counts)
    all_intrinsic_dims.append(ids)
    all_test_accuracies.append(test_acc)
    all_steps.append(step)
    all_unique_pattern_ratios.append(unique_pattern_ratios)

# Generate adversarial examples using the first checkpoint
first_ckpt_path = checkpoints['first']
checkpoint = torch.load(first_ckpt_path, map_location=device)
mlp = mlp_template
mlp.load_state_dict(checkpoint['model_state_dict'])
mlp.eval()
adv_examples, adv_labels = generate_fgsm_adversarial_examples(mlp, test, epsilon=0.1, device=device)
adv_test = torch.utils.data.TensorDataset(adv_examples, adv_labels)
print("Generated adversarial test set using first checkpoint")

all_adv_expected_counts = []
all_adv_intrinsic_dims = []
all_adv_accuracies = []
all_adv_unique_pattern_ratios = []
# Analyze adversarial examples for all checkpoints
#for name, ckpt_path in checkpoints.items():
for ckpt_file in use_checkpoints_files:
    ckpt_path = checkpoint_dir / ckpt_file

    if not ckpt_path.exists():
        continue
    checkpoint = torch.load(ckpt_path, map_location=device)
    mlp = mlp_template
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval()

    if 0:
        # generate adversarial examples for this checkpoint
        adv_examples, adv_labels = generate_fgsm_adversarial_examples(mlp, test, epsilon=0.1, device=device)
        adv_test = torch.utils.data.TensorDataset(adv_examples, adv_labels)

    counts_adv, expected_counts_adv, unique_pattern_ratios_adv = ID_bottleneck_upperbound(mlp, adv_test, device)
    print(f"Expected counts for adversarial examples at {ckpt_file} checkpoint: {expected_counts_adv}")

    #results_adv[name] = counts_adv
    all_adv_expected_counts.append(expected_counts_adv)
    all_adv_intrinsic_dims.append(compute_layer_ID(mlp, adv_test, device))
    all_adv_accuracies.append(compute_accuracy(mlp, adv_test, device))
    all_adv_unique_pattern_ratios.append(unique_pattern_ratios_adv)

    #adv_acc = compute_accuracy(mlp, adv_test, device)
    #acc_dict[name]['adv'] = adv_acc
    print(f"Analyzed adversarial examples for {ckpt_file} checkpoint")

if 0:
    # Run analysis on train data for each checkpoint
    results_train = {}
    for name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            continue
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        mlp = mlp_template
        mlp.load_state_dict(checkpoint['model_state_dict'])
        mlp.eval()
        
        # Analyze
        counts_per_layer, expected_counts = ID_bottleneck_upperbound(mlp, train, device)
        results_train[name] = counts_per_layer
        print(f"Analyzed train {name} checkpoint: step {checkpoint['step']}")

print(all_intrinsic_dims)
# plot curves of expected counts and intrinsic dims vs steps (test + adversarial)
if all_steps:
    try:
        steps_arr = np.array(all_steps)
        exp_arr = np.vstack(all_expected_counts)          # shape (n_ckpts, n_layers)
        id_arr = np.vstack(all_intrinsic_dims)           # shape (n_ckpts, n_layers)
        adv_exp_arr = np.vstack(all_adv_expected_counts) # same shapes assuming same checkpoints
        adv_id_arr = np.vstack(all_adv_intrinsic_dims)

        n_layers = exp_arr.shape[1]  # assumed equal for both
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        # left subplot for test data
        ax = axes[0]
        for layer in range(n_layers):
            ax.plot(steps_arr, exp_arr[:, layer], marker='o', label=f"exp L{layer+1}")
            ax.plot(steps_arr, id_arr[:, layer], marker='x', linestyle='--', label=f"ID L{layer+1}")
        # plot accuracies on secondary axis
        ax2 = ax.twinx()
        ax2.plot(steps_arr, all_test_accuracies, color='black', marker='s', linestyle=':', label='test acc')
        ax2.plot(steps_arr, all_adv_accuracies, color='red', marker='^', linestyle=':', label='adv acc')
        
        ax2.set_ylabel('accuracy', color='black')
        ax2.tick_params(axis='y', colors='black')
        ax.set_xlabel('training step')
        ax.set_ylabel('value')
        ax.set_title('Test data')
        ax.grid(True)
        # combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize='small')
        # right subplot for adversarial data
        ax = axes[1]
        for layer in range(n_layers):
            ax.plot(steps_arr, adv_exp_arr[:, layer], marker='o', label=f"exp L{layer+1}")
            ax.plot(steps_arr, adv_id_arr[:, layer], marker='x', linestyle='--', label=f"ID L{layer+1}")
        ax.set_xlabel('training step')
        ax.set_title('Adversarial data')
        ax.grid(True)
        ax.legend(fontsize='small')

        plt.suptitle('Expected counts and intrinsic dims vs step')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        try:
            plt.savefig('counts_and_ids_vs_steps.png', dpi=200)
        except Exception as e:
            print(f"Warning: could not save step-curves figure: {e}")
        plt.show()
    except Exception as e:
        print(f"Warning: failed to plot step curves: {e}")
    except Exception as e:
        print(f"Warning: failed to plot step curves: {e}")

# Plot unique pattern ratios vs steps
if all_steps and all_unique_pattern_ratios:
    try:
        steps_arr = np.array(all_steps)
        unique_pattern_ratios_arr = np.array(all_unique_pattern_ratios)  # shape (n_ckpts, n_layers)
        unique_pattern_ratios_arr_adv = np.array(all_adv_unique_pattern_ratios)  # same shape

        n_layers = unique_pattern_ratios_arr.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for layer in range(n_layers):
            ax.plot(steps_arr, unique_pattern_ratios_arr[:, layer], marker='o', label=f"Layer {layer+1}")
            ax.plot(steps_arr, unique_pattern_ratios_arr_adv[:, layer], marker='x', linestyle='--', label=f"Layer {layer+1} (adv)")
        ax.set_xlabel('Training step')
        ax.set_ylabel('Unique pattern ratio')
        ax.set_title('Unique pattern ratios vs training step')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
        plt.xscale('log')

        plt.tight_layout()
        try:
            plt.savefig('unique_pattern_ratios_vs_steps.png', dpi=200)
        except Exception as e:
            print(f"Warning: could not save unique pattern ratios figure: {e}")
        plt.show()
    except Exception as e:
        print(f"Warning: failed to plot unique pattern ratios: {e}")

