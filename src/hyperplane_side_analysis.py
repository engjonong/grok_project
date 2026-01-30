import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import os
from itertools import islice

def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
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
activation = 'ReLU'
initialization_scale = 8.0
download_directory = "."

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
def analyze_hyperplanes(model, dataset, device, batch_size=50):
    """
    For each linear layer, count how many hyperplanes each example falls on the positive side.
    Returns list of lists: counts_per_layer[layer_idx][sample_idx] = count
    """
    linear_indices = [1, 3, 5]  # indices of Linear layers in the sequential model
    counts_per_layer = [[] for _ in range(len(linear_indices))]
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model[0](x)  # Flatten
            
            for i, lin_idx in enumerate(linear_indices):
                preact = model[lin_idx](out)  # W @ out + b
                count = (preact >= 0).sum(dim=1).cpu().numpy()  # number of positive per sample
                counts_per_layer[i].extend(count)
                
                # Apply activation for next layer
                if lin_idx + 1 < len(model):
                    out = model[lin_idx + 1](preact)
    
    return counts_per_layer

# Checkpoint paths
#checkpoint_dir = Path("../grok_data2/checkpoints")  # Adjust if necessary
checkpoint_dir = Path("../grok_data/checkpoints")  # Adjust if necessary
checkpoints = {
    'first': checkpoint_dir / "mlp_checkpoint_step10000_depth3_width200_scale8.0.pt",
    'middle': checkpoint_dir / "mlp_checkpoint_step100000_depth3_width200_scale8.0.pt",
    'last': checkpoint_dir / "mlp_checkpoint_step190000_depth3_width200_scale8.0.pt"
}

# Run analysis on test data for each checkpoint
results = {}
acc_dict = {}
results_adv = {}
for name, ckpt_path in checkpoints.items():
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
    acc_dict[name] = {'test': test_acc, 'train': train_acc}
    
    # Analyze
    counts_per_layer = analyze_hyperplanes(mlp, test, device)
    results[name] = counts_per_layer
    print(f"Analyzed {name} checkpoint: step {checkpoint['step']}")

# Generate adversarial examples using the first checkpoint
first_ckpt_path = checkpoints['first']
checkpoint = torch.load(first_ckpt_path, map_location=device)
mlp = mlp_template
mlp.load_state_dict(checkpoint['model_state_dict'])
mlp.eval()
adv_examples, adv_labels = generate_fgsm_adversarial_examples(mlp, test, epsilon=0.1, device=device)
adv_test = torch.utils.data.TensorDataset(adv_examples, adv_labels)
print("Generated adversarial test set using first checkpoint")

# Analyze adversarial examples for all checkpoints
for name, ckpt_path in checkpoints.items():
    if not ckpt_path.exists():
        continue
    checkpoint = torch.load(ckpt_path, map_location=device)
    mlp = mlp_template
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.eval()
    counts_adv = analyze_hyperplanes(mlp, adv_test, device)
    results_adv[name] = counts_adv
    adv_acc = compute_accuracy(mlp, adv_test, device)
    acc_dict[name]['adv'] = adv_acc
    print(f"Analyzed adversarial examples for {name} checkpoint")

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
    counts_per_layer = analyze_hyperplanes(mlp, train, device)
    results_train[name] = counts_per_layer
    print(f"Analyzed train {name} checkpoint: step {checkpoint['step']}")

# Plot histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 datasets, 3 checkpoints
layer_names = ['Layer 1', 'Layer 2', 'Layer 3']
colors = ['blue', 'green', 'red']
adv_colors = ['orange', 'purple', 'brown']

datasets = [('Test', results), ('Adversarial', results_adv)]
for row, (data_name, counts_dict) in enumerate(datasets):
    for col, (ckpt_name, counts_per_layer) in enumerate(counts_dict.items()):
        ax = axes[row, col]
        if data_name == 'Test':
            acc = acc_dict[ckpt_name]['test']
            plot_colors = colors
        else:
            acc = acc_dict[ckpt_name]['adv']
            plot_colors = adv_colors
        for i, (layer_name, counts) in enumerate(zip(layer_names, counts_per_layer)):
            base_color = plot_colors[i]
            ax.hist(counts, bins=range(width + 2), alpha=0.5, label=layer_name, density=True, histtype='bar', color=base_color)
            # Darken the color for the step curve
            darker = mcolors.to_rgba(base_color)
            darker = (darker[0] * 0.7, darker[1] * 0.7, darker[2] * 0.7, 1.0)
            ax.hist(counts, bins=range(width + 2), density=True, histtype='step', color=darker, linewidth=2)
        ax.set_title(f'{data_name} - {ckpt_name.capitalize()} Checkpoint (Acc: {acc:.1%})')
        ax.set_xlabel('Number of Positive Hyperplanes')
        ax.set_ylabel('Density')
        ax.set_xlim(0, width)
        ax.legend()
        ax.set_xlabel('Number of Positive Hyperplanes')
        ax.set_ylabel('Density')
        ax.set_xlim(0, width)
        ax.legend()

plt.tight_layout()
plt.savefig('hyperplane_side_histograms.png', dpi=200)
plt.show()
#<parameter name="filePath">/home/engjon/work2/grok_project/src/hyperplane_side_analysis.py