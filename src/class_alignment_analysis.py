import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib
# use a non-GUI backend when running in environments without a display
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pathlib import Path
import os
from itertools import islice

# Parameters matching the training script
depth = 3
width = 200
activation = 'ReLU'
initialization_scale = 8.0   # the standard deviation of the normal distribution used for weight initialization; higher means more unconstrained optimization, lower means more constrained
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

checkpoint_dir = Path("./checkpoints")

all_steps = range(10000, 190001, 10000)
#all_stepes = [20000, 100000, 190000 ]
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

use_hid_units = [0,20,30]

weights_over_time = {h: {} for h in use_hid_units}
adv_test_acc_over_time = {}
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
    
    step = checkpoint['step']
    
    # Extract weights for hidden units
    output_layer = mlp[-1]  # Linear(200, 10)
    weights = output_layer.weight.detach().cpu().numpy()  # shape (10, 200)
    
    for h in use_hid_units:
        weights_over_time[h][step] = weights[:, h].copy()
        weights_over_time[h][step] /= np.sum(np.abs(weights_over_time[h][step]))  # normalize for better visualization
        
    # Get accuracies
    test_log_steps = checkpoint['test_log_steps']
    test_accuracies = checkpoint['test_accuracies']
    adv_test_accuracies = checkpoint['adv_test_accuracies']
    idx = test_log_steps.index(step)
    test_acc = test_accuracies[idx]
    
    adv_test_acc = adv_test_accuracies[idx]
    adv_test_acc_over_time[step] = adv_test_acc

    train_accuracies = checkpoint['train_accuracies']
    #train_acc = train_accuracies[idx]
    train_acc = []

# After processing all checkpoints, generate 2D plots (weight value vs training step)
for h in use_hid_units:
    fig, ax = plt.subplots(figsize=(10, 6))
    steps_list = [step for step in sorted(weights_over_time[h].keys()) if step in adv_test_acc_over_time]
    if len(steps_list) == 0:
        print(f'No data for hidden unit {h}, skipping plot.')
        continue
    # choose a color map with at least 10 distinct colors
    cmap = plt.get_cmap('tab10')
    class_lines = []
    class_labels = []
    for class_idx in range(10):
        y = [weights_over_time[h][step][class_idx] for step in steps_list]
        color = cmap(class_idx % 10)
        line, = ax.plot(steps_list, y, label=f'Class {class_idx}', color=color)
        class_lines.append(line)
        class_labels.append(f'Class {class_idx}')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Weight Value')
    ax.set_title(f'Weights from Hidden Unit {h} to Output Classes Over Training')
    ax.grid(True)

    ax2 = ax.twinx()
    adv_values = [adv_test_acc_over_time[step] for step in steps_list]
    adv_line, = ax2.plot(steps_list, adv_values, label='Adv Test Accuracy', color='black', linewidth=2, marker='o')
    ax2.set_ylabel('Adv Test Accuracy')
    ax2.set_ylim(0, 1)

    handles = class_lines + [adv_line]
    labels = class_labels + ['Adv Test Accuracy']
    ax.legend(handles, labels, loc='upper left', ncol=2)

    out_png = f'weights_hidden_unit_{h}_2d_adv_acc.png'
    fig.tight_layout()
    plt.savefig(out_png)
    print(f'Saved 2D plot to {out_png}')
    plt.close(fig)
