from collections import defaultdict
from itertools import islice
import random
import time
from pathlib import Path
import math

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

# Try to import TwoNN from skdim. Support both `skdim.TwoNN` and `skdim.id.TwoNN` locations.
try:
    from skdim import TwoNN
except Exception:
    try:
        from skdim.id import TwoNN
    except Exception:
        TwoNN = None

import matplotlib.pyplot as plt

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

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

def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points


import torch

def compute_layer_outputs(model, dataset, device, batch_size=50):
#def get_layer_outputs_and_labels(model, test_loader, device):
    """
    Returns:
        outputs_per_layer: list of length L where each element is a tensor
                           containing the output of layer i for all test samples.
        all_labels: tensor containing the labels of all test samples (in order).
    
    """
    model.eval()
    layers = list(model)
    L = len(layers)

    collected = [[] for _ in range(L)]
    labels_list = []

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            # Handle (x, y) or x-only datasets
            if isinstance(batch, (list, tuple)):
                x, y = batch
                labels_list.append(y.detach().cpu())
            else:
                x = batch
                y = None  # if no labels available
            
            x = x.to(device)

            out = x
            for i, layer in enumerate(layers):
                out = layer(out)
                collected[i].append(out.detach().cpu())

    # Concatenate all stored outputs and labels
    outputs_per_layer = [torch.cat(collected[i], dim=0) for i in range(L)]
    all_labels = torch.cat(labels_list, dim=0) if labels_list else None

    return outputs_per_layer, all_labels

def compute_intrinsic_dims(layer_outputs, sample_limit=10000):
    """Compute intrinsic dimensionality for each layer output using skdim's TwoNN.

    Returns a NumPy array of shape (num_layers,) with the estimated intrinsic dimension
    for each layer. If `TwoNN` is not available, raises ImportError. Layers that fail
    estimation will be assigned np.nan.
    """
    if TwoNN is None:
        raise ImportError("skdim TwoNN not found. Install 'skdim' to compute intrinsic dims.")

    dims = []
    for arr in layer_outputs:
        # Convert torch.Tensor inputs to NumPy arrays (TwoNN expects NumPy input)
        if isinstance(arr, torch.Tensor):
            np_arr = arr.cpu().numpy()
        else:
            np_arr = np.asarray(arr)

        # np_arr is (num_samples, features)
        n_samples = np_arr.shape[0]
        if n_samples < 3:
            dims.append(np.nan)
            continue

        # subsample if dataset is large to keep TwoNN fast
        if n_samples > sample_limit:
            idx = np.random.choice(n_samples, sample_limit, replace=False)
            X = np_arr[idx].astype(np.float64)
        else:
            X = np_arr.astype(np.float64)

        try:
            estimator = TwoNN()
            estimator.fit(X)
            # estimator.dimension_ is typically a scalar
            d = float(getattr(estimator, 'dimension_', np.nan))
        except Exception:
            d = np.nan
        dims.append(d)
    return np.array(dims)

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


optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}


train_points = 1000
optimization_steps = 200000
#optimization_steps = 10000 # for quick test to check code
batch_size = 200
loss_function = 'MSE'   # 'MSE' or 'CrossEntropy'
optimizer = 'AdamW'     # 'AdamW' or 'Adam' or 'SGD'
weight_decay = 0.01
lr = 1e-3
initialization_scale = 8.0
download_directory = "."

depth = 3               # the number of nn.Linear modules in the model
width = 200
activation = 'ReLU'     # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

log_freq = math.ceil(optimization_steps / 150)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
seed = 0


torch.set_default_dtype(dtype)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

# load dataset
train = torchvision.datasets.MNIST(root=download_directory, train=True, 
    transform=torchvision.transforms.ToTensor(), download=True)
test = torchvision.datasets.MNIST(root=download_directory, train=False, 
    transform=torchvision.transforms.ToTensor(), download=True)
train = torch.utils.data.Subset(train, range(train_points))
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)


assert activation in activation_dict, f"Unsupported activation function: {activation}"
activation_fn = activation_dict[activation]

# create model
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
mlp = nn.Sequential(*layers).to(device)
with torch.no_grad():
    for p in mlp.parameters():
        p.data = initialization_scale * p.data

# Generate adversarial examples from the test dataset
adv_examples, adv_labels = generate_fgsm_adversarial_examples(mlp, test, epsilon=0.1, device=device)
adv_test = torch.utils.data.TensorDataset(adv_examples, adv_labels)

# create optimizer
assert optimizer in optimizer_dict, f"Unsupported optimizer choice: {optimizer}"
optimizer = optimizer_dict[optimizer](mlp.parameters(), lr=lr, weight_decay=weight_decay)

# define loss function
assert loss_function in loss_function_dict
loss_fn = loss_function_dict[loss_function]()


train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

adv_test_accuracies = []
adv_test_losses = []

norms = []
last_layer_norms = []
train_log_steps = []
test_log_steps = []

print("Training now")
#optimization_steps = 100
steps = 0
one_hots = torch.eye(10, 10).to(device)
with tqdm(total=optimization_steps) as pbar:
    for x, labels in islice(cycle(train_loader), optimization_steps):
        if (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0:
            train_losses.append(compute_loss(mlp, train, loss_function, device, N=len(train)))
            train_accuracies.append(compute_accuracy(mlp, train, device, N=len(train)))
            train_log_steps.append(steps)
            with torch.no_grad():
                total = sum(torch.pow(p, 2).sum() for p in mlp.parameters())
                norms.append(float(np.sqrt(total.item())))
                last_layer = sum(torch.pow(p, 2).sum() for p in mlp[-1].parameters())
                last_layer_norms.append(float(np.sqrt(last_layer.item())))
            pbar.set_description("L: {0:1.1e}. A: {1:2.1f}%".format(
                train_losses[-1],
                train_accuracies[-1] * 100))

        # collect per-layer outputs every 100 steps and save
        if steps % 1000 == 0 and steps > 0:
            try:
                # obtain the layer outputs on clean test data
                layer_outputs, test_labels_for_save = compute_layer_outputs(mlp, test, device, batch_size=batch_size)
                test_losses.append(compute_loss(mlp, test, loss_function, device, N=len(test)))
                test_accuracies.append(compute_accuracy(mlp, test, device, N=len(test)))
                
                # obtain the layer outputs on adversarial test data (FGSM, epsilon=0.1)
                adv_layer_outputs, adv_test_labels_for_save = compute_layer_outputs(mlp, adv_test, device, batch_size=batch_size)
                adv_test_accuracies.append(compute_accuracy(mlp, adv_test, device, N=len(adv_test)))
                adv_test_losses.append(compute_loss(mlp, adv_test, loss_function, device, N=len(adv_test)))

                test_log_steps.append(steps)
                #print(layer_outputs)
                if layer_outputs:
                    # compute intrinsic dims using TwoNN (if available)
                    try:
                        intrinsic_dims = compute_intrinsic_dims(layer_outputs, sample_limit=2000)
                    except ImportError as ie:
                        intrinsic_dims = None
                        print(f"Warning: TwoNN not available, skipping intrinsic dim computation: {ie}")
                    
                    #print( intrinsic_dims )

                    out_dir = Path("layer_outputs")
                    out_dir.mkdir(exist_ok=True)
                    save_path = out_dir / f"mlp_test_layer_outputs_step{steps}_depth{depth}_width{width}_scale{initialization_scale}.npz"
                    # Convert torch.Tensor objects to NumPy arrays for robust saving
                    save_dict = {}
                    for i, arr in enumerate(layer_outputs):
                        if isinstance(arr, torch.Tensor):
                            save_dict[f"layer_{i}"] = arr.cpu().numpy()
                        else:
                            save_dict[f"layer_{i}"] = np.asarray(arr)
                    if isinstance(test_labels_for_save, torch.Tensor):
                        save_dict["labels"] = test_labels_for_save.cpu().numpy()
                    else:
                        save_dict["labels"] = np.asarray(test_labels_for_save)
                    if intrinsic_dims is not None:
                        save_dict["intrinsic_dims"] = np.asarray(intrinsic_dims)
                    np.savez_compressed(save_path, **save_dict)
                    # small, non-fatal message printed to stdout so user sees the save event
                    #print(f"[Saved layer outputs] step={steps} -> {save_path}")
                    # Save a training checkpoint (model + optimizer + metadata)
                    try:
                        checkpoint_dir = Path("checkpoints")
                        checkpoint_dir.mkdir(exist_ok=True)
                        ckpt_path = checkpoint_dir / f"mlp_checkpoint_step{steps}_depth{depth}_width{width}_scale{initialization_scale}.pt"
                        torch.save({
                            'step': steps,
                            'model_state_dict': mlp.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_losses': train_losses,
                            'test_losses': test_losses,
                            'adv_test_losses': adv_test_losses,
                            'train_accuracies': train_accuracies,
                            'test_accuracies': test_accuracies,
                            'adv_test_accuracies': adv_test_accuracies,
                            'train_log_steps': train_log_steps,
                            'test_log_steps': test_log_steps,
                        }, ckpt_path)
                        #print(f"[Saved checkpoint] step={steps} -> {ckpt_path}")
                    except Exception as e_ckpt:
                        print(f"Warning: failed saving checkpoint at step {steps}: {e_ckpt}")
            except Exception as e:
                # avoid training interruption if something goes wrong (I/O, device mismatch, etc.)
                print(f"Warning: failed saving layer outputs at step {steps}: {e}")


        optimizer.zero_grad()
        y = mlp(x.to(device))
        if loss_function == 'CrossEntropy':
            loss = loss_fn(y, labels.to(device))
        elif loss_function == 'MSE':
            loss = loss_fn(y, one_hots[labels])
        loss.backward()
        optimizer.step()
        steps += 1
        pbar.update(1)


print("Plotting")

# compute and save per-layer outputs for test data
print("Collecting layer outputs for test data")
layer_outputs, test_labels = compute_layer_outputs(mlp, test, device, batch_size=batch_size)
if layer_outputs:
    out_dir = Path("layer_outputs")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / f"mlp_test_layer_outputs_depth{depth}_width{width}_scale{initialization_scale}.npz"
    # create a dict for saving: layer_0, layer_1, ... and labels
    # compute intrinsic dims if possible and include them in save
    try:
        intrinsic_dims = compute_intrinsic_dims(layer_outputs, sample_limit=2000)
    except ImportError:
        intrinsic_dims = None
        print("Warning: skdim TwoNN not available; intrinsic dims not computed for final save.")

    save_dict = {}
    for i, arr in enumerate(layer_outputs):
        if isinstance(arr, torch.Tensor):
            save_dict[f"layer_{i}"] = arr.cpu().numpy()
        else:
            save_dict[f"layer_{i}"] = np.asarray(arr)
    if isinstance(test_labels, torch.Tensor):
        save_dict["labels"] = test_labels.cpu().numpy()
    else:
        save_dict["labels"] = np.asarray(test_labels)
    if intrinsic_dims is not None:
        save_dict["intrinsic_dims"] = np.asarray(intrinsic_dims)
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved per-layer test outputs to: {save_path}")
    # Save final training data used for the plot (log steps, accuracies, norms)
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    data_path = fig_dir / f"training_data_depth{depth}_width{width}_scale{initialization_scale}.npz"
    try:
        np.savez_compressed(
            data_path,
            test_log_steps=np.asarray(test_log_steps, dtype=np.float64),
            train_accuracies=np.asarray(train_accuracies, dtype=np.float64),
            test_accuracies=np.asarray(test_accuracies, dtype=np.float64),
            norms=np.asarray(norms, dtype=np.float64),
        )
        print(f"Saved training data to: {data_path}")
    except Exception as e:
        print(f"Warning: failed saving training data: {e}")
ax = plt.subplot(1, 1, 1)
plt.plot(train_log_steps, train_accuracies, color='red', label='train')
plt.plot(test_log_steps, test_accuracies, color='green', label='test')
plt.plot(test_log_steps, adv_test_accuracies, color='blue', label='adv test (FGSM ε=0.1)')
plt.xscale('log')
plt.xlim(10, None)
plt.xlabel("Optimization Steps")
plt.ylabel("Accuracy")
plt.legend(loc=(0.015, 0.75))

ax2 = ax.twinx()
ax2.set_ylabel("Weight Norm", color='purple')
ax2.plot(train_log_steps, norms, color='purple', label='weight norm')
# ax2.set_ylim(27, 63)
plt.legend(loc=(0.015, 0.65))
plt.title(f"depth-3 width-200 ReLU MLP on MNIST\nUnconstrained Optimization α = {initialization_scale}", fontsize=11)
# plt.tight_layout()
# save plot to PNG
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)
fig_path = fig_dir / f"training_curve_depth{depth}_width{width}_scale{initialization_scale}.png"
fig = plt.gcf()
try:
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved training plot to: {fig_path}")
except Exception as e:
    print(f"Warning: failed saving training plot: {e}")

plt.show()
