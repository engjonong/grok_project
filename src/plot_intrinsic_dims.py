#!/usr/bin/env python3
"""Load saved per-layer outputs (.npz) and plot intrinsic dimensionality over training steps.

This script looks for files in the `layer_outputs/` directory with names like
`mlp_test_layer_outputs_step{step}_depth{depth}_width{width}_scale{scale}.npz` and
extracts the saved `intrinsic_dims` (TwoNN) and `l2n2_intrinsic_dims` (L2N2) arrays from each file.

Produces three plots:
  1. TwoNN intrinsic dimensionality per layer over training steps (saved to layer_outputs/)
  2. L2N2 intrinsic dimensionality per layer over training steps (saved to layer_outputs/)
  3. Train/test accuracies over training steps (saved to figures/)
"""
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

_FN_RE = re.compile(r"step(?P<step>\d+).*(depth(?P<depth>\d+))?.*(width(?P<width>\d+))?", re.IGNORECASE)
_DATA_FN_RE = re.compile(r"training_data_(depth(?P<depth>\d+))?(.*width(?P<width>\d+))?(.*scale(?P<scale>[\d.]+))?", re.IGNORECASE)


def main(layer_dir: Path, out_png: Path):
    files = list(layer_dir.glob('mlp_test_layer_outputs_step*.npz'))
    # sort files by numeric step extracted from filename (place files without a step at the end)
    def _extract_step(fpath: Path):
        m = _FN_RE.search(fpath.name)
        if m and m.group('step'):
            return int(m.group('step'))
        return 10**9
    files.sort(key=_extract_step)
    if not files:
        print(f"No files found in {layer_dir} matching pattern 'mlp_test_layer_outputs*.npz'", file=sys.stderr)
        return 2

    # map layer_idx -> list of (step, dim)
    per_layer_data_twonn = {}
    per_layer_data_l2n2 = {}
    depth = None
    width = None

    for f in files:
        print(f.name)

        m = _FN_RE.search(f.name)
        step = None
        if m and m.group('step'):
            step = int(m.group('step'))
        else:
            # final file without step, place at the end using large step index
            step = 10**9
        if m and depth is None and m.group('depth'):
            depth = int(m.group('depth'))
        if m and width is None and m.group('width'):
            width = int(m.group('width'))

        data = np.load(f)
        try:
            print("TwoNN:", data['intrinsic_dims'])
            dims_twonn = np.array(data['intrinsic_dims']).astype(np.float64)

            # store
            for idx, d in enumerate(dims_twonn):
                per_layer_data_twonn.setdefault(idx, []).append((step, float(d)))
        except:
            print(f"  No 'intrinsic_dims' array found in {f}; skipping TwoNN.", file=sys.stderr)

        try:
            print("L2N2:", data['l2n2_intrinsic_dims'])
            dims_l2n2 = np.array(data['l2n2_intrinsic_dims']).astype(np.float64)

            # store
            for idx, d in enumerate(dims_l2n2):
                per_layer_data_l2n2.setdefault(idx, []).append((step, float(d)))
        except:
            print(f"  No 'l2n2_intrinsic_dims' array found in {f}; skipping L2N2.", file=sys.stderr)

    if not per_layer_data_twonn and not per_layer_data_l2n2:
        print("No intrinsic-dim data collected; nothing to plot.", file=sys.stderr)
        return 3

    def plot_intrinsic_dims(per_layer_data, method_name, depth, width, out_png):
        if not per_layer_data:
            return
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap('tab20')
        for layer_idx, values in sorted(per_layer_data.items()):
            values_sorted = sorted(values, key=lambda x: x[0])
            xs = [v[0] for v in values_sorted]
            ys = [v[1] for v in values_sorted]
            # convert step=1e9 (final file) to max step + small offset so it appears to the right
            xs = np.array(xs, dtype=np.float64)
            if np.any(xs >= 1e9):
                finite_max = np.max(xs[xs < 1e9]) if np.any(xs < 1e9) else 0
                xs = np.where(xs >= 1e9, finite_max * 1.02 + 1.0, xs)
            plt.plot(xs, ys, marker='o', label=f'layer {layer_idx}', color=cmap(layer_idx % 20))

        plt.xlabel('Training Step')
        plt.xscale('log')

        plt.ylabel(f'Intrinsic Dimensionality ({method_name})')
        title_parts = []
        if depth is not None:
            title_parts.append(f'depth={depth}')
        if width is not None:
            title_parts.append(f'width={width}')
        title = f'Intrinsic Dimensionality over Training Steps ({method_name})'
        if title_parts:
            title += ' (' + ', '.join(title_parts) + ')'
        plt.title(title)
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.3)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved {method_name} intrinsic-dim plot to {out_png}")

    twonn_out = out_png.parent / (out_png.stem + '_twonn' + out_png.suffix)
    l2n2_out = out_png.parent / (out_png.stem + '_l2n2' + out_png.suffix)
    plot_intrinsic_dims(per_layer_data_twonn, 'TwoNN', depth, width, twonn_out)
    plot_intrinsic_dims(per_layer_data_l2n2, 'L2N2', depth, width, l2n2_out)
    return 0


def plot_training_accuracies(fig_dir: Path, out_png: Path):
    """Load training data from figures/ and plot train/test accuracies.
    
    Looks for files like training_data_depth{depth}_width{width}_scale{scale}.npz
    and plots train/test accuracies over training steps.
    """
    files = list(fig_dir.glob('training_data_*.npz'))
    if not files:
        print(f"No training data files found in {fig_dir} matching pattern 'training_data_*.npz'", file=sys.stderr)
        return 2
    
    # use the first (or only) training data file found
    data_file = files[0]
    print(f"Loading training data from {data_file.name}")
    
    data = np.load(data_file)
    try:
        log_steps = np.array(data['train_log_steps'], dtype=np.float64)
        test_log_steps = np.array(data['test_log_steps'], dtype=np.float64)
        
        train_accuracies = np.array(data['train_accuracies'], dtype=np.float64)
        test_accuracies = np.array(data['test_accuracies'], dtype=np.float64)        
        adv_test_accuracies = np.array(data['adv_test_accuracies'], dtype=np.float64) if 'adv_test_accuracies' in data else None
        norms = np.array(data['norms'], dtype=np.float64)
        
        train_accuracies = train_accuracies[ log_steps >= 1000 ]
        test_accuracies = test_accuracies[ test_log_steps >= 1000 ]
        if adv_test_accuracies is not None:
            adv_test_accuracies = adv_test_accuracies[ test_log_steps >= 1000 ]
                
        norms = norms[ log_steps >= 1000 ]
        
        log_steps = log_steps[ log_steps >= 1000 ]
        test_log_steps = test_log_steps[ test_log_steps >= 1000 ]

    except KeyError as e:
        print(f"Error: missing expected key in training data file: {e}", file=sys.stderr)
        return 3
    
    # extract parameters from filename
    m = _DATA_FN_RE.search(data_file.name)
    depth = int(m.group('depth')) if m and m.group('depth') else None
    width = int(m.group('width')) if m and m.group('width') else None
    scale = float(m.group('scale')[:-1]) if m and m.group('scale') else None
    
    # prepare plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_train = 'tab:red'
    color_test = 'tab:green'
    
    ax1.set_xlabel('Training Step')
    ax1.set_xscale('log')
    ax1.set_ylabel('Accuracy', color='black')
    ax1.plot(log_steps, train_accuracies, color=color_train, label='Train Accuracy', marker='o', markersize=4)
    ax1.plot(test_log_steps, test_accuracies, color=color_test, label='Test Accuracy', marker='s', markersize=4)
    ax1.plot(test_log_steps, adv_test_accuracies, color='tab:orange', label='Adv Test Accuracy', marker='x', markersize=4) if adv_test_accuracies is not None else None
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # add weight norm on secondary y-axis
    ax2 = ax1.twinx()
    color_norm = 'tab:purple'
    ax2.set_ylabel('Weight Norm', color=color_norm)
    ax2.plot(log_steps, norms, color=color_norm, label='Weight Norm', marker='^', markersize=4, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_norm)
    
    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize='small')
    
    # title
    title_parts = []
    if depth is not None:
        title_parts.append(f'depth={depth}')
    if width is not None:
        title_parts.append(f'width={width}')
    if scale is not None:
        title_parts.append(f'scale={scale}')
    title = 'Train/Test Accuracy over Training Steps'
    if title_parts:
        title += ' (' + ', '.join(title_parts) + ')'
    ax1.set_title(title)
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved training accuracy plot to {out_png}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot intrinsic dimensionality per layer over training steps (TwoNN and L2N2)')
    parser.add_argument('--layer-dir', type=Path, default=Path('layer_outputs'), help='Directory containing saved .npz files')
    parser.add_argument('--out', type=Path, default=Path('layer_outputs/intrinsic_dims_over_steps.png'), help='Output PNG path for intrinsic dims plot')
    parser.add_argument('--fig-dir', type=Path, default=Path('figures'), help='Directory containing training data .npz files')
    parser.add_argument('--out-acc', type=Path, default=Path('figures/train_test_accuracy_over_steps.png'), help='Output PNG path for accuracy plot')
    args = parser.parse_args()
    
    # plot intrinsic dims
    ret1 = main(args.layer_dir, args.out)
    
    # plot training accuracies
    ret2 = plot_training_accuracies(args.fig_dir, args.out_acc)
    
    sys.exit(ret1 or ret2)
