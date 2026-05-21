#!/usr/bin/env python3
"""Load saved per-layer outputs (.npz) and plot intrinsic dimensionality over training steps.

This script looks for files in the `layer_outputs/` directory with names like
`mlp_test_layer_outputs_step{step}_depth{depth}_width{width}_scale{scale}.npz` and
extracts the saved `intrinsic_dims` (TwoNN), `l2n2_intrinsic_dims` (L2N2),
`adv_intrinsic_dims`, and `adv_l2n2_intrinsic_dims` arrays from each file.

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


def main(layer_dir: Path, out_png: Path, fig_dir: Path = Path('figures')):
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
    per_layer_data_adv_twonn = {}
    per_layer_data_adv_l2n2 = {}
    adv_test_acc_over_time = {}
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
            print("Adv TwoNN:", data['adv_intrinsic_dims'])
            dims_adv_twonn = np.array(data['adv_intrinsic_dims']).astype(np.float64)

            # store
            for idx, d in enumerate(dims_adv_twonn):
                per_layer_data_adv_twonn.setdefault(idx, []).append((step, float(d)))
        except:
            print(f"  No 'adv_intrinsic_dims' array found in {f}; skipping Adv TwoNN.", file=sys.stderr)

        try:
            print("L2N2:", data['l2n2_intrinsic_dims'])
            dims_l2n2 = np.array(data['l2n2_intrinsic_dims']).astype(np.float64)

            # store
            for idx, d in enumerate(dims_l2n2):
                per_layer_data_l2n2.setdefault(idx, []).append((step, float(d)))
        except:
            print(f"  No 'l2n2_intrinsic_dims' array found in {f}; skipping L2N2.", file=sys.stderr)

        try:
            print("Adv L2N2:", data['adv_l2n2_intrinsic_dims'])
            dims_adv_l2n2 = np.array(data['adv_l2n2_intrinsic_dims']).astype(np.float64)

            # store
            for idx, d in enumerate(dims_adv_l2n2):
                per_layer_data_adv_l2n2.setdefault(idx, []).append((step, float(d)))
        except:
            print(f"  No 'adv_l2n2_intrinsic_dims' array found in {f}; skipping Adv L2N2.", file=sys.stderr)

        if 'adv_test_accuracies' in data:
            try:
                adv_acc = np.array(data['adv_test_accuracies'], dtype=np.float64)
                if adv_acc.size == 1:
                    adv_test_acc_over_time[step] = float(adv_acc)
                elif 'test_log_steps' in data:
                    test_steps = np.array(data['test_log_steps'], dtype=np.float64)
                    match = np.where(test_steps == step)[0]
                    if match.size > 0:
                        adv_test_acc_over_time[step] = float(adv_acc[match[0]])
                    else:
                        adv_test_acc_over_time[step] = float(adv_acc[-1])
                else:
                    adv_test_acc_over_time[step] = float(adv_acc[-1])
            except Exception as e:
                print(f"  Failed to read adv_test_accuracies from {f}: {e}", file=sys.stderr)

    if not per_layer_data_twonn and not per_layer_data_l2n2:
        print("No intrinsic-dim data collected; nothing to plot.", file=sys.stderr)
        return 3

    def plot_intrinsic_dims(per_layer_data_test, per_layer_data_adv, method_name, depth, width, out_png, adv_test_acc_over_time=None):
        if not per_layer_data_test and not per_layer_data_adv:
            return
        fig, ax1 = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap('tab20')
        legend_handles = []
        legend_labels = []

        for layer_idx, values in sorted(per_layer_data_test.items()):
            values_sorted = sorted(values, key=lambda x: x[0])
            xs = [v[0] for v in values_sorted]
            ys = [v[1] for v in values_sorted]
            xs = np.array(xs, dtype=np.float64)
            if np.any(xs >= 1e9):
                finite_max = np.max(xs[xs < 1e9]) if np.any(xs < 1e9) else 0
                xs = np.where(xs >= 1e9, finite_max * 1.02 + 1.0, xs)
            line, = ax1.plot(xs, ys, marker='o', label=f'layer {layer_idx} (test)', color=cmap(layer_idx % 20))
            legend_handles.append(line)
            legend_labels.append(f'layer {layer_idx} (test)')

        #for layer_idx, values in sorted(per_layer_data_adv.items()):
        if 0: # skip plotting results for adversarial data for now since it's often very noisy and hard to interpret
            values_sorted = sorted(values, key=lambda x: x[0])
            xs = [v[0] for v in values_sorted]
            ys = [v[1] for v in values_sorted]
            xs = np.array(xs, dtype=np.float64)
            if np.any(xs >= 1e9):
                finite_max = np.max(xs[xs < 1e9]) if np.any(xs < 1e9) else 0
                xs = np.where(xs >= 1e9, finite_max * 1.02 + 1.0, xs)
            line, = ax1.plot(xs, ys, marker='x', linestyle='--', label=f'layer {layer_idx} (adv)', color=cmap(layer_idx % 20))
            legend_handles.append(line)
            legend_labels.append(f'layer {layer_idx} (adv)')

        ax1.set_xlabel('Training Step')
        ax1.set_xscale('log')
        ax1.set_ylabel(f'Intrinsic Dimensionality ({method_name})')
        ax1.grid(True, linestyle='--', alpha=0.3)

        ax2 = None
        if adv_test_acc_over_time:
            x_steps = set()
            for values in list(per_layer_data_test.values()) + list(per_layer_data_adv.values()):
                x_steps.update(v[0] for v in values)
            adv_steps = sorted(step for step in adv_test_acc_over_time.keys() if step in x_steps)
            if adv_steps:
                adv_values = [adv_test_acc_over_time[step] for step in adv_steps]
                ax2 = ax1.twinx()
                color_adv = 'tab:orange'
                adv_line, = ax2.plot(adv_steps, adv_values, color=color_adv, marker='x', linestyle='-', linewidth=2, label='Adv Test Accuracy')
                ax2.set_ylabel('Adv Test Accuracy', color=color_adv)
                ax2.tick_params(axis='y', labelcolor=color_adv)
                ax2.set_ylim(0.0, 1.0)
                legend_handles.append(adv_line)
                legend_labels.append('Adv Test Accuracy')

        title_parts = []
        if depth is not None:
            title_parts.append(f'depth={depth}')
        if width is not None:
            title_parts.append(f'width={width}')
        title = f'Intrinsic Dimensionality over Training Steps ({method_name})'
        if title_parts:
            title += ' (' + ', '.join(title_parts) + ')'
        ax1.set_title(title)

        ax1.legend(legend_handles, legend_labels, loc='best', fontsize='small')

        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Saved {method_name} intrinsic-dim plot to {out_png}")

    def load_adv_test_acc_from_figures(fig_dir: Path):
        out = {}
        files = list(fig_dir.glob('training_data_*.npz'))
        if not files:
            return out
        data = np.load(files[0])
        if 'adv_test_accuracies' not in data or 'test_log_steps' not in data:
            return out
        steps = np.array(data['test_log_steps'], dtype=np.float64)
        acc = np.array(data['adv_test_accuracies'], dtype=np.float64)
        for step, value in zip(steps, acc):
            key = int(step) if np.isfinite(step) and float(step).is_integer() else float(step)
            out[key] = float(value)
        return out

    # try to read adv test accuracy from training data files if available
    adv_test_accs_from_figures = load_adv_test_acc_from_figures(fig_dir)
    if adv_test_accs_from_figures:
        adv_test_acc_over_time = adv_test_accs_from_figures

    twonn_out = out_png.parent / (out_png.stem + '_twonn' + out_png.suffix)
    l2n2_out = out_png.parent / (out_png.stem + '_l2n2' + out_png.suffix)
    plot_intrinsic_dims(per_layer_data_twonn, per_layer_data_adv_twonn, 'TwoNN', depth, width, twonn_out, adv_test_acc_over_time=adv_test_acc_over_time)
    plot_intrinsic_dims(per_layer_data_l2n2, per_layer_data_adv_l2n2, 'L2N2', depth, width, l2n2_out, adv_test_acc_over_time=adv_test_acc_over_time)
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
        
        train_accuracies = train_accuracies[ log_steps >= 100 ]
        test_accuracies = test_accuracies[ test_log_steps >= 100 ]
        if adv_test_accuracies is not None:
            adv_test_accuracies = adv_test_accuracies[ test_log_steps >= 100 ]
                
        norms = norms[ log_steps >= 100 ]
        
        log_steps = log_steps[ log_steps >= 100 ]
        test_log_steps = test_log_steps[ test_log_steps >= 100 ]

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
    
    # plot intrinsic dims (and optionally include adv test accuracy from training data)
    ret1 = main(args.layer_dir, args.out, args.fig_dir)
    
    # plot training accuracies
    ret2 = plot_training_accuracies(args.fig_dir, args.out_acc)
    
    sys.exit(ret1 or ret2)
