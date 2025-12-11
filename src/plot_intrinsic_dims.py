#!/usr/bin/env python3
"""Load saved per-layer outputs (.npz) and plot intrinsic dimensionality over training steps.

This script looks for files in the `layer_outputs/` directory with names like
`mlp_test_layer_outputs_step{step}_depth{depth}_width{width}_scale{scale}.npz` and
extracts the saved `intrinsic_dims` array from each file.

Produces a single plot with one curve per layer and saves it as a PNG in
`layer_outputs/`.
"""
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

_FN_RE = re.compile(r"step(?P<step>\d+).*(depth(?P<depth>\d+))?.*(width(?P<width>\d+))?", re.IGNORECASE)


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
    per_layer_data = {}
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
            print(data['intrinsic_dims'])
            dims = np.array(data['intrinsic_dims']).astype(np.float64)

            # store
            for idx, d in enumerate(dims):
                per_layer_data.setdefault(idx, []).append((step, float(d)))
        except:
            print(f"  No 'intrinsic_dims' array found in {f}; skipping.", file=sys.stderr)

    if not per_layer_data:
        print("No intrinsic-dim data collected; nothing to plot.", file=sys.stderr)
        return 3

    # prepare plot
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab20')
    max_layers = max(per_layer_data.keys()) + 1

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
    plt.ylabel('Intrinsic Dimensionality (TwoNN)')
    title_parts = []
    if depth is not None:
        title_parts.append(f'depth={depth}')
    if width is not None:
        title_parts.append(f'width={width}')
    title = 'Intrinsic Dimensionality over Training Steps'
    if title_parts:
        title += ' (' + ', '.join(title_parts) + ')'
    plt.title(title)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved intrinsic-dim plot to {out_png}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot intrinsic dimensionality per layer over training steps')
    parser.add_argument('--layer-dir', type=Path, default=Path('layer_outputs'), help='Directory containing saved .npz files')
    parser.add_argument('--out', type=Path, default=Path('layer_outputs/intrinsic_dims_over_steps.png'), help='Output PNG path')
    args = parser.parse_args()
    sys.exit(main(args.layer_dir, args.out))
