#!/usr/bin/env python3
"""Utility for inspecting nearest-neighbour ratios in saved layer outputs.

This script loads an .npz file created by the training scripts (e.g.
`mnist_grok_l2n2_adv.py`, `mnist_grok_l2n2_adv_cnn.py`, etc.).  It picks the
second‑last layer by default, computes the ratio of the 2nd/1st neighbour
distances for every point and plots a histogram of those ratios.

By default it points at the last available checkpoint at step 179000; you
can override the path with ``--file`` on the command line.

The script also invokes :func:`tools.transformed_nn_ratios` simply to show
the aggregated statistic that it returns, but the histogram is built from the
raw ratios.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from tools import transformed_nn_ratios



def compute_raw_ratios(points):
    """Return array of second/first NN distance ratios for ``points``.

    The behaviour mirrors the internals of ``transformed_nn_ratios`` so that
    the histogram corresponds to the same set of values that are aggregated
    inside that helper.
    """
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
    nn.fit(points)
    dists, _ = nn.kneighbors(points)
    # drop the self-distance in column 0
    sorted_dists = dists[:, 1:]
    ratios = sorted_dists[:, 1] / sorted_dists[:, 0]
    # replicate the filtering from transformed_nn_ratios
    ratios = ratios[ratios > 1.0]
    ratios = ratios[np.isfinite(ratios)]
    return ratios


def main(npz_path: str):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"layer-output file not found: {npz_path}")

    data = np.load(npz_path)
    print(f"loaded data from {npz_path} with keys: {list(data.keys())}")
    layer_keys = sorted(
        [k for k in data.keys() if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if len(layer_keys) < 2:
        raise ValueError("need at least two layers in the NPZ file")

    # select which layers to use
    if args.layers:
        all_use_layer_keys = [lk.strip() for lk in args.layers.split(",") if lk.strip()]
    else:
        # by default take the last two layers (as before)
        all_use_layer_keys = layer_keys[-2:]
    if not all_use_layer_keys:
        raise ValueError("no layer keys specified")
    print("processing layer keys:", all_use_layer_keys)

    # container for plotting results
    plt.figure(figsize=(6, 4))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    from scipy.stats import gumbel_l

    for idx, target_key in enumerate(all_use_layer_keys):
        if target_key not in data:
            raise KeyError(f"requested layer key '{target_key}' not in file")
        points = data[target_key]
        print(f"using {target_key} with shape {points.shape}")

        agg = transformed_nn_ratios(points, [[0, 1]])
        print(f"transformed_nn_ratios({target_key}) returned", agg)

        ratios = compute_raw_ratios(points)
        ratios = np.log(np.log(ratios))
        print(f"{target_key}: computed {ratios.size} ratio values (after filtering and transform)")

        params = gumbel_l.fit(ratios)
        print(f"{target_key}: fitted gumbel_l parameters", params)

        col = colors[idx % len(colors)]
        plt.hist(ratios,
                 bins=100,
                 density=True,
                 log=True,
                 color=col,
                 edgecolor="none",
                 alpha=0.4,
                 label=f"{target_key} histogram")
        x = np.linspace(ratios.min(), ratios.max(), 1000)
        pdf = gumbel_l.pdf(x, *params)
        plt.plot(x, pdf, color=col, linestyle="-", label=f"{target_key} gumbel_l fit")

    plt.xlabel("log(log(2nd/1st NN distance))")
    plt.ylabel("density (log scale)")
    plt.title("Transformed NN ratio histograms")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot histogram of 2nd/1st nearest-neighbour ratios"
    )
    step_int = 179000
    #step_int = 10000
    file_str_default = "layer_outputs/mlp_test_layer_outputs_step%d_depth3_width200_scale8.0.npz"%step_int
    parser.add_argument(
        "--file",
        type=str,
        default=file_str_default,
        help="path to `.npz` file containing saved layer outputs",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="comma-separated list of layer keys to analyse (e.g. 'layer_1,layer_2');"
             " if omitted the last two layers are used",
    )
    args = parser.parse_args()
    main(args.file)
