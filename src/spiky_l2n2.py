from collections import defaultdict
from itertools import islice
import random
import time
from pathlib import Path
import math
import argparse
from tools import transformed_nn_ratios, PredDimFromRatio, MakeDimCurves

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

def l2n2_dim_est(points):
    use_slope =  0.904825
    use_intercept = 0.65670
    all_ratios = transformed_nn_ratios( points, [[0,1]] )
      
    # get the predictions from all the curves
    pred_id = PredDimFromRatio( all_ratios[0], use_slope, use_intercept )
    return pred_id


def generate_spiky_data(d, n, e):
    if e > d:
        raise ValueError("e must be <= d")
    selected_axes = random.sample(range(d), e)
    points_per_axis = n // e
    extra = n % e
    points = []
    for axis in selected_axes:
        num_points = points_per_axis + (1 if extra > 0 else 0)
        extra -= 1
        for _ in range(num_points):
            point = np.zeros(d)
            point[axis] = np.random.normal(0, 1)
            points.append(point)
    return np.array(points)


def generate_planes(d, n, e):
    selected_pairs = []
    for i in range(e):
        axis1 = random.randint(0, d-1)
        axis2 = random.randint(0, d-1)
        while axis2 == axis1:
            axis2 = random.randint(0, d-1)
        selected_pairs.append((axis1, axis2))
    points_per_plane = n // e
    extra = n % e
    points = []
    for pair in selected_pairs:
        axis1, axis2 = pair
        num_points = points_per_plane + (1 if extra > 0 else 0)
        extra -= 1
        for _ in range(num_points):
            point = np.zeros(d)
            point[axis1] = np.random.normal(0, 1)
            point[axis2] = np.random.normal(0, 1)
            points.append(point)
    return np.array(points)


def main():
    parser = argparse.ArgumentParser(description="Generate spiky data and estimate intrinsic dimensionality.")
    parser.add_argument('--n', type=int, default=15000, help='Number of points to generate (default: 15000)')
    parser.add_argument('--d', type=int, default=20, help='Total dimensionality (default: 20)')
    parser.add_argument('--e', type=int, default=15, help='Number of dimensions to use (default: 15)')
    args = parser.parse_args()
    
    points_spiky = generate_spiky_data(args.d, args.n, args.e)
    points_planes = generate_planes(args.d, args.n, args.e)
    points = np.vstack([points_spiky, points_planes])
    points = points_planes
    est_dim = l2n2_dim_est(points)
    print(f"L2N2 estimated intrinsic dimensionality: {est_dim}")
    
    if TwoNN is not None:
        two_nn_estimator = TwoNN()
        est_two_nn = two_nn_estimator.fit_transform(points)
        print(f"TwoNN estimated intrinsic dimensionality: {est_two_nn}")
    else:
        print("TwoNN not available, skdim not installed.")


if __name__ == "__main__":
    main()