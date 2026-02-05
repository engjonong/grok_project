import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
from scipy import stats

def PredDimFromRatio( locscale_ratio, slope, intercept ):
  pred_val = np.exp( ( locscale_ratio - intercept)/slope )

  return pred_val

def transformed_nn_ratios( points, all_nn_pairs, max_neighbors = 20 ):

    # Fit NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=max_neighbors + 1, algorithm='auto')
    nn.fit(points)
    
    # Compute distances to the nearest neighbors
    dists, _ = nn.kneighbors(points)
    
    # Exclude self-distance (which is always zero for the first neighbor)
    sorted_dists = dists[:, 1:]
    
    ret_results = []
    
    for p in all_nn_pairs:
      ratios = sorted_dists[ :, p[1] ] / sorted_dists[ :, p[0] ]
      ratios = ratios[ ratios > 1.0 ]
      ratios = ratios[ np.isfinite( ratios ) ]
      transformed = np.log(np.log(ratios))
              
      cur_res = np.mean( transformed )
      ret_results.append( -cur_res )
      
    return ret_results

def transformed_nn_ratios_ss1( points, all_nn_pairs, subset_size, max_neighbors = 20 ):
    n_points = len(points)
    if subset_size > n_points:
        raise ValueError(f"subset_size ({subset_size}) cannot exceed number of points ({n_points}).")

    # get random subset of points
    rng = np.random.default_rng(None)
    permuted_indices = rng.permutation(n_points)
    subset_idx = permuted_indices[:subset_size]
    subset_points = points[subset_idx]
    
    # Fit NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=max_neighbors + 1, algorithm='auto')
    nn.fit(points)
    
    # Compute distances from subset points to their nearest neighbors in full dataset
    dists, indices = nn.kneighbors(subset_points)

    # Exclude self-distance (which is always zero for the first neighbor)
    sorted_dists = dists[:, 1:]
    
    ret_results = []
    
    for p in all_nn_pairs:
      ratios = sorted_dists[ :, p[1] ] / sorted_dists[ :, p[0] ]
      transformed = np.log(np.log(ratios))
              
      cur_res = np.mean( transformed )
      ret_results.append( -cur_res )
      
    return ret_results
    
# load and produce the dimensionality inference curves
######################################################
# NOTE - we expect the last 2 numbers in the filename to be the index of the nearest neighbours to use.
def MakeDimCurves( all_fnames ):
  all_curves = []
  all_nn_pairs = []
  for curve_fname in all_fnames:
    # get the nearest neighbour indices
    s = curve_fname.split('.')
    
    cur_nn1_ind = int(s[-4])
    cur_nn2_ind = int(s[-3])
    
    dim_ls_data = np.loadtxt( curve_fname, delimiter=',')
      
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log( dim_ls_data[:,0]),-dim_ls_data[:,1] )
    all_curves.append( [slope,intercept] )
    print("Slope = %f Intercept = %f"%(slope, intercept) )
    all_nn_pairs.append( [cur_nn1_ind, cur_nn2_ind] )
    
  return all_curves, all_nn_pairs
