import numpy as np
import pandas as pd

import matplotlib.colors as mplcolors
from matplotlib import colormaps as mplcmaps
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import utils.miscellaneous_functions as MF

COOLWARM = mplcmaps["coolwarm"]
REDS = mplcmaps["Reds"]
BLUES = mplcmaps["Blues"]

def calculate_centered_cmap_fractions(vmin,vmax,total_idx=256):
    mid_idx = total_idx//2

    if vmin == vmax:
        raise ValueError("Cannot use the same `vmin` and `vmax`")

    if np.sign(vmin) == np.sign(vmax):
        raise ValueError("`vmin` and `vmax` have the same sign (and neither of them is 0). Use the colormap returned by `choose_cmap()` in plotting_helpers.py instead.")

    if vmin == 0:
        low_idx = mid_idx
        high_idx = total_idx
    elif vmax == 0:
        low_idx = 0
        high_idx = mid_idx
    elif vmin < 0 and vmax > 0:
        if abs(vmin) > vmax:
            low_idx = 0
            high_idx = mid_idx + abs(vmax/vmin)*mid_idx
        else:
            high_idx = total_idx
            low_idx = mid_idx - abs(vmin/vmax)*mid_idx
    
    return low_idx/total_idx, high_idx/total_idx

def get_truncated_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return mplcolors.LinearSegmentedColormap.from_list(
            colors = cmap(np.linspace(minval, maxval, n)),
            name = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval)
        )

def get_centered_cmap_from_vminvmax(vmin,vmax,cmap=COOLWARM):
    """
    An illustration of its usage is shown in `illustrate_centered_cmap_usage()` in mixed_plots.py
    """

    low,high = calculate_centered_cmap_fractions(vmin,vmax,cmap.N)
    
    return get_truncated_colormap(cmap,low,high)

def get_reds_cmap(cmap=COOLWARM):
    return get_truncated_colormap(cmap, minval=0.5,maxval=1.)

def get_blues_cmap(cmap=COOLWARM):
    return get_truncated_colormap(cmap, minval=0., maxval=0.5)

def choose_cmap(vmin,vmax,all_from_divergent_cmap=False,divergent_cmap=COOLWARM):
    if MF.is_negative(vmin*vmax):
        return get_centered_cmap_from_vminvmax(vmin=vmin,vmax=vmax,cmap=divergent_cmap)
    elif vmin >= 0:
        return get_reds_cmap(cmap=divergent_cmap) if all_from_divergent_cmap else REDS
    else:
        return get_blues_cmap(cmap=divergent_cmap) if all_from_divergent_cmap else BLUES
    
def get_cbar_extend(vmin,vmax, real_vmin, real_vmax):
    min_extend = vmin > real_vmin
    max_extend = vmax < real_vmax

    if min_extend and max_extend: return 'both'
    elif min_extend: return 'min'
    elif max_extend: return 'max'
    else: return 'neither'
    
def get_ellipse_coords(radius, ratio=1, tilt=0, centre=[0,0], phirange=[-180,180], n_datapoins=200):
    """
    Build ellipse going around the circle with phi from -180 to 180, calculating the values of x and y.
    The values of x are like in a circle: radius*cos(phi)
    The values of y are ratio*radius*sin(phi), which makes for the ellipse shape
    Then rotate the frame by the tilt.

    Returns
    -------
    A tuple (x,y) where x and y are numpy arrays of length `n_datapoints`.
    """
    
    phi = np.linspace(np.radians(phirange[0]), 
                        np.radians(phirange[1]), n_datapoins)
    
    x_ellipse = radius*np.cos(phi) + centre[0]
    y_ellipse = ratio*radius*np.sin(phi) + centre[1]
    
    xy_ellipse = np.vstack((x_ellipse,y_ellipse))
    
    rot_matrix = get_rot_matrix(tilt)
    
    coords_rot = np.matmul(rot_matrix, xy_ellipse)
                                
    return coords_rot[0], coords_rot[1]

def get_rot_matrix(angle):
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.array([[c, -s],[s, c]])

def rotate_by_angle(x,y,angle):
    original = [x,y]
    return np.dot(get_rot_matrix(-angle),original)

def get_equal_n_bin_edges(val_array, n_bins, verbose=False, pandas_way=False):
    """
    Constructs `n_bins` of at least `len(val_array)//n_bins` datapoints in each bin, and gives the bin edges.
    If there are leftover stars (n_bins-1 at most), i.e. `len(val_array)/n_bins` is not an integer, they will be appended to the rightmost bin.

    The returned array of edges is meant to be used such that, given a bin with certain left and right edges, the stars are binned as [left,right)
    except the rightmost bin which should be [left,right]

    Parameters
    ----------
    val_array: numpy array
        Values over which to construct the binning.
    n_bins: integer
        Number of bins to construct.
    pandas_way: boolean
        Whether to use the pandas qcut() method (instead of my manual one). The result should be the same.
    
    Returns
    -------
    bin_edges: numpy array
        Array containing the bin edges.
    """

    if pandas_way:
        if verbose:
            print("Using the pandas qcut() method.")

        return pd.qcut(val_array,q=n_bins,retbins=True)[1]

    val_array = np.array(val_array)

    total_n = len(val_array)
    sorted_idx = np.argsort(val_array)

    n_stars = total_n//n_bins

    edge_indices = np.array([i*n_stars for i in range(n_bins+1)])
    
    edge_indices[-1] = total_n - 1 # if there are no leftover stars this will equal `total_n` and the next line would give an error

    bin_edges = val_array[sorted_idx[edge_indices]]

    bin_edges[-1] = max(val_array)
        
    if verbose:
        print("total N",total_n)
        print("stars per bin",n_stars)
        print("leftover stars", total_n%n_stars)
        print("edge indices", edge_indices)
        print("bin edges", bin_edges)
        print("min value of array", min(val_array))
        print("max value of array", max(val_array))
    
    return bin_edges

def get_range_medians(array, minima, maxima):
    return np.array([np.median(array[(array>=m)&(array<=M)]) for m,M in zip(minima,maxima)])

def get_range_means(minima,maxima):
    return np.array([np.mean([m,M]) for m,M in zip(minima,maxima)])

def get_xerr(minima, maxima, plot, frac=2):
    """
    Returns array of shape (2,N), with N the number of datapoints, where the values of the first and second row
    are the xerr to the left and right respectively.
    
    The left/right error values cover `1/frac` of the distance between the min/max of the bin and the x-position of the datapoint.
    """
    
    return [(np.array(plot) - np.array(minima))/frac,
            (np.array(maxima) - np.array(plot))/frac]

def add_zero_array_key(dictionary, key="zero"):
    """
    Add a new key to a dictionary of any shape which has all entries set to 0.
    It is useful to plot variables for which you do not want to show any errors.
    """

    first_key = list(dictionary.keys())[0]
    shape = dictionary[first_key].shape
    dictionary[key] = np.zeros(shape=shape)

def shall_plot_zero_line(minima,maxima,max_frac=0.1,verbose=False):
    """
    This function is used in cases where you do not want to manually set the ylims of the plot, so if the zero value is too far from the data you do not
    want to show the zero line because otherwise the limits will adjust to it.

    It returns True if:
        - The zero line is enclosed within the data range
        - The distance from the closest point to the zero line is <10% of the distance between the furthest and closest points (to the zero line)
    """
    
    minima = np.array(minima)
    maxima = np.array(maxima)
    
    if np.any(minima < 0) and np.any(maxima > 0):
        return True
    
    closest_to_zero = min(np.nanmin(np.abs(minima)),np.nanmin(np.abs(maxima)))
    furthest_from_zero = max(np.nanmax(np.abs(minima)),np.nanmax(np.abs(maxima)))
    max_diff = furthest_from_zero - closest_to_zero
    
    if verbose:
        print(closest_to_zero, furthest_from_zero)
        print(max_diff)
        print(closest_to_zero/max_diff)
    
    return closest_to_zero/max_diff < max_frac

def get_equal_n_minmax_b_ranges(df, n_points=3, extra_variable="R",extra_min=0,extra_max=3.5,depth_var="l",depth_min=-2,depth_max=2,overall_bmin=1.5,overall_bmax=9,verbose=False):
    """
    Currently written to define the latitude bins along the minor axis for which we are producing the different kinematics vs age/metallicity plots.

    The pencil-beam clusters are hard-coded in.
    """

    df_extra = df[(df[depth_var]>=depth_min)&(df[depth_var]<=depth_max)\
                    &(df["b"]>=overall_bmin)&(df["b"]<=overall_bmax)\
                    &(df[extra_variable]>=extra_min)&(df[extra_variable]<=extra_max)]
    
    low_max = np.max(df_extra[df_extra["b"]<6.8]["b"])
    n_points_low = n_points-2 if overall_bmax == 13 else n_points-1
    edges_low = get_equal_n_bin_edges(df_extra[df_extra["b"]<=low_max].b.values, n_points_low,verbose=verbose)

    o_b_range_min,o_b_range_max = list(edges_low[:-1]), list(edges_low[1:])

    if overall_bmax >= 9: # first cluster
        high_min = np.min(df_extra[df_extra["b"]>6.8]["b"])
        high_max = np.max(df_extra[df_extra["b"]<9]["b"])

        o_b_range_min += [high_min]
        o_b_range_max += [high_max]

    if overall_bmax >= 13: # second cluster
        higher_min = np.min(df_extra[df_extra["b"]>9]["b"])
        higher_max = np.max(df_extra["b"])

        o_b_range_min += [higher_min]
        o_b_range_max += [higher_max]

    o_b_range_min = np.array(o_b_range_min)
    o_b_range_max = np.array(o_b_range_max)

    return o_b_range_min,o_b_range_max

def get_plot_values_from_hist(h,normalised=False):
    x = h[1]
    y = np.array([h[0][0]]+list(h[0]))
    if normalised:
        #True: histogram values add up to 1
        #False: integral is one
        y *= np.diff(x)[0]
    return (x,y)

def get_norm_from_count_list(count_list,log=True):
    vmin,vmax = float("inf"),float("-inf")

    for c in count_list:
        if log:
            c = c[c!=0]

        vmin = min(vmin, np.nanmin(c))
        vmax = max(vmax, np.nanmax(c))
    
    return LogNorm(vmin=vmin,vmax=vmax) if log else plt.Normalize(vmin=vmin,vmax=vmax)

def cbar_extend_list_union(extend_list):
    if "both" in extend_list or ("min" in extend_list and "max" in extend_list):
        return "both"
    if "min" in extend_list:
        return "min"
    elif "max" in extend_list:
        return "max"
    else:
        return "neither"
    
def get_asymmetric_error_text(value, lower_error, upper_error, dec=2):
    """
    Get a LaTeX string that renders to a value with upper and lower error bounds.
    
    Current method:     r"$1.23^{\,+0.4}_{\,-0.5}$"
    Alternative method: r"$1.23\genfrac{}{}{0}{}{+0.4}{-0.5}$"
    """

    return r"$%s^{\,+%s}_{\,-%s}$"%(MF.return_int_or_dec(value,dec=dec),
                                    MF.return_int_or_dec(upper_error,dec=dec),
                                    MF.return_int_or_dec(lower_error,dec=dec))