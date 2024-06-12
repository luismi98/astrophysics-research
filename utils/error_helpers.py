import copy
import numpy as np
import warnings
import pandas as pd

import src.compute_variables as CV

def get_function_parameters(func_name, vx, vy, bin_range=None):
    """
    Return the function and parameters needed to apply the given function in order to appropriately compute the given statistic.
    
    Parameters
    ----------
    func_name: string.
        The statistic of interest.
    vx,vy: 1D numpy arrays
        The horizontal/vertical velocities.
    bin_range: list of 2 tuples, optional. Default is None.
        The limits of the bins in position space. Example: [[-1,1], [3,5]]. Required if working with spherical tilt.
        
    Returns
    -------
    function: callable.
        The function to compute the statistic of interest.
    vx: 1D numpy array, or None.
        The horizontal velocity values. If the statistic only involves y, such as mean_vy or std_vy, vx will be None.
    vy: 1D numpy array, or None.
        The vertical velocity values. If the statistic only involves x, such as mean_vx or std_vx, vy will be None.
    R_hat: tuple, or None
        Coordinates of center of bin in position space, used to calculate the spherical tilt.
    tilt: boolean
        Whether the function is a vertex deviation.
    absolute: boolean
        If `tilt` is True, whether the vertex deviation is computed using the absolute value of the dispersion difference.
        Otherwise False.
    """
    
    if "spherical" in func_name and bin_range is None:
        raise ValueError("The bin range is needed to compute the spherical tilt.")
    
    if func_name == "mean_vx":
        return CV.calculate_mean, vx, None, None, False, False
    
    elif func_name == "mean_vy":
        return CV.calculate_mean, None, vy, None, False, False
    
    elif func_name == "std_vx":
        return CV.calculate_std, vx, None, None, False, False
    
    elif func_name == "std_vy":
        return CV.calculate_std, None, vy, None, False, False
    
    elif func_name == "anisotropy":
        return CV.calculate_anisotropy, vx, vy, None, False, False
    
    elif func_name == "correlation":
        return CV.calculate_correlation, vx, vy, None, False, False
    
    elif func_name == "tilt_abs":
        return CV.calculate_tilt, vx, vy, None, True, True
    
    elif func_name == "tilt":
        return CV.calculate_tilt, vx, vy, None, True, False
    
    elif func_name == "spherical_tilt":
        return CV.calculate_spherical_tilt, vx, vy, [np.mean(bin_range[0]), np.mean(bin_range[1])], True, False
    
    elif func_name == "abs_spherical_tilt":
        return CV.calculate_spherical_tilt, vx, vy, [np.mean(bin_range[0]), np.mean(bin_range[1])], True, True
    
    else:
        raise ValueError(f"Behaviour not yet implemented for function name `{func_name}`.")

def apply_function(function, vx, vy, R_hat, tilt, absolute):
    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")
    
    if type(vx) == pd.core.series.Series:
        vx = vx.values
    if type(vy) == pd.core.series.Series:
        vy = vy.values

    if R_hat is None:
        if tilt:
            return function(vx,vy,absolute)
        else:
            return function(vx,vy) if vx is not None and vy is not None else function(vx if vx is not None else vy)
    else:
        return function(vx,vy,R_hat,absolute=absolute)
    
def correct_tilt_branch(values, central_value, inplace=True):
    if not inplace:
        values = copy.deepcopy(values)

    values[(central_value - values)>90] += 180
    values[(central_value - values)<-90] -= 180

    return values if not inplace else None

def build_confidence_interval(values, central_value, symmetric=False):
    if symmetric:
        std = np.sqrt(np.nanmean((values-central_value)**2))
        CI_low,CI_high = std,std
    else:
        CI_low, CI_high = compute_lowhigh_std(central_value=central_value, values=values)

    return CI_low, CI_high

def compute_lowhigh_std(central_value, values):
    values = np.array(values)

    values_above = values[values > central_value]
    values_below = values[values < central_value]
    values_equal = values[values == central_value]

    # divide values equal to the central value proportionally between the above and below splits
    frac_equal_to_above = len(values_above) / (len(values_above) + len(values_below))
    idx_equal_to_above = int( len(values_equal) * frac_equal_to_above )

    values_above = np.append(values_above, values_equal[:idx_equal_to_above])
    values_below = np.append(values_below, values_equal[idx_equal_to_above:])
    
    if 0 < len(values_above) < 0.01*len(values):
        warnings.warn(f"Less than 1% of the values, namely {len(values_above)}, are above the central value. The corresponding std could be misleading as a result.")
    if 0 < len(values_below) < 0.01*len(values):
        warnings.warn(f"Less than 1% of the values, namely {len(values_below)}, are below the central value. The corresponding std could be misleading as a result.")

    std_low = np.sqrt(np.nanmean((values_below - central_value)**2)) if len(values_below) > 0 else 0
    std_high = np.sqrt(np.nanmean((values_above - central_value)**2)) if len(values_above) > 0 else 0

    return std_low, std_high