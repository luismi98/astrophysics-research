import copy
import numpy as np
import warnings

def apply_function(function, vx, vy, R_hat, tilt, absolute):
    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")

    if R_hat is None:
        if tilt:
            return function(vx,vy,absolute)
        else:
            return function(vx,vy) if vx is not None and vy is not None else function(vx if vx is not None else vy)
    else:
        return function(vx,vy,R_hat,absolute=absolute)
    
def correct_tilt_branch(values, true_value, inplace=True):
    if not inplace:
        values = copy.deepcopy(values)

    values[(true_value - values)>90] += 180
    values[(true_value - values)<-90] -= 180

    return values if not inplace else None

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