import numpy as np

import utils.ellipse_functions as EF
import plotting.mixed_plots as MP

# Note in all the code in this file vx & vy are placeholders for any desired velocity components.

def calculate_covariance(vx,vy):
    """
    Calculate pair-wise covariance over the last axis of the vx and vy arrays.

    Examples
    --------
    Uni-dimensional velocity components

    >>> vx,vy = [1,2,3],[1,2,3]
    >>> calculate_covariance(vx,vy)
    0.6666666666666666

    Several pairs of samples (5), with each sample of size 10.

    >>> vx,vy = np.random.normal(size=(2,5,10))
    >>> calculate_covariance(vx,vy)
    array([ 0.26500959,  0.2597704 ,  0.00100958, -0.17017883,  0.09466613])
    >>> calculate_covariance(vx,vy).shape
    (5,)
    """

    mean_x = calculate_mean(vx, keepdims=True)
    mean_y = calculate_mean(vy, keepdims=True)

    cov_xy = np.mean((vx - mean_x) * (vy - mean_y), axis=-1)
    return cov_xy

def calculate_mean(vx, keepdims = False):
    return np.mean(vx, axis=-1, keepdims=keepdims)

def calculate_var(vx, keepdims = False):
    return np.var(vx, axis=-1, keepdims=keepdims)

def calculate_std(vx, keepdims = False):
    return np.std(vx, axis=-1, keepdims=keepdims)

def calculate_correlation(vx,vy):
    return calculate_covariance(vx,vy) / ( calculate_std(vx) * calculate_std(vy) )

def calculate_anisotropy(vx,vy):
    return 1 - calculate_var(vy) / calculate_var(vx)

# TILTS ------------------------------------------------------------------------------------

def calculate_tilt_from_moments(varx, vary, covxy, absolute=True):
    var_diff = np.abs(varx-vary) if absolute else varx-vary
    tilt = np.degrees(np.arctan2(2.*covxy, var_diff)/2.)
    return tilt

def calculate_tilt(vx, vy, absolute=True):
    """
    I use the name "tilt" as synonym for "vertex deviation". If absolute=False, the tilt is the angle the semi-major axis of the velocity ellipse makes with vx, the
    horizontal velocity axis direction, (eg vr, the line-of-sight direction, if working with vrvl velocities), and its range is (-90,90] deg. 
    
    If absolute=True, the function takes the absolute value of the dispersion difference before computing the tilt, and the range is reduced to [-45,45] deg.
    In this case, the tilt is the angle that the semi-major axis of the velocity ellipse makes with the coordinate axis that it is closest to.
    
    The sign of the tilt is the same independent of `absolute`, and always equal to the sign of the correlation. If positive, the semi-major axis falls
    within the 1st (& 3rd) quadrant of the vx-vy plane. If negative, it falls on the 4th (& 2nd) quadrant.
    """
    return calculate_tilt_from_moments( calculate_var(vx), calculate_var(vy), calculate_covariance(vx,vy), absolute=absolute )

def calculate_spherical_tilt(vx,vy,R_hat,absolute=False):
    """
    The "spherical tilt" is a special case that measures the angle the semi-major axis of the velocity ellipse makes with what I call 
    "R_hat". I call R_hat the direction in position space that joins the GC with the "centre" of the selected group of stars (e.g. the 
    centre of the bin if constructing a kinematic map) out of which you have constructed the velocity ellipse. The spherical tilt 
    quantifies how much the direction of highest dispersion deviates from the direction towards the GC (in the projected 2D view, not 
    taking into account any 3D depth info). I call it "R_hat" because in the xy view the direction is equal to the Galactocentric cylindrical R hat.
    This is not true in other spatial representations (e.g. lb, where R_hat = (l,b) = l*e_l + b*e_b, where e_l and e_b are unit
    vectors in the l and b directions. In this case R_hat would not even be a straight line in real space due to the curved nature of the
    spherical Galactic coordinates).

    Note the spherical tilt is the same as the vertex deviation if we are working with vR-vphi velocities in xy space. However, in contrast
    with the calculate_tilt function, if absolute=True this function returns the absolute value of the spherical tilt, and the range changes 
    from (-90,90] to [0,90] deg.
    
    I follow the convention in Hagen et al. (2019) and define the angle as positive when the semi-major axis of the ellipse is tilted anti-
    clockwise with respect to R_hat, and negative otherwise.
    """ 
    cov = np.cov(vx,vy)
    max_vector = EF.get_max_vector(cov) # Direction of maximum dispersion
    spherical_tilt = calculate_spherical_tilt_helper(R_hat,max_vector)
    return abs(spherical_tilt) if absolute else spherical_tilt

def calculate_spherical_tilt_helper(R_hat, max_vector, visualise=False):
    """
    R_hat: see calculate_spherical_tilt docstring
    max_vector: vector defining the direction of highest dispersion in the plane
    """

    dot_prod = np.dot(R_hat,max_vector)
    fraction = dot_prod / (np.linalg.norm(R_hat)*np.linalg.norm(max_vector))
    
    # alpha is the angle between R_hat and the semi-major axis of the velocity ellipse as computed by arccos. 
    # BUT the range of arccos and the sign definition requires applying a correction (that's what all the code below does). 
    # Namely, arccos gives values in the range [0,180] but we want the spherical_tilt to range from -90 to 90, and we want it to 
    # be measured from R_hat onto max_vector (following the Hagen et al. (2019) sign convention I mentioned in the `calculate_spherical_tilt` docstring).
    alpha = np.degrees(np.arccos(fraction))
    
    # Azimuthal angle of R_hat
    phi_central = np.degrees(np.arctan2(R_hat[1],R_hat[0]))
    
    # Azimuthal angle of the ellipse's semi-major axis
    phi_vector = np.degrees(np.arctan2(max_vector[1],max_vector[0]))

    # Change range from (-180,180] to (0,360]
    if phi_central < 0:
        phi_central += 360
    if phi_vector < 0:
        phi_vector += 360
    
    # Divide plane into 4 quadrants, travelling anti-clockwise from the azimuthal angle of R_hat (phi_central)
    # The different quadrants are defined by the different beta angles. Note beta_array[0] = phi_central
    beta_array = [(phi_central + n*90)%360 for n in [0,1,2,3]]
    beta_array.append(beta_array[0]) #complete the cycle

    for i in range(len(beta_array) - 1): # -1 because we are doing [i+1] inside the loop
        
        # If True, quadrant i has the x-axis crossing through it
        found_intersection = beta_array[i] > beta_array[i+1]
        
        found_beta_intersection = found_intersection and (beta_array[i] <= phi_vector or phi_vector < beta_array[i+1])
        
        found_beta = not found_intersection and (beta_array[i] <= phi_vector and phi_vector < beta_array[i+1])
        
        if found_beta or found_beta_intersection:
            
            # First quadrant (i.e. phi_vector is less than 90° greater than phi_central) so alpha as returned by arccos lies in the range from 0 to 90. It requires no correction.
            if i == 0:
                spherical_tilt = alpha
                
            # Second quadrant so alpha as returned by arccos lies in the range from 90 to 180.
            # We need to reflect across the origin to use the other extreme of the semi-major axis, and have values from -90 to 0
            elif i == 1:
                spherical_tilt = alpha - 180
                
            # Third quadrant so alpha as returned by arccos lies in the range from 90 to 180.
            # We need to reflect across the origin to use the other extreme of the semi-major axis, and have values from 0 to 90
            elif i == 2:
                spherical_tilt = 180 - alpha
                
            # Fourth quadrant so alpha as returned by arccos lies in the range from 0 to 90.
            # We need to reflect across R_hat and have values from -90 to 0
            elif i == 3:
                spherical_tilt = -alpha
            break
    
    if visualise:
        print(beta_array)
        print(phi_vector)
        print(i)
        
        MP.visualise_spherical_tilt_calculation(beta_array[:-1], phi_vector)
    
    return spherical_tilt