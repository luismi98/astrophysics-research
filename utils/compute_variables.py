import numpy as np
import ellipse_functions as EF
import velocity_plot
import matplotlib.pyplot as plt

# Note in all the code in this file vx & vy are placeholders for any desired velocity components.

# TRIVIAL FUNCTIONS -------------------------------------------------------------------------

def calculate_covariance(vx,vy):
    cov = np.cov(vx,vy)
    return cov[0,1]

def calculate_correlation(vx,vy):
    cov = np.cov(vx,vy)
    return cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])

def calculate_anisotropy(vx,vy):
    cov = np.cov(vx,vy)
    return 1-cov[1,1]/cov[0,0]

def calculate_var_diff(vx,vy):
    cov = np.cov(vx,vy)
    return cov[0,0]-cov[1,1]

def calculate_cov_var_ratio(vx,vy):
    cov = np.cov(vx,vy)
    return 2*cov[0,1]/(cov[0,0]-cov[1,1])

# TILTS ------------------------------------------------------------------------------------

def calculate_tilt_from_moments(varx, vary, covxy, absolute=False):
    var_diff = abs(varx-vary) if absolute else varx-vary
    tilt = np.degrees(np.arctan2(2.*covxy, var_diff)/2.)
    return tilt

def calculate_tilt(vx, vy, absolute=False):
    """
    I use the name "tilt" as synonym for "vertex deviation". If absolute=False, the tilt is the angle the semi-major axis of the velocity ellipse makes with the
    horizontal velocity axis direction, vx, (eg line-of-sight direction if working with vrvl velocities), and its range is (-90,90] deg. 
    
    If absolute=True, the function takes the absolute value of the dispersion difference before computing the tilt, and the range is reduced to [-45,45] deg.
    In this case, the tilt is the angle that the semi-major axis of the velocity ellipse makes with the coordinate axis that it is closest to.
    
    The sign of the tilt is the same independent of `absolute`, and always equal to the sign of the correlation. If positive, the semi-major axis falls
    within the 1st (& 3rd) quadrant of the vx-vy plane. If negative, it falls on the 4th (& 2nd) quadrant.
    """

    cov = np.cov(vx, vy)
    return calculate_tilt_from_moments(cov[0,0],cov[1,1],cov[0,1],absolute=absolute)

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
    varx,vary,covxy = cov[0,0],cov[1,1],cov[0,1]
    max_vector = EF.get_max_vector(varx,vary,covxy) # Direction of maximum dispersion
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
        
        visualise_spherical_tilt_calculation(beta_array[:-1], phi_vector)
    
    return spherical_tilt

def visualise_spherical_tilt_calculation(beta_array, phi_vector):
    fig,ax = plt.subplots()

    for i,ang in enumerate(beta_array):
        ang *= np.pi/180
        ax.plot([0,np.cos(ang)],[0,np.sin(ang)],"k--")
        ax.text(np.cos(ang),np.sin(ang),s=str(i),color="r")
    
    phi_vector *= np.pi/180
    
    ax.plot([0,np.cos(phi_vector)],[0,np.sin(phi_vector)],color="g")

    ax.set_aspect("equal")
    plt.show()

# BOOTSTRAP ERROR ESTIMATION --------------------------------------------------------------------------------

def get_std_bootstrap(function, vx, vy=None, tilt=False, absolute=False, R_hat = None, size_fraction = 1, repeat = 100, give_values = False, show_vel_plots = False, show_freq = 10, velocity_kws={}):   
    '''
    Computes the standard deviation of a function applied to velocity components 
    using bootstrap resampling.

    Parameters:
    - vx, vy: array-like
        Arrays of desired velocity components. vy=None if only 1 is needed.

    - function: callable
        Function used to compute the desired variable whose error you want to estimate.

    - tilt: bool, optional
        Boolean variable indicating whether the error is being estimated on a tilt 
        (including spherical tilt). Default is False.

    - absolute: bool, optional
        Only has effect if tilt=True. Determines whether the tilt is being computed 
        with absolute value in the denominator or not. Default is False.

    - R_hat: array-like or None, optional
        Only has effect if tilt=True. If None, the tilt is a vertex deviation, otherwise 
        it is a spherical tilt. Default is None.

    - size_fraction: float, optional
        Size of bootstrap samples as fraction of original sample. According to 
        https://stats.stackexchange.com/questions/263710, it should be 1. Default is 1.

    - repeat: int, optional
        Number of bootstrap samples to take. Default is 100.

    - give_values: bool, optional
        If False, it only returns the estimated error. Otherwise it also returns an array 
        with all bootstrap values. Default is False.

    - show_vel_plots: bool, optional
        If True, velocity plots are shown. Default is False.

    - show_freq: int, optional
        Frequency of showing velocity plots. Default is 10.

    - velocity_kws: dict, optional
        Keyword arguments for the `velocity_plot` function. Default is an empty dictionary.

    Returns:
    - std: float
        The estimated standard deviation.

    - boot_values: array-like
        The bootstrap values. Only returned if `give_values` is True.
    '''
    
    if R_hat is None:
        true_value = function(vx,vy,absolute) if tilt else (function(vx,vy) if vy is not None else function(vx))
    else:
        true_value = function(vx,vy,R_hat,absolute=absolute)
    
    original_size = len(vx)
    indices_range = np.arange(original_size)
    
    boot_values = []
    bootstrap_size = int(size_fraction*original_size)
    
    for i in range(repeat):
        bootstrap_indices = np.random.choice(indices_range, size = bootstrap_size, replace=True)
        
        boot_vx = vx[bootstrap_indices]
        
        if vy is not None:
            boot_vy = vy[bootstrap_indices]
        
            if show_vel_plots and i%show_freq == 0:
                velocity_plot.velocity_plot(boot_vx,boot_vy,**velocity_kws)
        
        if R_hat is None:
            boot_val = function(boot_vx,boot_vy,absolute) if tilt else (function(boot_vx,boot_vy) if vy is not None else function(boot_vx))
        else:
            boot_val = function(boot_vx,boot_vy,R_hat,absolute=absolute)
            
        boot_values.append(boot_val)
        
    boot_values = np.array(boot_values)
    
    if tilt and not absolute:
        boot_values[(true_value - boot_values)>90] += 180
        boot_values[(true_value - boot_values)<-90] -= 180
    
    
    #Note this is a pseudo standard deviation: relative to true value as opposed to mean of bootstrap values
    std = np.sqrt(np.nanmean((boot_values-true_value)**2))
    
    if give_values:
        return boot_values, std
    else:
        return std
    
