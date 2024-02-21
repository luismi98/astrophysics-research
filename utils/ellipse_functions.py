import numpy as np
import compute_variables as CV
import plotting_helpers as PH

def get_vel_ellipse_coords(varx,vary,covxy,ellipse_factor=1):
    """
    Get the coordinates of the velocity ellipse, with radius of 2 sigma in each respective direction
    """
    cov = [[varx,covxy],[covxy,vary]]
    eigenvalues = np.linalg.eig(cov)[0]
    radius = ellipse_factor*2*np.sqrt(np.max(eigenvalues))
    ratio = np.sqrt(np.min(eigenvalues)/np.max(eigenvalues))
    tilt = CV.calculate_tilt_from_moments(varx,vary,covxy)
    x_ellipse, y_ellipse = PH.get_ellipse_coords(radius, ratio, tilt)
    
    return x_ellipse, y_ellipse

def get_max_vector(cov, factor=3):
    """
    Get the (x,y) vector coordinates of the semi-major axis of the ellipse described by the given covariance matrix.
    The vector returned has a length (norm) of the standard deviation along that direction multiplied by `factor`.
    """
    eigenvalues, eigvectors = np.linalg.eig(cov)
    raw_max_vector = eigvectors[:,eigenvalues.argmax()]
    max_vector = raw_max_vector*factor*np.sqrt(np.max(eigenvalues))
    
    return max_vector

def get_max_vector_from_moments(varx,vary,covxy):
    cov = [
        [varx, covxy],
        [covxy, vary]
    ]
    return get_max_vector(cov)