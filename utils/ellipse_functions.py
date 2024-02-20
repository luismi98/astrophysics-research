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

def get_max_vector(varx,vary,covxy):
    """
    Get the coordinates of the semi-major axis, with radius 3 sigma along that direction
    """
    cov = [[varx,covxy],[covxy,vary]]
    eigenvalues, eigvectors = np.linalg.eig(cov)
    raw_max_vector = eigvectors[:,eigenvalues.argmax()]
    max_vector = raw_max_vector*3*np.sqrt(np.max(eigenvalues))
    
    return max_vector