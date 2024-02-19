import numpy as np

def b_to_z(b,d):
    return d*np.tan(np.radians(b))
def z_to_b(z,d):
    return np.degrees(np.arctan(z/d))

def get_solar_height():
    """
    We know it is not truly 0, see Hawthorn & Gerhard 2016. However, it is small so I am setting it to 0 mainly because I not looked enough into how a non-zero 
    value affects the Galactic coordinate system (i.e. if we want b=0 to coincide with the true Galactic plane, then it ceases to be centered in the Sun)
    """
    return 0

def get_solar_radius():
    # Gravity 2018
    return 8.1

def get_solar_velocity(changing_reference_frame):
    """
    The solar velocity is required in some bovy_coords velocity conversions across coordinate systems.

    If you want to change reference frame (from LSR to GSR or vice-versa), set changing_reference_frame = True
    
    Observations by default are measured from Earth (or the Sun, as usually approximated). Observationalists in some papers
    correct for v_sun to work in the GSR, others do not. This is because the solar velocity can be tricky to measure and 
    hence it introduces a source of error.

    We want to measure it in the GSR because it provides a clearer picture of stars' motion in the context of the Galaxy as a whole,
    rather than relative to our specific motion.

    To work in the GSR:
    - For the simulation 
        All the velocities are already given in the GSR, so set `changing_reference_frame = False`.
        That will set v_sun to [0,0,0], and so the bovy_coords functions will not change the frame.
    - For the observations
        They need changing to the GSR, so set changing_reference_frame = True.
    """
            # Drimmel 2018
    return [12.9,245.6,7.78] if changing_reference_frame else [0,0,0]

def convert_pm_to_velocity(distance, pm):
    """
    Multiplying the proper motions (pm) by the distance to the Sun (d) gives the velocities.

    Note, when passing longitudinal proper motion (pml) values, they need to come multiplied by cos(b), 
    which is how they are automatically returned from using the function bovy_coords.vxvyvz_to_vrpmllpmbb()
    The reason is because of projection effects on the surface of a sphere. Think of the case on Earth:
    Travelling 1 degree of longitude at the equator means travelling a much longer distance than near the North Pole.
    Therefore, a certain pml at b=0 is multiplied by cos(0)=1 but at b=20 we have to diminish it by cos(b)=0.93 when converting to velocity.

    Convert the distance from kpc to km
    1 kpc = 3.086e16 km

    Convert the pm from mas/yr (milli-arcsec per year) to rad/s
    1 mas = 10^-3 * (1 deg / 3600 arcsec) * (pi rad / 180 deg) = 10^3*pi/(3600*180) rad
    1/1yr = 1/1yr * 1yr/3.1536e7 s = 1/3.1536e7 1/s

    Parameters
    ----------
    distance: float array
        Distance in kpc
    pm: float array
        Proper motion in mas/yr
    """

    d_factor = 3.086e16 * distance # km
    pm_conversion = 1e-3*np.pi/(180 * 3600 * 3.1536e7) # rad/s

    velocity = d_factor * pm_conversion * pm # km/s
    return velocity

def Oscar_radial_velocity_frame_change(velocity, long, lat, degrees = True, change_to_GSR = True):
    """
    Oscar's formula to convert radial velocity between LSR and GSR.
    It's referenced in Howard et al. 2008 and Ness et al. 2013

    Set `degrees = True` if `long` and `lat` are given in degrees.
    """
    
    if degrees:
        long = np.radians(long)
        lat = np.radians(lat)
    conversion_shift = 220.*(np.sin(long)*np.cos(lat)) + 16.5*(np.sin(lat)*np.sin(np.deg2rad(25.)) + np.cos(lat)*np.cos(np.deg2rad(25.))*np.cos(long-np.deg2rad(53.)))
    
    if change_to_GSR:
        return velocity + conversion_shift
    else:
        return velocity - conversion_shift