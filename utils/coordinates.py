import numpy as np
from galpy.util import coords as bovy_coords

def ang_to_rect_1D(ang,d=None,x=None):
    """
    Transform from "l" to "y", or from "b" to "z".
    
    Parameters
    ----------
    ang: float
        The value of an angular variable, "l" or "b", in degrees.
    d: float, optional
        Distance from Sun. Must be given if x is None.
    x: float, optional
        Heliocentric rectangular coordinate growing in the direction from the Sun (x=0) to the GC (x=R0).

    Returns
    -------
    rect: float
        The rectangular coordinate corresponding to the angular coordinate given.
    """
    if (d is None) + (x is None) != 1:
        raise ValueError("Give a value for `d` or `x` (but not both).")
    
    return d*np.sin(np.radians(ang)) if x is None else x*np.tan(np.radians(ang))
    
def rect_to_ang_1D(rect,d=None,x=None):
    """
    Transform from "y" to "l", or from "b" to "z".
    
    Parameters
    ----------
    rect: float
        The value of a rectangular variable, "y" or "z", in degrees.
    d: float, optional
        Distance from Sun. Must be given if x is None.
    x: float, optional
       Heliocentric rectangular coordinate growing in the direction from the Sun (x=0) to the GC (x=R0)

    Returns
    -------
    ang: float
        The angular coordinate corresponding to the rectangular coordinate given, in degrees.
    """
    if (d is None) + (x is None) != 1:
        raise ValueError("Give a value for `d` or `x` (but not both).")

    return np.degrees(np.sin(rect/d)) if x is None else np.degrees(np.tan(rect/x))

def get_bar_angle():
    return 27

def get_solar_height():
    """
    We know it is not truly 0, see Hawthorn & Gerhard 2016. However, it is small so I am setting it to 0 mainly because I have not looked enough into how a non-zero 
    value affects the Galactic coordinate system (i.e. if we want b=0 to coincide with the true Galactic plane, then it would cease to be centered in the Sun)
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

def get_conversion_mas_to_rad():
    """
    1 mas = 10^-3 * (1 deg / 3600 arcsec) * (pi rad / 180 deg) = 10^3*pi/(3600*180) rad
    """
    return 1e-3*np.pi/(180 * 3600)

def get_conversion_yr_to_s():
    """
    1yr = 1yr * 365 day/yr * 24 h/day * 3600 s/h = 3.1536e7 s
    """
    return 365*24*3600

def get_conversion_kpc_to_km():
    """
    1 kpc = 3.086e16 km
    """
    return 3.086e16

def convert_pm_to_velocity(distance, pm, kpc_bool=True, masyr_bool=True):
    """
    Multiplying the proper motions (pm) by the distance to the Sun (d) gives the velocities.

    Note, when passing longitudinal proper motion (pml) values, they need to come multiplied by cos(b), 
    which is how they are automatically returned from using the function bovy_coords.vxvyvz_to_vrpmllpmbb()
    The reason is because of projection effects on the surface of a sphere. Think of the case on Earth:
    Travelling 1 degree of longitude at the equator means travelling a much longer distance than near the North Pole.
    Therefore, a certain pml at b=0 is multiplied by cos(0)=1 but at b=20 we have to diminish it by cos(b)=0.93 when converting to velocity.

    Parameters
    ----------
    distance: float array
        Distance in kpc if kpc_bool, otherwise in km
    pm: float array
        Proper motion in mas/yr if masyr_bool, otherwise in rad/s
    """

    velocity = distance * pm

    if kpc_bool:
        velocity *= get_conversion_kpc_to_km()
    
    if masyr_bool:
        velocity *= get_conversion_mas_to_rad()/get_conversion_yr_to_s()

    return velocity

def oscar_radial_velocity_frame_change(velocity, long, lat, degrees = True, change_to_GSR = True):
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
    
########################################################################################################################

def lbd_to_xyz(df, GSR=True, R0=get_solar_radius(), Z0=get_solar_height()):
    """
    Here I am assuming if you want velocities in the GSR, you will want Galactocentric positions.
    This need not be the case though... as Galactic coordinates (and even velocities) for example are Heliocentric even if we're working in the GSR.
    """

    # Heliocentric
    XYZ = bovy_coords.lbd_to_XYZ(df['l'],df['b'],df['d'],degree=True)

    if not GSR:
        df['x'],df['y'],df['z'] = XYZ[:,0],XYZ[:,1],XYZ[:,2]
        return

    # Galactocentric
    xyz = bovy_coords.XYZ_to_galcenrect(XYZ[:,0],XYZ[:,1],XYZ[:,2],Xsun=-R0,Zsun=Z0)
    df['x'],df['y'],df['z'] = xyz[:,0],xyz[:,1],xyz[:,2]
    
def xyz_to_Rphiz(df):
    # Note strangely enough the cylindrical functions return tuples, unlike the others

    o_Rphiz = bovy_coords.rect_to_cyl(df['x'],df['y'],df['z'])
    df['R'],df['phi'],df['z'] = o_Rphiz[0],np.degrees(o_Rphiz[1]),o_Rphiz[2]  #Note: converting phi to degrees

    #Phi between 0 and 360
    df.loc[df.phi < 0, 'phi'] += 360

def Rphiz_to_xyz(df):
    xyz = bovy_coords.cyl_to_rect(df['R'],df['phi'],df['z'])
    df['x'],df['y'],df['z'] = xyz[0],xyz[1],xyz[2]

def xyz_to_XYZ(df, R0=get_solar_radius(),Z0=get_solar_height()):
    XYZ=bovy_coords.galcenrect_to_XYZ(df.x.values,df.y.values,df.z.values, Xsun=-R0,Zsun=Z0)
    df['X'],df['Y'],df['Z'] = XYZ[:,0], XYZ[:,1], XYZ[:,2]

def XYZ_to_lbd(df):
    lbd=bovy_coords.XYZ_to_lbd(df.X.values,df.Y.values,df.Z.values,degree=True)
    df['l'],df['b'],df['d'] = lbd[:,0], lbd[:,1], lbd[:,2]

    # Set longitude range from 0,360 to -180,180
    df.loc[df.l>180, 'l'] -= 360

def lb_to_radec(df):
    radec=bovy_coords.lb_to_radec(df.l.values,df.b.values,degree=True)
    df['ra'],df['dec']=radec[:,0],radec[:,1]

def radec_to_lb(df):
    lb=bovy_coords.radec_to_lb(df.ra.values,df.dec.values,degree=True)
    df['l'],df['b']=lb[:,0],lb[:,1]

########################################################################################################################

def pmrapmdec_to_pmlpmb(df):
    # Equatorial proper motions to galactic
    pmbpml = bovy_coords.pmrapmdec_to_pmllpmbb(df['pmra'], df['pmdec'], df['ra'], df['dec'], degree=True)
    df['pmlcosb'],df['pmb'] = pmbpml[:,0],pmbpml[:,1]

def pmlpmb_to_pmrapmdec(df):
    pmradec=bovy_coords.pmllpmbb_to_pmrapmdec(df.pmlcosb.values,df.pmb.values,
                                     df.l.values,df.b.values,degree=True)
    df['pmra'],df['pmdec']=pmradec[:,0],pmradec[:,1]
    
def vrpmlpmb_to_vxvyvz(df, v_sun, R0=get_solar_radius(), Z0=get_solar_height()):
    """
    If v_sun is non-zero, the returned rectangular velocities are Galactocentric, otherwise Heliocentric.
    """

    vXvYvZ = bovy_coords.vrpmllpmbb_to_vxvyvz(df['vr'],df['pmlcosb'],df['pmb'],df['l'],df['b'],df['d'],degree=True)

    # If v_sun is [0,0,0], the following conversion will be useless except that it applies a tiny rotation to align with Astropy's system
    vxvyvz = bovy_coords.vxvyvz_to_galcenrect(vXvYvZ[:,0],vXvYvZ[:,1],vXvYvZ[:,2], vsun=v_sun,Xsun=-R0,Zsun=Z0)
    df['vx'],df['vy'],df['vz'] = vxvyvz[:,0],vxvyvz[:,1],vxvyvz[:,2]

def vxvyvz_to_vrpmlpmb(df):
    vrpmlpmb = bovy_coords.vxvyvz_to_vrpmllpmbb(df["vx"], df["vy"], df["vz"], df["l"], df["b"], df["d"], degree=True)
    df["vr"], df["pmlcosb"], df["pmb"] = vrpmlpmb[:,0],vrpmlpmb[:,1],vrpmlpmb[:,2]

def vXvYvZ_to_vrpmlpmb(df):
    #vr in km/s, pmlcosb and pm (proper motion) in mas/yr (milli-arcsecond per year)
    vrpmlpmb=bovy_coords.vxvyvz_to_vrpmllpmbb(df.vX.values,df.vY.values,df.vZ.values,
                                        df.l.values,df.b.values,df.d.values,
                                        degree=True)
    df['vr'],df['pmlcosb'],df['pmb'] = vrpmlpmb[:,0], vrpmlpmb[:,1], vrpmlpmb[:,2]

def vxvyvz_to_vRvphivz(df):
    # Note strangely enough the cylindrical functions return tuples, unlike the others

    vRphiz = bovy_coords.rect_to_cyl_vec(df['vx'],df['vy'],df['vz'],df['x'],df['y'],df['z'])
    df['vR'],df['vphi'],df['vz'] = vRphiz[0],vRphiz[1],vRphiz[2]

def vRvphivz_to_vxvyvz(df):
    vxvyvz = bovy_coords.cyl_to_rect_vec(df['vR'],df['vphi'],df['vz'],df["phi"])
    df['vx'],df['vy'],df['vz'] = vxvyvz[0],vxvyvz[1],vxvyvz[2]

def pmlpmb_to_vlvb(df):
    df['vl'] = convert_pm_to_velocity(df.d.values,df.pmlcosb.values)
    df['vb'] = convert_pm_to_velocity(df.d.values,df.pmb.values)

def vxvyvz_to_vXvYvZ(df,v_sun,R0=get_solar_radius(),Z0=get_solar_height()):
    # If v_sun is [0,0,0], the only thing this will do is perform a tiny rotation to align with astropy's frame definition
    vXYZ=bovy_coords.galcenrect_to_vxvyvz(df.vx.values,df.vy.values,df.vz.values,
                                        vsun=v_sun,Xsun=-R0,Zsun=Z0)
    df['vX'],df['vY'],df['vZ'] = vXYZ[:,0], vXYZ[:,1], vXYZ[:,2]

def vxvy_to_vMvm(df,rot_angle=get_bar_angle()):
    """
    Get velocities along the bar axes.
    """

    c = np.cos(np.radians(rot_angle))
    s = np.sin(np.radians(rot_angle))

    df['vM'] = c*df['vx']-s*df['vy'] # semi-major axis
    df['vm'] = s*df['vx']+c*df['vy'] # semi-minor axis