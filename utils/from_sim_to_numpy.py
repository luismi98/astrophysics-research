import pandas as pd
import numpy as np
import os
import pynbody #https://pynbody.github.io/pynbody/reference/essentials.html
from galpy.util import coords as bovy_coords #https://docs.galpy.org/en/v1.6.0/reference/bovycoords.html
import coordinates
from load_sim import build_filename

BAR_ANGLE = 27
Z0_CONST = coordinates.get_solar_height()
R0_CONST = coordinates.get_solar_radius()

def calculate_bar_angle_from_inertia_tensor(x,y,mass):
    if (len(x) != len(y) or len(x) != len(mass)):
        raise ValueError("The length of the star's position and mass arrays did not match.")
    
    #Calculate the moment of inertia tensor
    I_xx, I_yy, I_xy = np.sum(mass*y**2), np.sum(mass*x**2), np.sum(mass*-1*x*y)
    I = np.array([[I_xx, I_xy], [I_xy, I_yy]])
    
    #Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(I)
    
    #The major axis (direction of the bar) is the direction of the eigenvector with smallest eigenvalue
    #That is because the bar has the smallest moment of inertia when rotated around the longitudinal axis of the bar
    index_lowest_eigenvalue = eigenvalues.argmin()
    major_axis = eigenvectors[:, index_lowest_eigenvalue]

    #Get the angle the major axis makes with the horizontal axis
    return np.degrees(np.arctan2(major_axis[1], major_axis[0]))

def extract_xymass_from_stars(stars):
    x = np.array(stars['x'].in_units('kpc'))
    y = np.array(stars['y'].in_units('kpc'))
    mass = np.array(stars['mass'].in_units('Msol'))

    return x,y,mass

def get_bar_stars(sim, tform_min=3, tform_max=6, zmin=0.2, Rmax=4):
    """
    - tform_min and tform_max default to 3 and 6, choosing stars between 4 and 7 years old
    - zmin defaults to 0.2 to avoid the spheroidal densest region in the plane
    """

    return sim.s[
            (sim.s['tform'] > tform_min)&
            (sim.s['tform'] < tform_max)&
            (np.abs(sim.s['z'].in_units("kpc")) > zmin)&
            (np.hypot(sim.s["x"].in_units("kpc"), sim.s["y"].in_units("kpc")) < Rmax)
        ]

def compute_bar_angle(sim,I_radius=4):

    bar_stars = get_bar_stars(sim=sim,Rmax=I_radius)

    x_bar, y_bar, mass_bar = extract_xymass_from_stars(bar_stars)
    
    maj_axis_angle = calculate_bar_angle_from_inertia_tensor(x_bar,y_bar,mass_bar)
    
    return maj_axis_angle

def apply_factors(df, position_factor, velocity_factor):
    #scale bar from 3kpc to 5kpc (size of Milky Way's bar)
    #Spatial factor should be 5/3=1.7 - I modified it to 1.3 and then 1.5 to investigate Oscar's result (31st March 2021)
    df.x = df.x*position_factor
    df.y = df.y*position_factor
    df.z = df.z*position_factor

    df.vx = df.vx*velocity_factor #Debattista et al 2017 - Ask Steven?
    df.vy = df.vy*velocity_factor
    df.vz = df.vz*velocity_factor

def flip_Lz(df):
    """
    Flip Lz to match the rotation of the MW

    Using pynbody.analysis.angmom.faceon(whole_sim, cen=(0,0,0)) aligns Lz so that the galaxy rotates 
    ANTI-CLOCKWISE with z to the North. Having the x axis towards the Sun, y in the direction of rotation,
    and z to the North is a right-hand system. 
    We have aligned the bar with the x-axis and rotated ANTI-CLOCKWISE by say 27° degrees. But we know the
    Milky Way rotates clockwise and that the bar angle is 27° CLOCKWISE with respect to the Sun-GC LOS
    Therefore, flip the galaxy 180° about the x-axis (by multiplying the y and vy components by -1, see
    drawing in logbook). That makes the galaxy rotate clockwise, with the y axis along the rotation. 
    Then change the sign of z and vz to have a right-handed system.
    """
    
    df.z = df.z*(-1)
    df.y = df.y*(-1)
    df.vz = df.vz*(-1)
    df.vy = df.vy*(-1)

def transform_coordinates(df, R0=R0_CONST, Z0=Z0_CONST, GSR=True, rot_angle=BAR_ANGLE):

    v_sun = coordinates.get_solar_velocity(not GSR)
    if GSR: assert v_sun == [0,0,0], "`v_sun` needs to be zero for the simulation as velocities are already in the GSR"

    # Heliocentric rectangular  ---------------------------------------------------------------------------------------------------
    #Transform from rectangular galacto-centric (center of Galaxy is x=y=z=0) to rectangular heliocentric (Sun is x=y=z=0)
    #Xsun and Zsun are the cylindrical distance of the Sun from the GC and its height above the Galatic plane respectively
    XYZ=bovy_coords.galcenrect_to_XYZ(df.x.values,df.y.values,df.z.values, Xsun=-R0,Zsun=Z0)
    df['X'],df['Y'],df['Z'] = XYZ[:,0], XYZ[:,1], XYZ[:,2]

    # If galactocentric=True, v_sun will be [0,0,0] and the only thing this will do is perform a tiny rotation to align with astropy's frame definition
    vXYZ=bovy_coords.galcenrect_to_vxvyvz(df.vx.values,df.vy.values,df.vz.values,
                                        vsun=v_sun,Xsun=-R0,Zsun=Z0)
    df['vX'],df['vY'],df['vZ'] = vXYZ[:,0], vXYZ[:,1], vXYZ[:,2]

    # spherical Galactic  --------------------------------------------------------------------------------------------------------

    # Cart to spherical Galactic Coords https://en.wikipedia.org/wiki/Galactic_coordinate_system
    #If degree=True, then l=longitude(degrees) b=latitude(degrees) d=distance from GC(kpc)
    lbd=bovy_coords.XYZ_to_lbd(df.X.values,df.Y.values,df.Z.values,degree=True)
    df['l'],df['b'],df['d'] = lbd[:,0], lbd[:,1], lbd[:,2]

    # Set longitude range from 0,360 to -180,180
    df.loc[df.l>180, 'l'] -= 360

    # rectangular to spherical Galactic (heliocentric but in GSR if GSR=True)
    #vr in km/s, pmlcosb and pm (proper motion) in mas/yr (milli-arcsecond per year)
    pmlbvr=bovy_coords.vxvyvz_to_vrpmllpmbb(df.vX.values,df.vY.values,df.vZ.values,
                                        df.l.values,df.b.values,df.d.values,
                                        degree=True)
    df['vr'],df['pmlcosb'],df['pmb'] = pmlbvr[:,0], pmlbvr[:,1], pmlbvr[:,2]

    # Convert proper motions to velocities. Requires d in kpc and pm in mas/yr
    df['vl'] = coordinates.convert_pm_to_velocity(df.d.values, df.pmlcosb.values) # km/s
    df['vb'] = coordinates.convert_pm_to_velocity(df.d.values, df.pmb.values)

    # Cylindrical  ------------------------------------------------------------------------------------------------------------

    rect_to_cyl(df)
    
    # Bar frame ----------------------------------------------------------------------------------------------------------------

    c = np.cos(np.radians(rot_angle))
    s = np.sin(np.radians(rot_angle))

    df['vM'] = c*df['vx']-s*df['vy'] # along semi-major axis
    df['vm'] = s*df['vx']+c*df['vy'] # along semi-minor axis

    # Equatorial Coords ----------------------------------------------------------------------------------------------------------------
    # #ra=right ascension, dec=declination, both in degrees if degree=True
    # radec=bovy_coords.lb_to_radec(df.l.values,df.b.values,degree=True)
    # df['ra'],df['dec']=radec[:,0],radec[:,1]
    # del radec
    # pmradec=bovy_coords.pmllpmbb_to_pmrapmdec(df.pml.values,df.pmb.values,
    #                                  df.l.values,df.b.values,degree=True)
    # df['pmra'],df['pmdec']=pmradec[:,0],pmradec[:,1]
    # del pmradec

def rect_to_cyl(df):
    # Note strangely enough these functions return tuples, unlike the others

    Rphiz = bovy_coords.rect_to_cyl(df['x'],df['y'],df['z'])
    df['R'],df['phi'],df['z'] = Rphiz[0],np.degrees(Rphiz[1]),Rphiz[2] #Note: converting phi to degrees
    
    #Phi between 0 and 360
    df.loc[df.phi < 0, 'phi'] += 360

    vRphiz = bovy_coords.rect_to_cyl_vec(df['vx'],df['vy'],df['vz'],df['x'],df['y'],df['z'])
    df['vR'],df['vphi'],df['vz'] = vRphiz[0],vRphiz[1],vRphiz[2]

def cyl_to_rec(df):
    # Note strangely enough these functions return tuples, unlike the others

    xyz = bovy_coords.cyl_to_rect(df['R'],df['phi'],df['z'])
    df['x'],df['y'],df['z'] = xyz[0],xyz[1],xyz[2]

    vxvyvz = bovy_coords.cyl_to_rect_vec(df['vR'],df['vphi'],df['vz'],df["phi"])
    df['vx'],df['vy'],df['vz'] = vxvyvz[0],vxvyvz[1],vxvyvz[2]

def axisymmetrise(df):
    rect_to_cyl(df)
    
    df['phi'] += 360*np.random.random(len(df))
    df.loc[df['phi']>360, 'phi'] -= 360

    df.drop(columns = ['x','y','vx','vy'], inplace=True)

    cyl_to_rec(df)

def convert_sim_to_df(sim_stars, pos_factor=1.7, vel_factor=0.48, R0=R0_CONST,Z0=Z0_CONST,angle=27, zabs=True, GSR=True, axisymmetric=False):
    positions = np.array(sim_stars['pos'].in_units('kpc'))
    velocities = np.array(sim_stars['vel'].in_units('km s**-1'))
    tform = np.array(sim_stars['tform']) #https://pynbody.github.io/pynbody/reference/derived.html

    df = pd.DataFrame()
    df['x'],df['y'],df['z'] = positions[:,0],positions[:,1],positions[:,2]         
    df['vx'],df['vy'],df['vz'] = velocities[:,0],velocities[:,1],velocities[:,2]
    df['age'] = tform.max() - tform
    
    if axisymmetric:
        axisymmetrise(df)

    apply_factors(df, pos_factor, vel_factor)
    flip_Lz(df)

    # mirror below the plane onto above the plane to increase stats
    if zabs:
        df.loc[df.z<0, 'vz'] *= -1
        df.loc[df.z<0, 'z'] *= -1

    transform_coordinates(df, R0=R0, Z0=Z0, rot_angle=angle, GSR=GSR)

    df.drop(columns = ['X','Y','Z','vX','vY','vZ','pmlcosb','pmb'], inplace=True)
    return df

def save_as_np(df, save_path, filename):
    
    sim_array = np.array(df.values.astype(np.float32))
    column_array = np.array(df.columns.to_list())

    np.save(save_path + filename, sim_array)
    print("Saved:", save_path+filename)

    np.save(save_path + filename + "_columns", column_array)
    print("Saved:", save_path+"columns")

    with open(save_path + filename + "_columns.txt", 'w') as f:
        f.write(f"Saved simulation with datatype np.float32.\nThe columns are: {list(column_array)}")

def load_pynbody_sim(filepath):
    sim = pynbody.load(filepath)

    #https://pynbody.github.io/pynbody/reference/analysis.html
    #Calculates the centre of mass and re-centers the simulation
    #'hyb' mode is the shrink sphere method but faster because starts near a potential minimum 
    pynbody.analysis.halo.center(sim, mode="hyb")
    
    #Re-positions and rotates the simulation such that the disk lies in the x-y plane
    pynbody.analysis.angmom.faceon(sim, cen=(0,0,0))

    return sim

def load_process_and_save(simulation_filepath, save_path, angle_list = [BAR_ANGLE], axisymmetric=False,pos_factor = 1.7, vel_factor = 0.48,\
                           R0=R0_CONST, Z0=Z0_CONST, zabs = False, GSR = True, I_radius=4, choice="708main"):
    if not os.path.isfile(simulation_filepath):
        raise FileNotFoundError(f"File not found at: `{simulation_filepath}`.")
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"Directory to save to not found: `{save_path}`.")

    sim = load_pynbody_sim(simulation_filepath)

    if axisymmetric:
        df = convert_sim_to_df(sim.s, pos_factor=pos_factor, vel_factor=vel_factor, R0=R0, Z0=Z0, zabs=zabs, GSR=GSR, axisymmetric=axisymmetric)

        filename = build_filename(choice=choice,R0=R0,Z0=Z0,axisymmetric=True,zabs=zabs,pos_factor=pos_factor,GSR=GSR)

        save_as_np(df, save_path=save_path, filename=filename)

        return

    bar_angle = compute_bar_angle(sim,I_radius=I_radius)
    sim.rotate_z(-bar_angle) # align with x axis
    
    for angle in angle_list:
        
        sim.rotate_z(angle) #anti-clockwise rotation as seen from above, as we will then flip y

        df = convert_sim_to_df(sim.s, pos_factor=pos_factor, vel_factor=vel_factor, R0=R0, angle=angle, zabs=zabs, GSR=GSR)

        filename = build_filename(choice=choice,rot_angle=angle,R0=R0,Z0=Z0,axisymmetric=False,zabs=zabs,pos_factor=pos_factor,GSR=GSR)

        save_as_np(df, save_path=save_path, filename=filename)

        if len(angle_list) > 1:
            # I could not do copy.copy(sim) to create a deep copy, see https://stackoverflow.com/questions/58415397/
            # Therefore, I re-use the same sim object in all iterations. Rotate it back so I can use it again.
            sim.rotate_z(-angle)