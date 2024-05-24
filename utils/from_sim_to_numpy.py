import pandas as pd
import numpy as np
import os
import pynbody #https://pynbody.github.io/pynbody/reference/essentials.html

import utils.coordinates as coordinates
from utils.load_sim import build_filename

BAR_ANGLE = coordinates.get_bar_angle()
Z0_CONST = coordinates.get_solar_height()
R0_CONST = coordinates.get_solar_radius()

def calculate_bar_angle_from_inertia_tensor(x,y,mass):
    if not len(x) == len(y) == len(mass):
        raise ValueError("The length of the star's positions and mass arrays did not match.")
    
    #Calculate the moment of inertia tensor
    I_xx, I_yy, I_xy = np.sum(mass*y**2), np.sum(mass*x**2), -np.sum(mass*x*y)
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
    df.x = df.x*position_factor
    df.y = df.y*position_factor
    df.z = df.z*position_factor

    df.vx = df.vx*velocity_factor
    df.vy = df.vy*velocity_factor
    df.vz = df.vz*velocity_factor

def flip_Lz(df):
    """
    Flip Lz to match the rotation of the MW.

    Using pynbody.analysis.angmom.faceon(whole_sim, cen=(0,0,0)) as performed in `load_pynbody_sim()` aligns Lz so that the galaxy rotates
    ANTI-CLOCKWISE with z to the North. Moreover, the function `load_process_and_save()` aligns the bar with the x-axis and rotates it 
    ANTI-CLOCKWISE by the desired bar angle, say 27˚.

    We know the Milky Way rotates CLOCKWISE and that the bar angle is 27° CLOCKWISE with respect to the Sun-GC LOS. Therefore, flip the
    galaxy 180° about the x-axis (by multiplying the y and vy components by -1). This makes the model rotate CLOCKWISE, with the y axis along
    the rotation. 
    
    Then change the sign of z and vz to have a right-handed system. Note x grows from the Sun towards the GC.
    """
    df.y = df.y*(-1)
    df.vy = df.vy*(-1)

    df.z = df.z*(-1)
    df.vz = df.vz*(-1)

def transform_coordinates(df, R0=R0_CONST, Z0=Z0_CONST, GSR=True, rot_angle=None):

    v_sun = coordinates.get_solar_velocity(changing_reference_frame = not GSR)
    if GSR: assert v_sun == [0,0,0], "`v_sun` needs to be zero for the simulation as velocities are already in the GSR"

    coordinates.xyz_to_XYZ(df,R0=R0,Z0=Z0)
    coordinates.vxvyvz_to_vXvYvZ(df,v_sun=v_sun,R0=R0,Z0=Z0)

    coordinates.XYZ_to_lbd(df)
    coordinates.vXvYvZ_to_vrpmlpmb(df) # In GSR if GSR=True

    coordinates.pmlpmb_to_vlvb(df)

    coordinates.lb_to_radec(df)
    coordinates.pmlpmb_to_pmrapmdec(df)

    coordinates.xyz_to_Rphiz(df)
    coordinates.vxvyvz_to_vRvphivz(df)
    
    if rot_angle is not None: # None when axisymmetric
        coordinates.vxvy_to_vMvm(df,rot_angle=rot_angle)

def axisymmetrise(df):
    coordinates.xyz_to_Rphiz(df)
    coordinates.vxvyvz_to_vRvphivz(df)
    
    df['phi'] += 360*np.random.random(len(df))
    df.loc[df['phi']>360, 'phi'] -= 360

    df.drop(columns = ['x','y','vx','vy'], inplace=True)

    coordinates.Rphiz_to_xyz(df)
    coordinates.vRvphivz_to_vxvyvz(df)

def convert_sim_to_df(sim_stars, pos_factor=1.7, vel_factor=0.48, R0=R0_CONST,Z0=Z0_CONST,angle=BAR_ANGLE, zabs=True, GSR=True, axisymmetric=False):
    positions = np.array(sim_stars['pos'].in_units('kpc'))
    velocities = np.array(sim_stars['vel'].in_units('km s**-1'))
    tform = np.array(sim_stars['tform']) #https://pynbody.github.io/pynbody/reference/derived.html

    df = pd.DataFrame()
    df['x'],df['y'],df['z'] = positions[:,0],positions[:,1],positions[:,2]         
    df['vx'],df['vy'],df['vz'] = velocities[:,0],velocities[:,1],velocities[:,2]
    df['age'] = tform.max() - tform
    
    if axisymmetric:
        axisymmetrise(df)
        angle = None

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
    print("Saved:", save_path+filename+"_columns")

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
    """
    Load, process, and save a simulation file (.gz) into a numpy array (.npy).

    Parameters
    ----------
    simulation_filepath : str
        Path to the simulation file to be loaded.
    save_path : str
        Directory path where the processed numpy arrays will be saved.
    angle_list : list of float, optional
        List of angles in degrees by which to rotate the bar relative to the l=0˚ line, clockwise. Default is given in coordinates.get_bar_angle()
    axisymmetric : bool, optional
        If True, the simulation will be processed to be axisymmetric. Default is False.
    pos_factor : float, optional
        Factor by which to scale the positions. Default is 1.7, which scales 708main's bar from 3kpc to 5kpc (size of Milky Way's bar).
    vel_factor : float, optional
        Factor by which to scale the velocities. Default is 0.48. See Debattista et al., 2017.
    R0 : float, optional
        Solar radius in kpc. Default is given in coordinates.get_solar_radius()
    Z0 : float, optional
        Solar height in kpc. Default is given in coordinates.get_solar_height()
    zabs : bool, optional
        If True, mirror data below the plane to above the plane to increase statistics. Default is False.
    GSR : bool, optional
        If True, transforms will be done assuming Galactic Standard of Rest. Default is True.
    I_radius : float, optional
        Radius within which to consider stars for computing the bar angle. Default is 4 kpc.
    choice : str, optional
        String identifier for the filename. Default is "708main".

    Returns
    -------
    None

    Saves (onto `save_path`)
    ------------------------
    sim: 2D numpy array
        Contains a row per star, and a column per variable of interest (e.g. stellar age).
    
    columns: 1D numpy array
        Contains the name of each column in the sim.
        Current columns are: 
        ['x', 'y', 'z', 'vx', 'vy', 'vz', 'age', 'l', 'b', 'd', 'vr', 'vl', 'vb', 'ra', 'dec', 'pmra', 'pmdec', 'R', 'phi', 'vR', 'vphi']
        And, if `axisymmetric` is False, also ['vM', 'vm']
    
    info: .txt file
        Currently informs of the datatype and the columns.
    """

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