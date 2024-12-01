import numpy as np
import pandas as pd
import os
import utils.coordinates as coordinates

Z0_CONST = coordinates.get_solar_height()
R0_CONST = coordinates.get_solar_radius()

def build_filename(choice="708main",rot_angle=27,R0=R0_CONST,Z0=Z0_CONST,axisymmetric=False,zabs=True,pos_factor=1.7,GSR=True):
    bar_string = '' if axisymmetric else f'_bar{rot_angle}'
    scaling_string = "_scale" + str(pos_factor)
    frame_string = '' if GSR else '_LSR'
    zabs_string = '' if zabs else '_noZabs'
    axi_string = '_axisymmetric' if axisymmetric else ''    
    R0_string = f'_{R0}R0'
    Z0_string = f"_{Z0}Z0" if Z0 != Z0_CONST else ''

    alert_of_unusual_loading_config(choice=choice, rot_angle=rot_angle, R0=R0, Z0=Z0, axisymmetric=axisymmetric, zabs=zabs, pos_factor=pos_factor, GSR=GSR)

    return f"{choice}_MWout{bar_string}{scaling_string}{R0_string}{Z0_string}{frame_string}{zabs_string}{axi_string}"

def load_simulation(path, filename):

    if not os.path.isdir(path):
        raise FileNotFoundError("Could not find directory:", path)

    df = pd.DataFrame(np.load(path + filename + ".npy"))
    columns = np.load(path + filename + "_columns.npy")

    if len(columns) == df.shape[1]:
        df.columns = columns
    else:
        raise ValueError("The number of columns to set did not have the right size!")
    
    print(filename,"loaded successfully.")

    return df

def alert_of_unusual_loading_config(**params):

    if "sim_choice" in params:
        if params["sim_choice"] != "708main":
            print(f"Warning: not working with 708main")

    if "zabs" in params:
        if not params["zabs"]:
            print("Warning: using zabs = False")
    
    if "R0" in params:
        if params["R0"] != coordinates.get_solar_radius():
            print(f"Warning: R0 is not the usual {coordinates.get_solar_radius()}")
    
    if "Z0" in params:
        if params["Z0"] != coordinates.get_solar_height():
            print(f"Warning: Z0 is not the usual {coordinates.get_solar_height()}")
    
    if "GSR" in params:
        if not params["GSR"]:
            print("Warning: using GSR = False")

    if "rot_angle" in params:
        if params["rot_angle"] != coordinates.get_bar_angle():
            print(f"Warning: rot_angle != {coordinates.get_bar_angle()}")
    
    if "axisymmetric" in params:
        if params["axisymmetric"]:
            print(f"Warning: axisymmetric = True")
    
    if "pos_factor" in params:
        if params["pos_factor"] != 1.7:
            print(f"Warning: pos_factor is not 1.7")