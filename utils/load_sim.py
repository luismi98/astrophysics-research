import numpy as np
import pandas as pd
import os
import coordinates

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

    return f"{choice}_MWout{bar_string}{scaling_string}{R0_string}{Z0_string}{frame_string}{zabs_string}{axi_string}.npy"

def load_simulation(path, filename):

    if not os.path.isdir(path):
        raise FileNotFoundError("Could not find directory:", path)

    df = pd.DataFrame(np.load(path + filename))
    columns = np.load(path + filename + "_columns.npy")

    if (len(columns) == df.shape[1]):
        df.columns = columns
    else:
        raise ValueError("The number of columns to set did not have the right size!")
    
    print(filename,"loaded successfully.")

    return df