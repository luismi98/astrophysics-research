import numpy as np
import pandas as pd
import os
import coordinates

R0 = coordinates.get_solar_radius()

def load_simulation(path, choice="708main", rot_angle = 27, R0=R0, axisymmetric=False, zabs=True, pos_factor=1.7, GSR=True):
    
    if choice == "708main":

        if not os.path.isdir(path):
            raise FileNotFoundError("Could not find directory:", path)

        bar_string = '' if axisymmetric else f'_bar{rot_angle}'
        scaling_string = "_scale" + str(pos_factor)
        frame_string = '' if GSR else '_LSR'
        zabs_string = '' if zabs else '_noZabs'
        axi_string = '_axisymmetric' if axisymmetric else ''    
        R0_string = f'_{R0}R0'

        filename = f"708MWout{bar_string}{scaling_string}{R0_string}{frame_string}{zabs_string}{axi_string}.npy"

        df = pd.DataFrame(np.load(path + filename))
        columns = np.load(path + "columns.npy")

        if (len(columns) == df.shape[1]):
            df.columns = columns
        else:
            raise ValueError("The number of columns to set did not have the right size!")
        
        print(filename,"loaded successfully.")

    elif choice == "stuart":
        df = pd.DataFrame(np.load(path+"D60_stars_stuart.npy"))

        df.loc[df.l > 180, 'l'] -= 360
    
        print("Stuart's simulation loaded successfully.")
        
    return df