import numpy as  np

import src.compute_variables as CV
import src.bootstrap_errors as bootstrap
import src.montecarlo_errors as montecarlo

def check_all_maps_are_filled(map_dict, full_map_string_list):
    not_filled = []
    for map_string in full_map_string_list:
        if map_dict.get(map_string) is None:
            not_filled.append(map_string)
    
    if any(not_filled):
        raise ValueError("These maps were not filled:",not_filled)

def compute_fractional_errors(map_dict,full_map_string_list):
    for map_string in full_map_string_list:
        if "fractionalerror" in map_string:
            value_string = map_string.split("_fractionalerror")
            map_dict[map_string] = abs(map_dict[value_string[0]+"_error"+value_string[1]]/map_dict[value_string[0]])

def get_error(df,true_value,function,error_type,vel_x_var=None,vel_y_var=None,montecarloconfig=None,bootstrapconfig=None,tilt=False,absolute=False,R_hat=None):

    if error_type=="MC" and montecarloconfig is None:
        raise ValueError(f"Estimating MC errors but montecarloconfig was None.")
    if error_type=="bootstrap" and bootstrapconfig is None:
        raise ValueError(f"Estimating bootstrap errors but bootstrapconfig was None.")

    if vel_x_var is None and vel_y_var is None:
        raise ValueError("At least one of vel_x_var and vel_y_var must not be None.")

    if error_type == "bootstrap":

        vx,vy = None,None

        if vel_x_var is not None:
            vx = df["v"+vel_x_var].values
        if vel_y_var is not None:
            vy = df["v"+vel_y_var].values

        if bootstrapconfig.scipy:
            result = bootstrap.scipy_bootstrap(function=function,repeats=bootstrapconfig.repeats,vx=vx,vy=vy,tilt=tilt,absolute=absolute,R_hat=R_hat,CI_as_distance=True)
        else:
            result = bootstrap.get_std_bootstrap(function=function, vx=vx, vy=vy, tilt=tilt, absolute=absolute, R_hat=R_hat,config=bootstrapconfig)

        return result.confidence_interval
    
    elif error_type == "MC":
        std_low,std_high,*_ = montecarlo.get_std_MC(df=df, true_value=true_value,function=function, montecarloconfig=montecarloconfig, vel_x_var=vel_x_var, vel_y_var=vel_y_var, \
                                         tilt=tilt, absolute=absolute, R_hat=R_hat)
        
        return std_low,std_high
    
    else:
        raise ValueError(f"Error type `{error_type}` not recognised.")

def get_all_variable_values_and_errors(df_vals,vel_x_var,vel_y_var,full_map_string_list,df_errors=None,montecarloconfig=None,bootstrapconfig=None,\
                                       min_number=50, bin_surface=None, R_hat=None, x_var=None,y_var=None,error_type="bootstrap"):
    """
    Create a dictionary with the values of all the variables in full_map_string_list.

    Parameters
    ----------
    df_vals : pandas DataFrame
        Data, observational or simulated, to compute the variables for.
    vel_x_var : str
        Suffix of x-component of velocity.
    vel_y_var : str
        Suffix of y-component of velocity.
    full_map_string_list : list of str
        List of statistics to compute (e.g., 'mean_vx', 'std_vx', 'n_density').
    df_errors : pandas DataFrame, optional
        Data previous to applying any cuts that affect the Monte Carlo error calculation.
        Defaults to None, in which case df_vals is used.
    montecarloconfig : MonteCarloConfig, optional
        Configuration for Monte Carlo error estimation. See src/errorconfig.py
        Required if error_type is 'MC'. Defaults to None.
    bootstrapconfig : BootstrapConfig, optional
        Configuration for bootstrap error estimation. See src/errorconfig.py
        Required if error_type is 'bootstrap'. Defaults to None.
    min_number : int, optional
        Minimum number of stars required to compute statistics. Defaults to 50.
    bin_surface : float, optional
        Surface area of each bin, required to calculate the number density. 
        Required if 'n_density' is in full_map_string_list. Defaults to None.
    R_hat : tuple, optional
        Radial distance, required for spherical tilt calculations. Defaults to None.
    x_var : str, optional
        Horizontal position variable, required for spherical tilt calculations. Defaults to None.
    y_var : str, optional
        Vertical position variable, required for spherical tilt calculations. Defaults to None.
    error_type : str, optional
        Type of error estimation to use ('bootstrap' or 'MC'). Defaults to 'bootstrap'.

    Returns
    -------
    map_dict : dict
        A dictionary with the values of all the variables in full_map_string_list.
    """
    
    if "spherical_tilt" in full_map_string_list and (x_var is None or y_var is None or R_hat is None):
        raise ValueError("Cannot compute spherical_tilt without R_hat, x_var and y_var.")
    if "n_density" in full_map_string_list and not bin_surface:
        raise ValueError("Cannot compute n_density without bin_surface.")
    if error_type == "MC":
        if montecarloconfig is None:
            raise ValueError("The MC config needs to be passed when error_type is MC, but it was None.")
        elif df_errors is None:
            raise ValueError("df_errors needs to be passed when error_type is MC, but it was None.")
        elif len(df_errors) == len(df_vals) and np.all(df_errors == df_vals): # need the length check otherwise element-to-element comparison fails
            print("The dataframe for values and errors were identical.")
    elif error_type == "bootstrap":
        if bootstrapconfig is None:
            raise ValueError("The bootstrap config needs to be passed when error_type is bootstrap, but it was None.")
    if error_type != "MC" and df_errors is not None and np.any(df_errors != df_vals):
        raise ValueError(f"The dataframe for values and errors should be the same when estimating the latter using `{error_type}`.")

    if df_errors is None:
        df_errors = df_vals

    vx,vy = df_vals["v"+vel_x_var].values, df_vals["v"+vel_y_var].values

    map_dict = {}

    number = len(vx)

    if "number" in full_map_string_list:
        map_dict["number"] = number
    
    if "n_density" in full_map_string_list:
        n_density = number/bin_surface
        map_dict["n_density"] = n_density
    
    if number < min_number:
        for variable in full_map_string_list:
            if variable != "number" and variable != "n_density":
                map_dict[variable] = np.nan 
        return map_dict

    common_error_args = {
        "df": df_errors,
        "error_type": error_type,
        "montecarloconfig": montecarloconfig,
        "bootstrapconfig": bootstrapconfig
    }
    
    if "mean_vx" in full_map_string_list:
        map_dict["mean_vx"] = np.mean(vx)
        map_dict["mean_vx_error_low"],map_dict["mean_vx_error_high"] = get_error(true_value=map_dict["mean_vx"],function=np.mean,vel_x_var=vel_x_var,**common_error_args)
    
    if "mean_vy" in full_map_string_list:
        map_dict["mean_vy"] = np.mean(vy)
        map_dict["mean_vy_error_low"],map_dict["mean_vy_error_high"] = get_error(true_value=map_dict["mean_vy"],function=np.mean,vel_y_var=vel_y_var,**common_error_args)

    if "std_vx" in full_map_string_list:
        map_dict["std_vx"] = np.std(vx)
        map_dict["std_vx_error_low"],map_dict["std_vx_error_high"] = get_error(true_value=map_dict["std_vx"],function=np.std,vel_x_var=vel_x_var,**common_error_args)

    if "std_vy" in full_map_string_list:
        map_dict["std_vy"] = np.std(vy)
        map_dict["std_vy_error_low"],map_dict["std_vy_error_high"] = get_error(true_value=map_dict["std_vy"],function=np.std,vel_y_var=vel_y_var,**common_error_args)
    
    if "anisotropy" in full_map_string_list:
        map_dict["anisotropy"] = CV.calculate_anisotropy(vx=vx,vy=vy)
        map_dict["anisotropy_error_low"],map_dict["anisotropy_error_high"] = get_error(true_value=map_dict["anisotropy"],function=CV.calculate_anisotropy,vel_x_var=vel_x_var,vel_y_var=vel_y_var,**common_error_args)

    if "correlation" in full_map_string_list:
        map_dict["correlation"] = CV.calculate_correlation(vx=vx,vy=vy)
        map_dict["correlation_error_low"],map_dict["correlation_error_high"] = get_error(true_value=map_dict["correlation"],function=CV.calculate_correlation,vel_x_var=vel_x_var,vel_y_var=vel_y_var,**common_error_args)
    
    if "tilt_abs" in full_map_string_list:
        map_dict["tilt_abs"] = CV.calculate_tilt(vx=vx,vy=vy,absolute=True)
        map_dict["tilt_abs_error_low"],map_dict["tilt_abs_error_high"] = get_error(true_value=map_dict["tilt_abs"],function=CV.calculate_tilt,vel_x_var=vel_x_var,vel_y_var=vel_y_var,tilt=True,absolute=True,**common_error_args)
    
    if "spherical_tilt" in full_map_string_list:
        vx_s,vy_s = df_vals["v"+x_var].values, df_vals["v"+y_var].values

        map_dict["spherical_tilt"] = CV.calculate_spherical_tilt(vx_s,vy_s,R_hat,absolute=False)
        map_dict["spherical_tilt_error_low"],map_dict["spherical_tilt_error_high"] =\
              get_error(true_value=map_dict["spherical_tilt"],function=CV.calculate_spherical_tilt,vel_x_var=x_var,vel_y_var=y_var,tilt=True,absolute=False,R_hat=R_hat, **common_error_args)

    if "abs_spherical_tilt" in full_map_string_list:
        map_dict["abs_spherical_tilt"] = abs(map_dict["spherical_tilt"])
        map_dict["abs_spherical_tilt_error_low"],map_dict["abs_spherical_tilt_error_high"] =\
              get_error(true_value=map_dict["abs_spherical_tilt"],function=CV.calculate_spherical_tilt,vel_x_var=x_var,vel_y_var=y_var,absolute=True,R_hat=R_hat, **common_error_args)

    if "tilt" in full_map_string_list:
        map_dict["tilt"] = CV.calculate_tilt(vx=vx,vy=vy,absolute=False)
        map_dict["tilt_error"] = get_error(true_value=map_dict["tilt"],function=CV.calculate_tilt, vel_x_var=vel_x_var,vel_y_var=vel_y_var, tilt=True,absolute=False, **common_error_args)        

    compute_fractional_errors(map_dict,full_map_string_list)

    check_all_maps_are_filled(map_dict,full_map_string_list)
    
    return map_dict