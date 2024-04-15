import numpy as  np
import warnings

import compute_variables as CV

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

def get_error(df,function,vel_x_var=None,vel_y_var=None,montecarloconfig=None,tilt=False, absolute=False, R_hat = None, repeat = 500, type="bootstrap"):

    if type=="MC" and montecarloconfig is None:
        raise ValueError(f"Estimating MC errors but montecarloconfig was None.")
    elif type != "MC" and montecarloconfig is not None:
        raise ValueError(f"Estimating `{type}` errors but montecarloconfig was not None.")

    if vel_x_var is None and vel_y_var is None:
        raise ValueError("At least one of vel_x_var and vel_y_var must not be None.")

    if type == "bootstrap":

        vx = df["v"+vel_x_var].values
        if vel_y_var is not None:
            vy = df["v"+vel_y_var].values

        std,_ = CV.get_std_bootstrap(function=function, vx=vx, vy=vy, tilt=tilt, absolute=absolute, R_hat = R_hat, size_fraction = 1, repeat = repeat)

        return std,std # symmetric low/high y-errors
    
    elif type == "MC":
        std_low,std_high,*_ = CV.get_std_MC(df=df, function=function, montecarloconfig=montecarloconfig, vel_x_var=vel_x_var, vel_y_var=vel_y_var, \
                                         tilt=tilt, absolute=absolute, R_hat = R_hat, repeat = repeat)
        
        return std_low,std_high
    
    else:
        raise ValueError(f"Error type `{type}` not recognised.")

def get_all_variable_values_and_errors(df_vals,vel_x_var,vel_y_var,full_map_string_list,df_errors=None,montecarloconfig=None,\
                                       repeat=500, min_number=50, bin_surface=None, R_hat=None, x_var=None,y_var=None,error_type="bootstrap"):
    
    if "spherical_tilt" in full_map_string_list and (x_var is None or y_var is None or R_hat is None):
        raise ValueError("Cannot compute spherical_tilt without R_hat, x_var and y_var.")
    if "n_density" in full_map_string_list and not bin_surface:
        raise ValueError("Cannot compute n_density without bin_surface.")
    if error_type == "MC":
        if montecarloconfig is None:
            raise ValueError("The MC config needs to be passed when error_type is MC, but it was None.")
        if df_errors is None or (len(df_errors) == len(df_vals) and np.all(df_errors == df_vals)): # need the length check otherwise element-to-element comparison fails
            warnings.warn("Are you sure you want to use the same dataframe for values and errors when estimating the latter using MC?")
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
        print(f"Found `{number}` stars, less than the min of `{min_number}`")
        for variable in full_map_string_list:
            if variable != "number" and variable != "n_density":
                map_dict[variable] = np.nan 
        return map_dict

    cov = np.cov(vx,vy)
    varx = cov[0,0]
    vary = cov[1,1]
    covxy = cov[0,1]

    common_error_args = {
        "df": df_errors,
        "repeat": repeat,
        "type": error_type,
        "montecarloconfig": montecarloconfig,
    }
    
    if "mean_vx" in full_map_string_list:
        map_dict["mean_vx"] = np.mean(vx)
        map_dict["mean_vx_error_low"],map_dict["mean_vx_error_high"] = get_error(function=np.mean,vel_x_var=vel_x_var,**common_error_args)

        if error_type == "MC" and montecarloconfig.affected_var == "d" and vel_x_var=="r":
            print(f"Computed MC distance errors for mean_vx, but note the perturbation affects vr only indirectly (from stars moving across the radial limit).")
    
    if "mean_vy" in full_map_string_list:
        map_dict["mean_vy"] = np.mean(vy)
        map_dict["mean_vy_error_low"],map_dict["mean_vy_error_high"] = get_error(function=np.mean,vel_y_var=vel_y_var,**common_error_args)

    if "std_vx" in full_map_string_list:
        map_dict["std_vx"] = np.sqrt(varx)
        map_dict["std_vx_error_low"],map_dict["std_vx_error_high"] = get_error(function=np.std,vel_x_var=vel_x_var,**common_error_args)

        if error_type == "MC" and montecarloconfig.affected_var == "d" and vel_x_var=="r":
            print(f"Computed MC distance errors for std_vx, but note the perturbation affects vr only indirectly (from stars moving across the radial limit).")

    if "std_vy" in full_map_string_list:
        map_dict["std_vy"] = np.sqrt(vary)
        map_dict["std_vy_error_low"],map_dict["std_vy_error_high"] = get_error(function=np.std,vel_y_var=vel_y_var,**common_error_args)
    
    if "anisotropy" in full_map_string_list:
        map_dict["anisotropy"] = 1-vary/varx
        map_dict["anisotropy_error_low"],map_dict["anisotropy_error_high"] = get_error(function=CV.calculate_anisotropy,vel_x_var=vel_x_var,vel_y_var=vel_y_var,**common_error_args)

    if "correlation" in full_map_string_list:
        map_dict["correlation"] = covxy/np.sqrt(varx*vary)
        map_dict["correlation_error_low"],map_dict["correlation_error_high"] = get_error(function=CV.calculate_correlation,vel_x_var=vel_x_var,vel_y_var=vel_y_var,**common_error_args)
    
    if "tilt_abs" in full_map_string_list:
        map_dict["tilt_abs"] = np.degrees(np.arctan2(2.*covxy,abs(varx - vary))/2.)
        map_dict["tilt_abs_error_low"],map_dict["tilt_abs_error_high"] = get_error(function=CV.calculate_tilt,vel_x_var=vel_x_var,vel_y_var=vel_y_var,tilt=True,absolute=True,**common_error_args)
    
    if "spherical_tilt" in full_map_string_list:
        vx_s,vy_s = df_vals["v"+x_var].values, df_vals["v"+y_var].values

        map_dict["spherical_tilt"] = CV.calculate_spherical_tilt(vx_s,vy_s,R_hat,absolute=False)
        map_dict["spherical_tilt_error_low"],map_dict["spherical_tilt_error_high"] =\
              get_error(function=CV.calculate_spherical_tilt,vel_x_var=x_var,vel_y_var=y_var,tilt=True,absolute=False,R_hat=R_hat, **common_error_args)

    if "abs_spherical_tilt" in full_map_string_list:
        map_dict["abs_spherical_tilt"] = abs(map_dict["spherical_tilt"])
        map_dict["abs_spherical_tilt_error_low"],map_dict["abs_spherical_tilt_error_high"] =\
              get_error(function=CV.calculate_spherical_tilt,vel_x_var=x_var,vel_y_var=y_var,absolute=True,R_hat=R_hat, **common_error_args)

    if "tilt" in full_map_string_list:
        map_dict["tilt"] = np.degrees(np.arctan2(2.*covxy,varx - vary)/2.)
        map_dict["tilt_error"] = get_error(function=CV.calculate_tilt, vel_x_var=vel_x_var,vel_y_var=vel_y_var, tilt=True,absolute=False, **common_error_args)        

    compute_fractional_errors(map_dict,full_map_string_list)

    check_all_maps_are_filled(map_dict,full_map_string_list)
    
    return map_dict