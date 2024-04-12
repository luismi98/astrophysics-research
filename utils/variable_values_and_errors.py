import numpy as  np
import compute_variables as CV

def all_maps_are_filled(map_dict, full_map_string_list):
    for map_string in full_map_string_list:
        if not np.any(map_dict[map_string]):
            return False
    return True

def compute_fractional_errors(map_dict,full_map_string_list):
    for map_string in full_map_string_list:
        if "fractionalerror" in map_string:
            value_string = map_string.split("_fractionalerror")[0]
            map_dict[map_string] = abs(map_dict[value_string+"_error"]/map_dict[value_string])

def get_error(df,function,vel_x_var=None,vel_y_var=None,tilt=False, absolute=False, R_hat = None, repeat = 500, type="bootstrap", Rmax_MC=3.5):
    if type == "bootstrap":
        if vel_x_var is None:
            raise ValueError("vel_x_var cannot be None if computing the bootstrapping error.")

        vx = df[vel_x_var].values
        if vel_y_var is not None:
            vy = df[vel_y_var].values
        return CV.get_std_bootstrap(function=function, vx=vx, vy=vy, tilt=tilt, absolute=absolute, R_hat = R_hat, size_fraction = 1, repeat = repeat)
    else:
        return CV.get_std_MC(df=df,function=function,tilt=tilt, absolute=absolute, R_hat = R_hat, repeat = repeat, Rmax=Rmax_MC)

def get_all_variable_values_and_errors(df,vel_x_var,vel_y_var,full_map_string_list,repeat=500, min_number=50, all_maps=False, bin_surface=None, R_hat=None,\
                                        x_var=None,y_var=None,error_type="bootstrap",Rmax_MC=3.5):
    
    if "spherical_tilt" in full_map_string_list and (x_var is None or y_var is None or R_hat is None):
        raise ValueError("Cannot compute spherical_tilt without having R_hat, x_var and y_var.")
    
    vx,vy = df["v"+vel_x_var].values, df["v"+vel_y_var].values

    number = len(vx)
    map_dict = {
        "number": int(number)
    }
    
    if bin_surface is not None:
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
        "df": df,
        "repeat": repeat,
        "type": error_type,
        "Rmax_MC": Rmax_MC
    }
    
    map_dict["mean_vx"] = np.mean(vx)
    map_dict["mean_vy"] = np.mean(vy)
    map_dict["mean_vx_error"] = get_error(function=np.mean,vel_x_var=vel_x_var,**common_error_args)
    map_dict["mean_vy_error"] = get_error(function=np.mean,vel_y_var=vel_y_var,**common_error_args)
    map_dict["std_vx"] = np.sqrt(varx)
    map_dict["std_vy"] = np.sqrt(vary)
    map_dict["std_vx_error"] = get_error(function=np.std,vel_x_var=vel_x_var,**common_error_args)
    map_dict["std_vy_error"] = get_error(function=np.std,vel_y_var=vel_y_var,**common_error_args)
    map_dict["anisotropy"] = 1-vary/varx
    map_dict["anisotropy_error"] = get_error(function=CV.calculate_anisotropy,vel_x_var=vel_x_var,vel_y_var=vel_y_var,**common_error_args)
    map_dict["correlation"] = covxy/np.sqrt(varx*vary)
    map_dict["correlation_error"] = get_error(function=CV.calculate_correlation,vel_x_var=vel_x_var,vel_y_var=vel_y_var,**common_error_args)
    map_dict["tilt_abs"] = np.degrees(np.arctan2(2.*covxy,abs(varx - vary))/2.)
    map_dict["tilt_abs_error"] = get_error(function=CV.calculate_tilt,vel_x_var=vel_x_var,vel_y_var=vel_y_var,tilt=True,absolute=True,**common_error_args)
    
    if R_hat is not None:
        vx_s,vy_s = df["v"+x_var].values, df["v"+y_var].values

        spherical_tilt = CV.calculate_spherical_tilt(vx_s,vy_s,R_hat,absolute=False)
        spherical_tilt_error = get_error(function=CV.calculate_spherical_tilt,vel_x_var=x_var,vel_y_var=y_var,tilt=True,absolute=False,R_hat=R_hat, **common_error_args)

        map_dict["spherical_tilt"] = spherical_tilt
        map_dict["spherical_tilt_error"] = spherical_tilt_error

    if not all_maps:
        compute_fractional_errors(map_dict,full_map_string_list) # needs to be computed once all values and errors are calculated, as they are used

        if not all_maps_are_filled(map_dict,full_map_string_list):
            raise ValueError("Some of the maps in `full_map_string_list` were not filled!")
        return map_dict
    
    map_dict["tilt"] = np.degrees(np.arctan2(2.*covxy,varx - vary)/2.)
    map_dict["tilt_error"] = get_error(function=CV.calculate_tilt, vel_x_var=vel_x_var,vel_y_var=vel_y_var, tilt=True,absolute=False, **common_error_args)

    if R_hat is not None:
        abs_spherical_tilt = abs(spherical_tilt)
        abs_spherical_tilt_error = get_error(function=CV.calculate_spherical_tilt,vel_x_var=x_var,vel_y_var=y_var,absolute=True,R_hat=R_hat, **common_error_args)

        map_dict["abs_spherical_tilt"] = abs_spherical_tilt
        map_dict["abs_spherical_tilt_error"] = abs_spherical_tilt_error

    compute_fractional_errors(map_dict,full_map_string_list)

    if not all_maps_are_filled(map_dict,full_map_string_list):
        raise ValueError("Some of the maps in `full_map_string_list` were not filled!")
    
    return map_dict