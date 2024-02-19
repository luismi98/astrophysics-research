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

def get_all_variable_values_and_errors(vx,vy, full_map_string_list, repeat=500, min_number=50, all_maps=False, bin_surface=None, R_hat=None, vx_s=None,vy_s=None):
    
    if "spherical_tilt" in full_map_string_list and (vx_s is None or vy_s is None or R_hat is None):
        raise ValueError("Cannot compute spherical_tilt without having R_hat and velocities vx_s,vy_s")
    
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
    
    map_dict["mean_vx"] = np.mean(vx)
    map_dict["mean_vy"] = np.mean(vy)
    map_dict["mean_vx_error"] = CV.get_std_bootstrap(np.mean,vx,repeat=repeat)
    map_dict["mean_vy_error"] = CV.get_std_bootstrap(np.mean,vy,repeat=repeat)
    map_dict["std_vx"] = np.sqrt(varx)
    map_dict["std_vy"] = np.sqrt(vary)
    map_dict["std_vx_error"] = CV.get_std_bootstrap(np.std,vx,repeat=repeat)
    map_dict["std_vy_error"] = CV.get_std_bootstrap(np.std,vy,repeat=repeat)
    map_dict["anisotropy"] = 1-vary/varx
    map_dict["anisotropy_error"] = CV.get_std_bootstrap(CV.calculate_anisotropy,vx,vy,repeat=repeat)
    map_dict["correlation"] = covxy/np.sqrt(varx*vary)
    map_dict["correlation_error"] = CV.get_std_bootstrap(CV.calculate_correlation,vx,vy,repeat=repeat)
    map_dict["tilt_abs"] = np.degrees(np.arctan2(2.*covxy,abs(varx - vary))/2.)
    map_dict["tilt_abs_error"] = CV.get_std_bootstrap(CV.calculate_tilt, vx, vy, repeat=repeat, tilt=True, absolute=True)
    
    if R_hat is not None:
        spherical_tilt = CV.calculate_spherical_tilt(vx_s,vy_s,R_hat,absolute=False)
        spherical_tilt_error = CV.get_std_bootstrap(CV.calculate_spherical_tilt,vx_s,vy_s,tilt=True,absolute=False,R_hat=R_hat, repeat=repeat)

        map_dict["spherical_tilt"] = spherical_tilt
        map_dict["spherical_tilt_error"] = spherical_tilt_error

    if not all_maps:
        compute_fractional_errors(map_dict,full_map_string_list)

        if not all_maps_are_filled(map_dict,full_map_string_list):
            raise ValueError("Some of the maps in `full_map_string_list` were not filled!")
        return map_dict
    
    map_dict["tilt"] = np.degrees(np.arctan2(2.*covxy,varx - vary)/2.)
    map_dict["tilt_error"] = CV.get_std_bootstrap(CV.calculate_tilt, vx, vy, repeat=repeat, tilt=True)

    if R_hat is not None:
        abs_spherical_tilt = abs(spherical_tilt)
        abs_spherical_tilt_error = CV.get_std_bootstrap(CV.calculate_spherical_tilt,vx_s,vy_s,absolute=True,R_hat=R_hat,repeat=repeat)

        map_dict["abs_spherical_tilt"] = abs_spherical_tilt
        map_dict["abs_spherical_tilt_error"] = abs_spherical_tilt_error

    compute_fractional_errors(map_dict,full_map_string_list)

    if not all_maps_are_filled(map_dict,full_map_string_list):
        raise ValueError("Some of the maps in `full_map_string_list` were not filled!")
    
    return map_dict