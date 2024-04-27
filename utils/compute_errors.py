import pandas as pd
import numpy as np

from velocity_plot import velocity_plot
import coordinates
import miscellaneous_functions as MF

def apply_function(function, vx, vy, R_hat, tilt, absolute):
    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")

    if R_hat is None:
        if tilt:
            return function(vx,vy,absolute)
        else:
            return function(vx,vy) if vx is not None and vy is not None else function(vx if vx is not None else vy)
    else:
        return function(vx,vy,R_hat,absolute=absolute)

def validate_MC_method(expected_perturbed_var,df,montecarloconfig,vel_x_var,vel_y_var,n_expected_affected_cuts,expected_vel_x_var=None,expected_vel_y_var=None):
    if vel_x_var is None and vel_y_var is None:
        raise ValueError("Both velocity components were set to None!")
    if montecarloconfig.perturbed_var != expected_perturbed_var:
        raise ValueError(f"Expected to perturb `{expected_perturbed_var}` but the config was set to `{montecarloconfig.perturbed_var}`.")
    if vel_x_var is not None and expected_vel_x_var is not None and vel_x_var != expected_vel_x_var:
        raise ValueError(f"Expected the x velocity component to be `{expected_vel_x_var}` but it was `{vel_x_var}`.")
    if vel_y_var is not None and expected_vel_y_var is not None and vel_y_var != expected_vel_y_var:
        raise ValueError(f"Expected the y velocity component to be `{expected_vel_y_var}` but it was `{vel_y_var}`.")
    if len(montecarloconfig.affected_cuts_dict) != n_expected_affected_cuts:
        raise ValueError(f"Expected to find `{n_expected_affected_cuts}` cuts affected by the MC, but found `{len(montecarloconfig.affected_cuts_dict)}`.")
    if f"{expected_perturbed_var}_error" not in df and montecarloconfig.error_frac is None:
        raise ValueError(f"{expected_perturbed_var}_error was not found in the dataframe and montecarloconfig.error_frac is None. Please specify the errors.")

def apply_MC_distance(df,vel_x_var,vel_y_var,montecarloconfig):
    validate_MC_method("d",df,montecarloconfig,vel_x_var,vel_y_var,n_expected_affected_cuts=1,expected_vel_x_var="r",expected_vel_y_var="l")

    vr = df["vr"] if vel_x_var is not None else None
    vl = df["vl"] if vel_y_var is not None else None

    d_error = montecarloconfig.error_frac * df["d"] if montecarloconfig.error_frac is not None else df["d_error"]

    helper_df = pd.DataFrame(df[["l","b"]])

    MC_d = df["d"] + np.random.normal(scale=d_error)
    helper_df["d"] = MC_d
    coordinates.lbd_to_xyz(helper_df)

    cut_var = list(montecarloconfig.affected_cuts_dict.keys())[0]

    within_cut = MF.build_lessgtr_condition(array=np.hypot(helper_df["x"],helper_df["y"]),\
                                             low=montecarloconfig.affected_cuts_dict[cut_var][0],\
                                             high=montecarloconfig.affected_cuts_dict[cut_var][1],\
                                             include=montecarloconfig.affected_cuts_lims_dict[cut_var])
    
    MC_vr, MC_vl = None, None

    if vr is not None:
        MC_vr = vr[within_cut]
    if vl is not None:
        pmlcosb = vl / df["d"] # km/s/kpc

        MC_d,MC_pmlcosb = MC_d[within_cut], pmlcosb[within_cut]
        MC_vl = MC_pmlcosb*MC_d

    return MC_vr, MC_vl,within_cut

def apply_MC_vr(df,vel_x_var,vel_y_var,montecarloconfig):
    validate_MC_method("vr",df,montecarloconfig,vel_x_var,vel_y_var,n_expected_affected_cuts=0,expected_vel_x_var="r",expected_vel_y_var="l")

    if vel_x_var is None:
        return df["vl"] if vel_y_var is not None else None, None

    vr_error = montecarloconfig.error_frac * np.abs(df["vr"]) if montecarloconfig.error_frac is not None else df["vr_error"]
    MC_vr = df["vr"] + np.random.normal(scale=vr_error)

    return MC_vr, df["vl"] if vel_y_var is not None else None
    
def apply_MC_pmra(df,vel_x_var,vel_y_var,montecarloconfig):
    validate_MC_method("pmra",df,montecarloconfig,vel_x_var,vel_y_var,n_expected_affected_cuts=0,expected_vel_x_var="r",expected_vel_y_var="l")

    if vel_y_var is None:
        return df["vr"] if vel_x_var is not None else None, None

    df_helper = pd.DataFrame(df[["l","b"]])
    df_helper["pmlcosb"] = df["vl"]/df["d"]
    df_helper["pmb"] = df["vb"]/df["d"]

    for pm in ["pmlcosb", "pmb"]:
        df_helper[pm] *= coordinates.get_conversion_factor_from_mas_per_yr_to_rad_per_s(inverse=True)
        df_helper[pm] /= coordinates.get_conversion_factor_from_kpc_to_km()

    coordinates.pmlpmb_to_pmrapmdec(df_helper)

    if "pmdec" in df and (df["pmdec"] != df_helper["pmdec"]).all():
        print(df["pmdec"])
        print(df_helper["pmdec"])
        raise ValueError("The computed pmdec is different than the one already existing in the df!")
    if "pmra" in df and (df["pmra"] != df_helper["pmra"]).all():
        print(df["pmra"])
        print(df_helper["pmra"])
        raise ValueError("The computed pmra is different than the one already existing in the df!")

    pmra_error = montecarloconfig.error_frac * df_helper["pmra"] if montecarloconfig.error_frac is not None else df["pmra_error"]

    df_helper["pmra"] += np.random.normal(scale=pmra_error)

    coordinates.pmrapmdec_to_pmlpmb(df_helper)

    MC_vl = coordinates.convert_pm_to_velocity(df["d"], df_helper["pmlcosb"], kpc_bool=False, masyr_bool=True)

    return df["vr"] if vel_x_var is not None else None, MC_vl

def compute_lowhigh_std(true_value, perturbed_values):
    values_above = perturbed_values[perturbed_values > true_value]
    values_below = perturbed_values[perturbed_values < true_value]
    values_equal = perturbed_values[perturbed_values == true_value]

    # divide perturbed values which result equal to the true value proportionally between the above and below values
    frac_equal_to_above = len(values_above) / (len(values_above) + len(values_below))
    idx_equal_to_above = int( len(values_equal) * frac_equal_to_above )

    values_above = np.append(values_above, values_equal[:idx_equal_to_above])
    values_below = np.append(values_below, values_equal[idx_equal_to_above:])

    std_low = np.sqrt(np.mean((values_below - true_value)**2)) if len(values_below) > 0 else 0
    std_high = np.sqrt(np.mean((values_above - true_value)**2)) if len(values_above) > 0 else 0

    return std_low,std_high

def get_std_MC(df,true_value,function,montecarloconfig,vel_x_var=None,vel_y_var=None,tilt=False, absolute=False, R_hat=None, show_vel_plots=False, show_freq=10, velocity_kws={}):

    if vel_x_var is None and vel_y_var is None:
        raise ValueError("Both velocity components were set to None!")
    if montecarloconfig is None:
        raise ValueError("montecarloconfig cannot be None when estimating MC uncertainties.")
    
    within_cut = None
    MC_values = np.empty(shape=(montecarloconfig.repeats))

    if montecarloconfig.random_resampling_indices is not None:
        df_resampled = df.loc[montecarloconfig.random_resampling_indices]

    for i in range(montecarloconfig.repeats):
        
        if montecarloconfig.perturbed_var == "d":
            
            if (vel_x_var is not None and vel_x_var != "r") or (vel_y_var is not None and vel_y_var != "l"):
                raise ValueError("Expected the velocity components to be vr and/or vl when estimating the MC errors with distance uncertainties.")

            if montecarloconfig.random_resampling_indices is not None:
                MC_resampled_vx,MC_resampled_vy,within_cut = apply_MC_distance(df_resampled,vel_x_var,vel_y_var,montecarloconfig)

                extra_N = len(df_resampled) - len(MC_resampled_vx if vel_x_var is not None else MC_resampled_vy)
                MC_extra_vx = pd.Series() if vel_x_var is not None else None
                MC_extra_vy = pd.Series() if vel_y_var is not None else None

                if extra_N > 0: # some of the resampled stars have fallen beyond the affected cut - let's take some extra ones

                    while len(MC_extra_vx if vel_x_var is not None else MC_extra_vy) < extra_N:
                        extra_indices = pd.Index(np.random.choice(df.index, size=10*extra_N, replace=False))\
                                        .difference(montecarloconfig.random_resampling_indices)\
                                        .difference(MC_extra_vx.index if vel_x_var is not None else MC_extra_vy.index)

                        if len(extra_indices) == 0:
                            continue
                        
                        ex_vx,ex_vy,_ = apply_MC_distance(df.loc[extra_indices],vel_x_var,vel_y_var,montecarloconfig)

                        MC_extra_vx = pd.concat([MC_extra_vx, ex_vx]) if vel_x_var is not None else None
                        MC_extra_vy = pd.concat([MC_extra_vy, ex_vy]) if vel_y_var is not None else None

                MC_vx = pd.concat([MC_resampled_vx, MC_extra_vx[:extra_N]]) if vel_x_var is not None else None
                MC_vy = pd.concat([MC_resampled_vy, MC_extra_vy[:extra_N]]) if vel_y_var is not None else None

            else:
                MC_vx,MC_vy,within_cut = apply_MC_distance(df,vel_x_var,vel_y_var,montecarloconfig)

        elif montecarloconfig.perturbed_var == "vr":
            MC_vx,MC_vy = apply_MC_vr(df if montecarloconfig.random_resampling_indices is None else df_resampled,\
                                                       vel_x_var,vel_y_var,montecarloconfig)
        elif montecarloconfig.perturbed_var == "pmra":
            MC_vx,MC_vy = apply_MC_pmra(df if montecarloconfig.random_resampling_indices is None else df_resampled,\
                                                       vel_x_var,vel_y_var,montecarloconfig)
        else:
            raise ValueError(f"MC behaviour undefined for perturbed variable `{montecarloconfig.perturbed_var}`.")

        if show_vel_plots and i%show_freq == 0:
            velocity_plot(MC_vx,MC_vy,**velocity_kws)

        MC_values[i] = apply_function(function,MC_vx,MC_vy,R_hat,tilt,absolute)

    if tilt and not absolute:
        MC_values[(true_value - MC_values)>90] += 180
        MC_values[(true_value - MC_values)<-90] -= 180

    std_low, std_high = compute_lowhigh_std(true_value=true_value,perturbed_values=MC_values)
    
    return std_low,std_high,MC_values,within_cut

def get_std_bootstrap(function,bootstrapconfig,vx=None,vy=None,tilt=False,absolute=False,R_hat=None,show_vel_plots=False,show_freq=10,velocity_kws={}):
    '''
    Computes the standard deviation of a function applied to velocity components using bootstrap resampling.

    Parameters
    ----------
    vx, vy: array-like
        Arrays of desired velocity components. 
        vy=None if only 1 is needed.

    function: callable
        Function used to compute the desired variable whose error you want to estimate.

    tilt: bool, optional
        Boolean variable indicating whether the error is being estimated on a tilt (including spherical tilt). 
        Default is False.

    absolute: bool, optional
        Only has effect if tilt=True. Determines whether the tilt is being computed with absolute value in the denominator or not. 
        Default is False.

    R_hat: array-like or None, optional
        Only has effect if tilt=True. If None, the tilt is a vertex deviation, otherwise it is a spherical tilt. 
        Default is None.

    show_vel_plots: bool, optional
        If True, velocity plots are shown. 
        Default is False.

    show_freq: int, optional
        Frequency of showing velocity plots. 
        Default is 10.

    velocity_kws: dict, optional
        Keyword arguments for the `velocity_plot` function. 
        Default is an empty dictionary.

    Returns
    -------
    std: float
        The estimated standard deviation.

    boot_values: array-like
        The bootstrap values.
    '''
    
    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")

    true_value = apply_function(function,vx,vy,R_hat,tilt,absolute)

    original_size = len(vx) if vx is not None else len(vy)
    indices_range = np.arange(original_size)

    bootstrap_size = original_size if bootstrapconfig.bootstrap_size is None else bootstrapconfig.bootstrap_size
    
    boot_values = np.empty(shape=(bootstrapconfig.repeats))

    for i in range(bootstrapconfig.repeats):
        bootstrap_indices = np.random.choice(indices_range, size = bootstrap_size, replace = bootstrapconfig.replacement)
        
        boot_vx,boot_vy = None,None

        if vx is not None:
            boot_vx = vx[bootstrap_indices]
        if vy is not None:
            boot_vy = vy[bootstrap_indices]
        
        if show_vel_plots and i%show_freq == 0 and boot_vx is not None and boot_vy is not None:
            velocity_plot(boot_vx,boot_vy,**velocity_kws)
            
        boot_values[i] = apply_function(function,boot_vx,boot_vy,R_hat,tilt,absolute)
    
    if tilt and not absolute:
        boot_values[(true_value - boot_values)>90] += 180
        boot_values[(true_value - boot_values)<-90] -= 180
    
    #Note this is a pseudo standard deviation: relative to true value as opposed to mean of bootstrap values
    if bootstrapconfig.symmetric:
        std = np.sqrt(np.nanmean((boot_values-true_value)**2))
        std_low,std_high = std,std
    else:
        std_low, std_high = compute_lowhigh_std(true_value=true_value,perturbed_values=boot_values)
    
    return std_low, std_high, boot_values