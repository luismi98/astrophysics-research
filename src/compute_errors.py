import numpy as np
import copy

from plotting.velocity_plot import velocity_plot
import utils.coordinates as coordinates
import utils.miscellaneous_functions as MF

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

def apply_MC(df, var, error_frac):
    if error_frac is None and var+"_error" not in df:
        raise ValueError(f"`{var}_error` was not found in the dataframe and error_frac is None. Please specify the errors.")

    error = error_frac * np.abs(df[var]) if error_frac is not None else df[var+"_error"]

    df[var] += np.random.normal(scale=error)

def build_within_cut_boolean_array(df, affected_cuts_dict, affected_cuts_lims_dict):
    if len(affected_cuts_dict) == 0:
        return None
    if len(affected_cuts_dict) > 1:
        raise ValueError(f"Expected a single spatial cut to be affected but found `{len(affected_cuts_dict)}`, namely `{affected_cuts_dict}`.")

    if "R" in affected_cuts_dict:
        coordinates.lbd_to_xyz(df)

        return MF.build_lessgtr_condition(array=np.hypot(df["x"],df["y"]), low=affected_cuts_dict["R"][0], high=affected_cuts_dict["R"][1],
                                          include=affected_cuts_lims_dict["R"])
    
    elif "d" in affected_cuts_dict:
        return MF.build_lessgtr_condition(array=df["d"], low=affected_cuts_dict["d"][0], high=affected_cuts_dict["d"][1],
                                          include=affected_cuts_lims_dict["d"])
    
    else:
        raise ValueError(f"Unexpected spatial cut `{affected_cuts_dict}`.")

def add_equatorial_coord_and_pm_to_df_if_needed(df):

    if "pmra" not in df or "pmdec" not in df:
        
        if "pmlcosb" not in df or "pmb" not in df:
            
            df["pmlcosb"] = df["vl"]/df["d"] # km/(s*kpc)
            df["pmb"] = df["vb"]/df["d"]

            for pm in ["pmlcosb", "pmb"]:
                df[pm] /= coordinates.get_conversion_kpc_to_km() # rad/s
                df[pm] /= coordinates.get_conversion_mas_to_rad() # mas/s
                df[pm] *= coordinates.get_conversion_yr_to_s() # mas/yr

        coordinates.pmlpmb_to_pmrapmdec(df)

    if "ra" not in df or "dec" not in df:
        coordinates.lb_to_radec(df)

    return df

def get_std_MC(df,true_value,function,montecarloconfig,vel_x_var=None,vel_y_var=None,tilt=False, absolute=True, R_hat=None, show_vel_plots=False, show_freq=10, velocity_kws={}):
    """
    Compute a Monte Carlo error in the statistical variable of interest given individual (one for each star) uncertainties.
    Given each star, takes the value of the variable to be perturbed and adds to it a number extracted from a Gaussian of mean 0 and standard
    deviation the uncertainty in the measurement. Does that for all the stars and repeats R number of times, each time computing the resulting
    statistic of interest. Compute the mean squared difference between the true value (computed from the original population) and all the 
    perturbed values.

    Parameters
    ----------
    df: pandas dataframe.
        Dataframe previous to applying any cuts affected by the perturbation (see montecarloconfig).
    true_value: float
        Value of the statistic of interest as computed using the unperturbed population (with all cuts applied).
    montecarloconfig: MonteCarloConfig object
        See docstring in errorconfig.py
    vel_x_var, vel_y_var: string or None
        If string, indicate the horizontal/vertical velocity components. For example, "r" or "l".
    tilt: boolean
        Whether the statistic of interest is a tilt (i.e. a vertex deviation).
    absolute: boolean
        Whether the tilt uses the absolute value of the dispersion difference. Only has effect if tilt is True.
    R_hat: tuple or None
        If a tuple, the statistic of interest is a spherical tilt, and R_hat indicates the 2D coordinates of the center of the bin of selected stars.
    show_vel_plots: boolean
        Whether to show a velocity plot of the stars after the perturbation.
    show_freq: integer
        Show a velocity plot every show_freq MC repetitions. Only has effect if show_vel_plots is True.
    velocity_kws: dictionary
        Keyword arguments for the velocity plot function.

    Returns
    -------
    std_low: float
        Mean squared difference between all the MC values of the statistic of interest and the true value computed from the unperturbed population.
        If montecarloconfig.symmetric is True, std_low is computed using all the MC values, and it will be identical to std_high. Otherwise,
        it will be computed using only the MC values below the true value.
    std_high: float
        See std_low. If montecarloconfig.symmetric is False, it will be computed using the MC values above the true value.
    MC_values: array
        NumPy array containing all the MC values.
    within_cut: array
        NumPy boolean array flagging the stars which, after the last perturbation, fell within the montecarloconfig.affected_cuts_dict
    """

    if vel_x_var is None and vel_y_var is None:
        raise ValueError("Both velocity components were set to None!")
    if montecarloconfig is None:
        raise ValueError("montecarloconfig cannot be None when estimating MC uncertainties.")
    if len(montecarloconfig.perturbed_vars) == 0:
        raise ValueError("The list of montecarloconfig.perturbed_vars was empty!")
    
    if "pmra" in montecarloconfig.perturbed_vars or "pmdec" in montecarloconfig.perturbed_vars or "d" in montecarloconfig.perturbed_vars:
        add_equatorial_coord_and_pm_to_df_if_needed(df)

    within_cut = None
    MC_values = np.empty(shape=(montecarloconfig.repeats))

    for i in range(montecarloconfig.repeats):

        df_helper = copy.deepcopy(df)

        if "vr" in montecarloconfig.perturbed_vars:
            apply_MC(df_helper, "vr", montecarloconfig.error_frac)

        if "pmra" in montecarloconfig.perturbed_vars:
            apply_MC(df_helper, "pmra", montecarloconfig.error_frac)

        if "pmdec" in montecarloconfig.perturbed_vars:
            apply_MC(df_helper, "pmdec", montecarloconfig.error_frac)

        if "d" in montecarloconfig.perturbed_vars:
            apply_MC(df_helper, "d", montecarloconfig.error_frac)

            within_cut = build_within_cut_boolean_array(df_helper, montecarloconfig.affected_cuts_dict, montecarloconfig.affected_cuts_lims_dict)
            
            if within_cut is not None:        
                df_helper = df_helper[within_cut]

            if montecarloconfig.random_resampling_indices is not None:

                resampled_indices = df_helper.index.intersection(montecarloconfig.random_resampling_indices)

                extra_N = len(montecarloconfig.random_resampling_indices) - len(resampled_indices)
                if extra_N > 0: # some of the resampled stars fell outside the spatial cut, let's take some extra ones
                    
                    extra_indices = np.random.choice(
                        df_helper.index.difference(resampled_indices), size=extra_N, replace=False
                    )

                    df_helper = df_helper.loc[resampled_indices.union(extra_indices)]
                else:
                    df_helper = df_helper.loc[resampled_indices]

        else:
            if montecarloconfig.random_resampling_indices is not None:
                df_helper = df_helper.loc[montecarloconfig.random_resampling_indices]

        MC_vx, MC_vy = extract_velocities_after_MC(df_helper, montecarloconfig.perturbed_vars, vel_x_var, vel_y_var)

        if vel_x_var is not None and vel_y_var is not None and show_vel_plots and i%show_freq == 0:
            velocity_plot(MC_vx,MC_vy,**velocity_kws)

        MC_values[i] = apply_function(function,MC_vx,MC_vy,R_hat,tilt,absolute)

    if tilt and not absolute:
        MC_values[(true_value - MC_values)>90] += 180
        MC_values[(true_value - MC_values)<-90] -= 180

    #Note this is a pseudo standard deviation: relative to true value as opposed to mean of MC values
    if montecarloconfig.symmetric:
        std = np.sqrt(np.nanmean((MC_values-true_value)**2))
        std_low,std_high = std,std
    else:
        std_low, std_high = compute_lowhigh_std(central_value=true_value,perturbed_values=MC_values)
    
    return std_low,std_high,MC_values,within_cut

def extract_velocities_after_MC(df, perturbed_vars, vel_x_var=None, vel_y_var=None):
    unexpected_perturbed_vars = [v for v in perturbed_vars if v not in ["pmra","pmdec","d","vr"]]
    if len(unexpected_perturbed_vars) > 0:
        raise ValueError(f"Got some unexpected perturbed variables: `{unexpected_perturbed_vars}`")
    
    if "pmra" in perturbed_vars or "pmdec" in perturbed_vars:
        coordinates.pmrapmdec_to_pmlpmb(df)
        coordinates.pmlpmb_to_vlvb(df)
    elif "d" in perturbed_vars:
        coordinates.pmlpmb_to_vlvb(df)

    MC_vx,MC_vy = None,None

    if vel_x_var is not None:
        if vel_x_var == "r":
            MC_vx = df["vr"]
        else: # this is here because extra conversions (e.g. vlvb to vxvy) will be needed to propagate the uncertainties to other velocity components
            raise ValueError(f"Velocity extraction undefined for vel_x_var `{vel_x_var}`.")

    if vel_y_var is not None:
        if vel_y_var == "l":
            MC_vy = df["vl"]
        else:
            raise ValueError(f"Velocity extraction undefined for vel_y_var `{vel_y_var}`.")

    return MC_vx,MC_vy

def get_std_bootstrap(function,bootstrapconfig,vx=None,vy=None,tilt=False,absolute=False,R_hat=None,show_vel_plots=False,show_freq=10,velocity_kws={}):
    '''
    Estimate the bootstrap standard error of a statistic, by computing the standard deviation of its sampling distribution.

    Parameters
    ----------
    function: callable
        Function used to compute the desired statistic whose error you want to estimate.

    bootstrapconfig: BootstrapConfig object
        See docstring in errorconfig.py

    vx, vy: array-like
        Arrays of desired velocity components. 
        vy=None if only 1 is needed.

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
    
    boot_values = np.full(shape=(bootstrapconfig.repeats), fill_value=None, dtype=float)

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

    assert None not in boot_values, "Some bootstrap values were not filled correctly."
    
    if tilt and not absolute:
        boot_values[(true_value - boot_values)>90] += 180
        boot_values[(true_value - boot_values)<-90] -= 180
    
    if bootstrapconfig.symmetric:
        std = np.std(boot_values)
        std_low,std_high = std,std
    else:
        std_low, std_high = compute_lowhigh_std(central_value=np.mean(boot_values),perturbed_values=boot_values)

    return std_low, std_high, boot_values

def get_std_bootstrap_recursive(function,bootstrapconfig,vx=None,vy=None,tilt=False,absolute=False,R_hat=None,show_vel_plots=False,show_freq=10,velocity_kws={}):
    """
    This function does the same as get_std_bootstrap above, but it computes the bootstrap standard error of both the original population and of
    each of the bootstrap samples.
    It can be used to test the key assumption of bootstrapping. The bootstrap attempts to estimate the standard error of a population using a
    discrete sample from it, under the assumption that the sample is representative of the population. If this assumptions holds true, the 
    simulated/theoretical standard error and the bootstrap standard error should be similar. Otherwise, one may find that the bootstrap standard 
    error underestimates the standard error (because the sample does not properly capture the variability of the underlying population) or 
    overestimates it.
    This function allows you to compute both so they can be compared. To compute the standard error of the desired statistic for samples
    of N stars, vx and vy should be given from the total population (which has number of stars >> N), and bootstrapconfig.bootstrap_size should be N.
    After computing the statistic for each sample for many bootstrap repeats, the standard deviation of the resulting distribution (the sampling
    distribution of the statistic) is the standard error. The bootstrap standard error is computed for each of the N-sized samples and also given.
    One may then compare the standard error and perhaps the mean of the bootstrap standard errors to check whether on average the latter 
    matches/overestimates/underestimates the actual standard error.
    """


    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")

    true_value = apply_function(function,vx,vy,R_hat,tilt,absolute)

    original_size = len(vx) if vx is not None else len(vy)
    indices_range = np.arange(original_size)

    bootstrap_size = original_size if bootstrapconfig.bootstrap_size is None else bootstrapconfig.bootstrap_size
    
    boot_values = np.full(shape=(bootstrapconfig.repeats), fill_value=None, dtype=float)

    nested_boot_errors = np.full(shape=(bootstrapconfig.repeats), fill_value=None, dtype=float)

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

        nested_boot_errors[i],*_ = get_std_bootstrap(function=function,bootstrapconfig=bootstrapconfig,vx=boot_vx,vy=boot_vy,tilt=tilt,absolute=absolute,R_hat=R_hat)
    
    assert None not in boot_values and None not in nested_boot_errors, "Some bootstrap values / nested errors were not filled correctly."

    if tilt and not absolute:
        boot_values[(true_value - boot_values)>90] += 180
        boot_values[(true_value - boot_values)<-90] -= 180
    
    if bootstrapconfig.symmetric:
        std = np.std(boot_values)
        std_low,std_high = std,std
    else:
        std_low, std_high = compute_lowhigh_std(central_value=np.mean(boot_values),perturbed_values=boot_values)
    
    return std_low, std_high, boot_values, nested_boot_errors

def compute_lowhigh_std(central_value, perturbed_values):
    values_above = perturbed_values[perturbed_values > central_value]
    values_below = perturbed_values[perturbed_values < central_value]
    values_equal = perturbed_values[perturbed_values == central_value]

    # divide perturbed values which result equal to the mean proportionally between the above and below values
    frac_equal_to_above = len(values_above) / (len(values_above) + len(values_below))
    idx_equal_to_above = int( len(values_equal) * frac_equal_to_above )

    values_above = np.append(values_above, values_equal[:idx_equal_to_above])
    values_below = np.append(values_below, values_equal[idx_equal_to_above:])

    std_low = np.sqrt(np.mean((values_below - central_value)**2)) if len(values_below) > 0 else 0
    std_high = np.sqrt(np.mean((values_above - central_value)**2)) if len(values_above) > 0 else 0

    return std_low, std_high