import numpy as np
import copy
import scipy.stats as stats
from collections import namedtuple

import src.compute_variables as CV
from plotting.velocity_plot import velocity_plot
import utils.coordinates as coordinates
import utils.miscellaneous_functions as MF
import utils.error_helpers as EH

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
    Compute a Monte Carlo error in the statistical variable of interest given individual uncertainties (one for each star), like so:
        1. Given each star, take the value of the variable to be perturbed and adds to it a number extracted from a Gaussian of mean 0 and standard
            deviation the uncertainty in the measurement. 
        2. Do that for all the stars and repeat R number of times, each time computing the resulting statistic of interest. 
        3. Split the distribution in two groups, for values below and above the true value.
        4. Values exactly equal to the true value are divided proportionally between the below and above split.
        5. Separately for the below and above splits, compute the mean squared deviation of the MC values from the true value.
            If not done separately, this would be equivalent to adding the standard deviation of the MC distribution and its bias,
             which is the difference between the mean of the MC distribution and the true value, in quadrature: https://stats.stackexchange.com/questions/646916
        6. The left and right limits of the confidence interval are then given as the square root of the mean squared left/right deviations from step 5.

    Parameters
    ----------
    df: pandas dataframe.
        Dataframe previous to applying any cuts affected by the perturbation (see montecarloconfig docstring).
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
    confidence_interval: tuple
        The left,right limits of the confidence interval of the true value. They are given as a distance from the true value.
        The confidence intervals are 1 pseudo-std (see explanation above), resulting from adding the mean of the MC values and the bias in quadrature.
    MC_values: array
        NumPy array containing all the MC values.
    bias: float
        The difference between the mean of the MC distribution and the true value.
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

        MC_values[i] = EH.apply_function(function,MC_vx,MC_vy,R_hat,tilt,absolute)

    if tilt and not absolute:
        MC_values[(true_value - MC_values)>90] += 180
        MC_values[(true_value - MC_values)<-90] -= 180

    if montecarloconfig.symmetric:
        pseudo_std = np.sqrt(np.nanmean((MC_values-true_value)**2))
        CI_low,CI_high = pseudo_std,pseudo_std
    else:
        CI_low, CI_high = EH.compute_lowhigh_std(central_value=true_value, values=MC_values)

    Result = namedtuple("Result", ["confidence_interval", "MC_distribution", "bias", "within_cut"])
    
    return Result(confidence_interval=(CI_low,CI_high), MC_distribution=MC_values, bias=np.mean(MC_values)-true_value, within_cut=within_cut)

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

################################################################### BOOTSTRAPPING #############################################################################

def get_std_bootstrap(function,bootstrapconfig,vx=None,vy=None,tilt=False,absolute=False,R_hat=None, show_vel_plots=False,show_freq=10,velocity_kws={}):
    '''
    Estimate the confidence interval of your sample estimate using bootstrapping. The confidence interval takes into account both bias and skew. It is obtained the following way:
        1. Build the bootstrap distribution.
        2. Split the distribution in two groups, for values below and above the original sample estimate.
        3. Values exactly equal to the original sample estimate are divided proportionally between the below and above split.
        4. Separately for the below and above splits, compute the mean squared deviation of the bootstrap esimates from the original sample estimate.
            If not done separately, this would be equivalent to adding the bootstrap standard error and the bias in quadrature: https://stats.stackexchange.com/questions/646916
        5. The left and right limits of the confidence interval are then given as the square root of the mean squared left/right deviations from step 4.

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
    confidence_interval: tuple
        The left,right limits of the confidence interval of the original sample estimate. They are given as a distance from the original sample estimate.
        The confidence intervals are 1 pseudo-std (see explanation above), resulting from adding the standard error and bias in quadrature.
    
    boot_values: array-like
        The bootstrap values.
    
    std: float
        The bootstrap standard error, computed as the standard deviation of the bootstrap distribution.

    bias: float
        The difference between the mean of the bootstrap distribution and the original sample estimate.
    '''
    
    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")

    original_sample_estimate = EH.apply_function(function,vx,vy,R_hat,tilt,absolute)

    original_sample_size = len(vx) if vx is not None else len(vy)
    indices_range = np.arange(original_sample_size)

    bootstrap_sample_size = original_sample_size if bootstrapconfig.bootstrap_size is None else bootstrapconfig.bootstrap_size
    
    boot_values = np.full(shape=(bootstrapconfig.repeats), fill_value=None, dtype=float)

    for i in range(bootstrapconfig.repeats):
        bootstrap_indices = np.random.choice(indices_range, size = bootstrap_sample_size, replace = bootstrapconfig.replacement)
        
        boot_vx,boot_vy = None,None

        if vx is not None:
            boot_vx = vx[bootstrap_indices]
        if vy is not None:
            boot_vy = vy[bootstrap_indices]
        
        if show_vel_plots and i%show_freq == 0 and boot_vx is not None and boot_vy is not None:
            velocity_plot(boot_vx,boot_vy,**velocity_kws)
            
        boot_values[i] = EH.apply_function(function,boot_vx,boot_vy,R_hat,tilt,absolute)

    assert None not in boot_values, "Some bootstrap values were not filled correctly."
    
    if tilt and not absolute:
        boot_values[(original_sample_estimate - boot_values)>90] += 180
        boot_values[(original_sample_estimate - boot_values)<-90] -= 180

    if bootstrapconfig.symmetric:
        pseudo_std = np.sqrt(np.nanmean((boot_values-original_sample_estimate)**2))
        CI_low,CI_high = pseudo_std,pseudo_std
    else:
        CI_low, CI_high = EH.compute_lowhigh_std(central_value=original_sample_estimate, values=boot_values)

    Result = namedtuple("Result", ["confidence_interval", "bootstrap_distribution", "standard_error", "bias"])
    
    return Result(confidence_interval=(CI_low,CI_high), bootstrap_distribution=boot_values, standard_error=np.std(boot_values), bias=np.mean(boot_values)-original_sample_estimate)

def get_std_bootstrap_recursive(function,bootstrapconfig,nested_bootstrapconfig=None,vx=None,vy=None,tilt=False,absolute=False,R_hat=None,show_vel_plots=False,show_freq=10,velocity_kws={}):
    """
    This function does the same as get_std_bootstrap above but on two depth levels: for the original sample (or population) and for each of the bootstrap samples.

    It can be used to test the key assumption of bootstrapping, which is that the sample you are working with is representative of the underlying population. 
    If this assumptions holds true, the standard error (standard deviation of the sampling distribution, extracted from the population) and the bootstrap 
    standard error (standard deviation of the bootstrap distribution) should be similar. Otherwise, one may find that the bootstrap standard error 
    underestimates the standard error (because the sample does not properly capture the variability of the underlying population) or overestimates it.
    
    This function allows you to compute both so they can be compared. To compute the standard error of the desired statistic for samples
    of N stars, vx and vy should be given from the total population (which has number of stars >> N), and bootstrapconfig.bootstrap_size should be N.
    After computing the statistic for each sample for many repeats, the standard deviation of the resulting distribution (the sampling
    distribution of the statistic) is the standard error. The bootstrap standard error is computed for each of the N-sized samples and also given.
    One may then compare the standard error and perhaps the mean of the bootstrap standard errors to check whether on average the latter 
    matche/overestimate/underestimate the actual standard error.
    """

    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")
    if nested_bootstrapconfig is None:
        nested_bootstrapconfig = bootstrapconfig

    original_sample_estimate = EH.apply_function(function,vx,vy,R_hat,tilt,absolute)

    original_sample_ize = len(vx) if vx is not None else len(vy)
    indices_range = np.arange(original_sample_ize)

    bootstrap_sample_size = original_sample_ize if bootstrapconfig.bootstrap_size is None else bootstrapconfig.bootstrap_size
    
    boot_values = np.full(shape=(bootstrapconfig.repeats), fill_value=None, dtype=float)
    nested_boot_errors = np.full(shape=(bootstrapconfig.repeats), fill_value=None, dtype=float)

    for i in range(bootstrapconfig.repeats):
        bootstrap_indices = np.random.choice(indices_range, size = bootstrap_sample_size, replace = bootstrapconfig.replacement)
        
        boot_vx,boot_vy = None,None

        if vx is not None:
            boot_vx = vx[bootstrap_indices]
        if vy is not None:
            boot_vy = vy[bootstrap_indices]
        
        if show_vel_plots and i%show_freq == 0 and boot_vx is not None and boot_vy is not None:
            velocity_plot(boot_vx,boot_vy,**velocity_kws)
            
        boot_values[i] = EH.apply_function(function,boot_vx,boot_vy,R_hat,tilt,absolute)

        nested_boot_errors[i],*_ = get_std_bootstrap(function=function,bootstrapconfig=nested_bootstrapconfig,vx=boot_vx,vy=boot_vy,tilt=tilt,absolute=absolute,R_hat=R_hat)
    
    assert None not in boot_values and None not in nested_boot_errors, "Some bootstrap values / nested errors were not filled correctly."

    if tilt and not absolute:
        boot_values[(original_sample_estimate - boot_values)>90] += 180
        boot_values[(original_sample_estimate - boot_values)<-90] -= 180

    if bootstrapconfig.symmetric:
        pseudo_std = np.sqrt(np.nanmean((boot_values-original_sample_estimate)**2))
        CI_low,CI_high = pseudo_std,pseudo_std
    else:
        CI_low, CI_high = EH.compute_lowhigh_std(central_value=original_sample_estimate, values=boot_values)

    Result = namedtuple("Result", ["confidence_interval", "bootstrap_distribution", "standard_error", "bias"])
    
    return Result(confidence_interval=(CI_low,CI_high), bootstrap_distribution=boot_values, standard_error=np.std(boot_values), bias=np.mean(boot_values)-original_sample_estimate)

################################################################################################################################################

def get_error_vertex_deviation_roca_fabrega(n,vx,vy):
    """
    Expression from Roca-Fabrega et al. (2014), at https://doi.org/10.1093/mnras/stu437
    """

    mu_110 = CV.calculate_covariance(vx,vy)
    mu_200 = np.var(vx)
    mu_020 = np.var(vy)
    mu_220 = np.mean( (vx-np.mean(vx))**2 * (vy-np.mean(vy))**2 )
    mu_400 = stats.moment(vx,moment=4)
    mu_040 = stats.moment(vy,moment=4)

    a1 = 2/(n-1)-3/n
    a2 = 1/(n-1)-2/n
    a3 = 1/(n-1)-1/n
    a4 = 1/(mu_110*(a1+4))
    b1 = (mu_400+mu_040)/n
    b2 = mu_200**2 + mu_020**2
    b3 = (mu_200-mu_020)**2 / mu_110**2
    
    parenthesis = mu_220/n + mu_110**2 * a2 + mu_200*mu_020*a3

    return np.abs(a4) * np.sqrt(b1 + a1*b2 + b3*parenthesis)