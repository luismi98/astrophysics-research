import numpy as np
import copy
from collections import namedtuple

from plotting.velocity_plot import velocity_plot
import utils.coordinates as coordinates
import utils.miscellaneous_functions as MF
import utils.error_helpers as error_helpers

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

def get_std_MC(df,true_value,function,config,vel_x_var=None,vel_y_var=None,tilt=False, absolute=True, R_hat=None, show_vel_plots=False, show_freq=10, velocity_kws={}):
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
        Dataframe previous to applying any cuts affected by the perturbation (see src.MonteCarloConfig docstring).
    true_value: float
        Value of the statistic of interest as computed using the unperturbed population (with all cuts applied).
    config: MonteCarloConfig object
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
        NumPy boolean array flagging the stars which, after the last perturbation, fell within the `config.affected_cuts_dict`
    """

    if vel_x_var is None and vel_y_var is None:
        raise ValueError("Both velocity components were set to None!")
    if config is None:
        raise ValueError("config cannot be None when estimating MC uncertainties.")
    if len(config.perturbed_vars) == 0:
        raise ValueError("The list of config.perturbed_vars was empty!")
    
    if "pmra" in config.perturbed_vars or "pmdec" in config.perturbed_vars or "d" in config.perturbed_vars:
        add_equatorial_coord_and_pm_to_df_if_needed(df)

    within_cut = None
    MC_values = np.empty(shape=(config.repeats))

    for i in range(config.repeats):

        df_helper = copy.deepcopy(df)

        if "vr" in config.perturbed_vars:
            apply_MC(df_helper, "vr", config.error_frac)

        if "pmra" in config.perturbed_vars:
            apply_MC(df_helper, "pmra", config.error_frac)

        if "pmdec" in config.perturbed_vars:
            apply_MC(df_helper, "pmdec", config.error_frac)

        if "d" in config.perturbed_vars:
            apply_MC(df_helper, "d", config.error_frac)

            within_cut = build_within_cut_boolean_array(df_helper, config.affected_cuts_dict, config.affected_cuts_lims_dict)
            
            if within_cut is not None:        
                df_helper = df_helper[within_cut]

            if config.random_resampling_indices is not None:

                resampled_indices = df_helper.index.intersection(config.random_resampling_indices)

                extra_N = len(config.random_resampling_indices) - len(resampled_indices)
                if extra_N > 0: # some of the resampled stars fell outside the spatial cut, let's take some extra ones
                    
                    extra_indices = np.random.choice(
                        df_helper.index.difference(resampled_indices), size=extra_N, replace=False
                    )

                    df_helper = df_helper.loc[resampled_indices.union(extra_indices)]
                else:
                    df_helper = df_helper.loc[resampled_indices]

        else:
            if config.random_resampling_indices is not None:
                df_helper = df_helper.loc[config.random_resampling_indices]

        MC_vx, MC_vy = extract_velocities_after_MC(df_helper, config.perturbed_vars, vel_x_var, vel_y_var)

        if vel_x_var is not None and vel_y_var is not None and show_vel_plots and i%show_freq == 0:
            velocity_plot(MC_vx,MC_vy,**velocity_kws)

        MC_values[i] = error_helpers.apply_function(function,MC_vx,MC_vy,R_hat,tilt,absolute)

    if tilt and not absolute:
        error_helpers.correct_tilt_branch(MC_values, true_value)

    CI_low, CI_high = error_helpers.build_confidence_interval(MC_values, true_value, symmetric=config.symmetric)

    Result = namedtuple("Result", ["confidence_interval", "MC_distribution", "bias", "within_cut"])
    
    return Result(confidence_interval=(CI_low,CI_high), MC_distribution=MC_values, bias=np.mean(MC_values)-true_value, within_cut=within_cut)