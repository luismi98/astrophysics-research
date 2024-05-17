import numpy as np
from collections import namedtuple

from plotting.velocity_plot import velocity_plot
import utils.error_helpers as EH

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
        EH.correct_tilt_branch(boot_values, original_sample_estimate)

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
        EH.correct_tilt_branch(boot_values, original_sample_estimate)

    if bootstrapconfig.symmetric:
        pseudo_std = np.sqrt(np.nanmean((boot_values-original_sample_estimate)**2))
        CI_low,CI_high = pseudo_std,pseudo_std
    else:
        CI_low, CI_high = EH.compute_lowhigh_std(central_value=original_sample_estimate, values=boot_values)

    Result = namedtuple("Result", ["confidence_interval", "bootstrap_distribution", "standard_error", "bias"])
    
    return Result(confidence_interval=(CI_low,CI_high), bootstrap_distribution=boot_values, standard_error=np.std(boot_values), bias=np.mean(boot_values)-original_sample_estimate)