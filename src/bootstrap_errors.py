import numpy as np
import scipy.stats as stats
from collections import namedtuple

import utils.error_helpers as error_helpers

def get_std_bootstrap(function, config, vx=None, vy=None, tilt=False, absolute=True, R_hat=None, vectorised=False, batch_size=None):
    """
    Estimate the confidence interval of your sample estimate, given as a distance relative to the sample estimate, using the Monte Carlo sampling
    implementation of non-parametric bootstrapping. See, e.g., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4784504/ for more details.

    The confidence interval takes into account both bias and skew. The low/high limits are obtained the following way:
        1. Build the bootstrap distribution (taking samples with replacement from the original sample).
        2. Split the distribution in two groups, for values below and above the original sample estimate.
        3. Values exactly equal to the original sample estimate are divided proportionally between the below and above split.
        4. Separately for the below and above splits, compute the mean squared deviation of the bootstrap esimates from the original sample estimate.
            Computing the deviation from the original sample estimate is equivalent to adding the bootstrap standard error and the bias in quadrature.
            The boostrap standard error is the standard deviation of the bootstrap distribution, and the bias is the difference between the
            mean of the bootstrap distribution (called the bootstrap mean) and the original sample estimate.
            See https://stats.stackexchange.com/questions/646916
        5. The low/high limits of the confidence interval are given as the square root of the mean squared below/above deviations from step 4.

    The above changes when:
    - config.symmetric is True: all the values are treated the same, without the below/above split, hence the low/high limits of the
                                         confidence interval are the same distance away from the original sample estimate.
    - config.from_mean is True: the confidence interval is computed around the bootstrap mean instead of the original sample estimate.
    - config.replace is False: the bootstrap distribution is built without replacement. This can be used to build the sampling distribution
                                        of the statistic of interest, by sampling without replacement from the population (with 
                                        bootstrap.sample_size << population size). Note this is not bootstrapping anymore.

    Parameters
    ----------
    function: callable
        Function used to compute the desired statistic whose error you want to estimate.

    config: BootstrapConfig object
        See docstring in src/errorconfig.py

    vx, vy: array-like, optional. Default is None.
        Arrays of desired velocity components. 
        At least one of the components must be given. If both are None, it raises a ValueError.

    tilt: bool, optional. Default is False.
        Boolean variable indicating whether the error is being estimated on a tilt (including spherical tilt). 

    absolute: bool, optional. Default is False.
        Only has effect if tilt=True. Determines whether the tilt is being computed with absolute value in the denominator or not. 

    R_hat: array-like, optional. Default is None.
        Only has effect if tilt=True. If None, the tilt is a vertex deviation, otherwise it is a spherical tilt.

    vectorised: bool, optional. Default is False.
        Whether to perform the bootstrap repeats in parallel.

    batch_size: int, optional. Default is None.
        Only has effect if `vectorised` is True. Size of batches in which to divide the bootstrap repeats, to reduce memory usage.
        If None, all repeats will be processed in a single batch.

    Returns
    -------
    A Result object with the below attributes.

    confidence_interval: tuple
        The low and high limits of the confidence interval. They are given as a distance from the original sample estimate (or mean if config.from_mean is True).
        If config.from_mean is False, the confidence interval is 1 pseudo-std (see explanation above), resulting from adding the standard error and bias in quadrature.
    
    bootstrap_distribution: 1D array of length `config.repeats`
        All the bootstrap estimates.
    
    standard_error: float
        The bootstrap standard error, computed as the standard deviation of the bootstrap distribution.

    bias: float
        The difference between the mean of the bootstrap distribution and the original sample estimate.
    """
    
    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")

    original_sample_estimate = error_helpers.apply_function(function,vx,vy,R_hat,tilt,absolute)
    
    boot_values = build_bootstrap_distribution(function=function,config=config,vx=vx,vy=vy,R_hat=R_hat,tilt=tilt,absolute=absolute,vectorised=vectorised,batch_size=batch_size)

    assert not np.any(np.isnan(boot_values)), "Some bootstrap values were not filled correctly."
    
    central_value = np.mean(boot_values) if config.from_mean else original_sample_estimate

    if tilt and not absolute:
        error_helpers.correct_tilt_branch(boot_values, central_value)

    CI_low, CI_high = error_helpers.build_confidence_interval(boot_values, central_value, symmetric=config.symmetric)

    Result = namedtuple("Result", ["confidence_interval", "bootstrap_distribution", "standard_error", "bias"])
    
    return Result(confidence_interval=(CI_low,CI_high), bootstrap_distribution=boot_values, standard_error=np.std(boot_values), bias=np.mean(boot_values)-original_sample_estimate)

def build_bootstrap_distribution(function, config, vx, vy, tilt, absolute, R_hat, vectorised, batch_size):
    """
    See get_std_bootstrap docstring.
    """

    original_sample_size = len(vx) if vx is not None else len(vy)
    bootstrap_sample_size = original_sample_size if config.sample_size is None else config.sample_size
    total_repeats = config.repeats

    bootstrap_distribution = np.full(shape=(total_repeats), fill_value=None, dtype=float)

    if not vectorised:
        for i in range(total_repeats):
            indices = np.random.choice(original_sample_size, size = bootstrap_sample_size, replace = config.replace)
            
            boot_vx = vx[indices] if vx is not None else None
            boot_vy = vy[indices] if vy is not None else None
                
            bootstrap_distribution[i] = error_helpers.apply_function(function,boot_vx,boot_vy,R_hat,tilt,absolute)

    else:
        for i in range(0, total_repeats, batch_size):
            indices = get_2D_indices_to_resample(data_length=original_sample_size, sample_size=bootstrap_sample_size, repeats=min(batch_size, total_repeats-i), replace_bool=config.replace)

            boot_vx = vx[indices] if vx is not None else None
            boot_vy = vy[indices] if vy is not None else None

            max_i = i + batch_size if i + batch_size < total_repeats else total_repeats

            bootstrap_distribution[i:max_i] = error_helpers.apply_function(function, boot_vx, boot_vy, R_hat, tilt, absolute)
    
    return bootstrap_distribution

def get_2D_indices_to_resample(data_length, sample_size, repeats, replace_bool):
    """
    Builds an array of indices, of size (repeats, sample_size), representing `repeats` samples, each of size `sample_size`.
    Elements can repeat across different samples. Indices within each sample are drawn with/without replacement (depending on `replace`) from the data range.

    Usage:
    samples = data[indices]
    """

    if replace_bool:
        indices = np.random.choice(data_length, size=(repeats,sample_size), replace=True)
    else:
        indices = np.empty((repeats, sample_size), dtype=int)
        for i in range(repeats):
            indices[i] = np.random.choice(data_length, size=sample_size, replace=False)
        
    return indices

def get_std_bootstrap_recursive(function,config,nested_config=None,vx=None,vy=None,tilt=False,absolute=False,R_hat=None):
    """
    This function does the same as get_std_bootstrap above but on two depth levels: for a population and for each sample in its sampling distribution.

    It can be used to test the key assumption of bootstrapping, which is that the sample you are working with is representative of the underlying population. 
    If this assumptions holds true:
        - The bootstrap standard error (standard deviation of the bootstrap distribution) should be a good approximation of the standard error 
          (standard deviation of the sampling distribution, extracted from the population).
        - The bootstrap bias (difference between the mean of the bootstrap distribution and the original sample estimate) should be a good
          approximation of the bias (difference between the mean of the sampling distribution and the true value, computed from the population).
    
    The sampling distribution (for a given sample size N) is obtained by passing to this function the vx and vy from the total population 
    (which has number of stars >> N). The population is resampled many times, with sampling size N, and the sampling distribution is the histogram
    of the statistic of interest computed from all of those samples. For each of these samples of size N, a bootstrap distribution is produced,
    sampling with replacement from them.

    Given we will have a set of bootstrap estimates (of confidence interval, standard error, bias) for each sample in the sampling distribution,
    we can compare some statistic across these estimates (e.g. average amplitude of confidence interval, mean bootstrap standard error, mean bias)
    with the single value (of confidence interval, standard error, or bias) obtained from the sampling distribution itself.

    Parameters
    ----------
    See get_std_bootstrap docstring for an explanation of all parameters except:

    nested_config: BootstrapConfig object, optional. Default is None, in which case it will be set to `config`.
        Bootstrap configuration for the nested bootstrapping, i.e. for the construction of the bootstrap distributions for each sample in the
        sampling distribution. The sampling distribution itself is built using the configuration in `config`.

    Returns
    -------
    A Result object with the below attributes.

    confidence_interval: tuple
        The low and high limits of the confidence interval computed from the sampling distribution.

    bootstrap_confidence_intervals: 2D array of shape (nested_bootstrap_config.repeats, 2)
        The low and high limits of the confidence intervals computed from the bootstrap distribution of each sample in the sampling distribution
    
    standard_error: float
        The standard error, computed as the standard deviation of the sampling distribution.

    bootstrap_standard_errors: 1D array of length `nested_bootstrap_config.repeats`
        The bootstrap standard errors, computed as the standard deviation of the bootstrap distribution of each sample in the sampling distribution

    bias: float
        The difference between the mean of the sampling distribution and the true value (computed from the population).

    bootstrap_biases: 1D array of length `nested_bootstrap_config.repeats`
        All the differences between the mean of each bootstrap distribution and its associated sample estimate.
    """

    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")
    if nested_config is None:
        nested_config = config

    true_value = error_helpers.apply_function(function,vx,vy,R_hat,tilt,absolute)

    population_size = len(vx) if vx is not None else len(vy)
    sample_size = population_size if config.sample_size is None else config.sample_size
    
    sampling_values = np.full(shape=(config.repeats), fill_value=np.nan)
    bootstrap_confidence_intervals = np.full(shape=(nested_config.repeats, 2), fill_value=np.nan)
    bootstrap_standard_errors = np.full(shape=(nested_config.repeats), fill_value=np.nan)
    bootstrap_biases = np.full(shape=(nested_config.repeats), fill_value=np.nan)

    for i in range(config.repeats):
        bootstrap_indices = np.random.choice(population_size, size = sample_size, replace = config.replace)
        
        sampled_vx = vx[bootstrap_indices] if vx is not None else None
        sampled_vy = vy[bootstrap_indices] if vy is not None else None
            
        sampling_values[i] = error_helpers.apply_function(function,sampled_vx,sampled_vy,R_hat,tilt,absolute)

        res = get_std_bootstrap(function=function,config=nested_config,vx=sampled_vx,vy=sampled_vy,tilt=tilt,absolute=absolute,R_hat=R_hat)
        bootstrap_confidence_intervals[i] = res.confidence_interval
        bootstrap_standard_errors[i] = res.standard_error
        bootstrap_biases[i] = res.bias
    
    assert not np.any(np.isnan(sampling_values)), "Some bootstrap values were not filled correctly."
    assert not np.any(np.isnan(bootstrap_standard_errors)), "Some nested errors were not filled correctly."
    assert not np.any(np.isnan(bootstrap_confidence_intervals)), "Some nested confidence intervals were not filled correctly"
    assert not np.any(np.isnan(bootstrap_biases)), "Some nested biases were not filled correctly"

    central_value = np.mean(sampling_values) if config.from_mean else true_value

    if tilt and not absolute:
        error_helpers.correct_tilt_branch(sampling_values, central_value)

    CI_low, CI_high = error_helpers.build_confidence_interval(sampling_values, central_value, symmetric=config.symmetric)

    Result = namedtuple("Result", ["confidence_interval", "nested_confidence_intervals", 
                                   "standard_error", "nested_standard_errors",
                                   "bias", "bootstrap_biases"])
    
    return Result(confidence_interval = (CI_low,CI_high),
                  nested_confidence_intervals = bootstrap_confidence_intervals, 
                  standard_error = np.std(sampling_values), 
                  nested_standard_errors = bootstrap_standard_errors,
                  bias = np.mean(sampling_values) - true_value,
                  bootstrap_biases = bootstrap_biases)

def scipy_bootstrap(function, config, vx=None, vy=None, tilt=False,absolute=False,R_hat=None):
    """
    Apply the BCa (bias-corrected and accelerated) bootstrap method, using `stats.scipy.bootstrap`.
    This method takes into account both bias and skew when computing the confidence interval.

    Note: the low/high limits of the confidence interval are given as values of the statistic, contrary to the limits given by
          get_std_bootstrap which are given as a distance from the original sample estimate.

    See get_std_bootstrap docstring for an explanation of the parameters and return values.
    """

    if vx is None and vy is None:
        raise ValueError("At least one of vx and vy must not be None.")

    res = stats.bootstrap(
            data=(vx,vy),
            statistic=lambda vx,vy: error_helpers.apply_function(function=function,vx=vx,vy=vy, R_hat=R_hat, tilt=tilt, absolute=absolute),
            n_resamples=config.repeats,
            vectorized=False,
            paired=True,
            confidence_level=0.68,
            method="BCa"
        )
    
    Result = namedtuple("Result", ["confidence_interval", "bootstrap_distribution", "standard_error", "bias"])
    
    original_sample_estimate = error_helpers.apply_function(function=function,vx=vx,vy=vy, R_hat=R_hat, tilt=tilt, absolute=absolute)

    return Result(confidence_interval=res.confidence_interval,
                  bootstrap_distribution=res.bootstrap_distribution,
                  standard_error=np.std(res.bootstrap_distribution),
                  bias=np.mean(res.bootstrap_distribution)-original_sample_estimate)