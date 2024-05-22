
class BootstrapConfig():
    def __init__(self, sample_size=None, repeats=5000, replace=True, symmetric=True, from_mean=False, batch_size=None):
        """
        Parameters
        ----------
        sample_size: int, optional
            Size of each bootstrap sample. For proper bootstrapping it should be the same as the original sample size (see e.g. https://stats.stackexchange.com/questions/263710).
            The default behaviour, i.e. when `sample_size` is set to None, is to use the original sample size.
        
        replace: boolean, optional
            Whether to take the samples with replacement or not. Defaults to True.
        
        repeats: int, optional. Default is 5000
            Number of bootstrap samples to take. 
            
        symmetric: boolean, optional. Default is True.
            Whether to the confidence interval should be computed symmetrically with respect to the original sample estimate, or values below/above it should be 
            used separately. See src.bootstrap_errors.get_std_bootstrap docstring

        from_mean: boolean, optional. Default is False.
            Whether the confidence interval should be computed around the mean. If False, it is computed from the original sample estimate.

        batch_size: int, optional. Default is None.
            If using vectorised code in src.bootstrap_errors for the different bootstrap samples (i.e. the different repeats) and experiencing memory issues,
            set a batch_size. The default behaviour, i.e. when `batch_size` is set to None, is to use a single batch containing all repeats.
        """
        self.repeats = repeats
        self.replace = replace
        self.sample_size = sample_size
        self.symmetric = symmetric
        self.from_mean = from_mean
        self.batch_size = batch_size if batch_size is not None else repeats

class MonteCarloConfig():
    def __init__(self, perturbed_vars, affected_cuts_dict, repeats=5000, error_frac=None,symmetric=False, affected_cuts_lims_dict=None,\
                  random_resampling_indices=None):
        """
        Parameters
        ----------
        perturbed_vars: list of strings
            List of variables to perturb.
        affected_cuts_dict: dictionary or None
            Dictionary of cuts, where each key is a variable (str) and the values are a tuple containing the minimum and maximum values,
            which are affected by the perturbation of any of the perturbed variables. These cuts have to be applied after the MC perturbation
            as the stars might move across the limit (for example when applying a distance perturbation and wanting a cut on R). 
            In the calculation of MC values, we exclude/include the stars which fall outside/inside the limits after the perturbation respectively.
        repeats: integer, optional. Default is 5000
            Number of MC repeats.
        error_frac: float or None. Default is None
            If float, it is the fractional uncertainty in the measurements.
            If None, take the error from the actual measurement uncertainties (which exist in the observations).
        symmetric: boolean. Default is False.
            If False, the MC method computes the standard deviation for values above and below the true value separately.
        affected_cuts_lims_dict: dictionary or None. Default is None.
            Dictionary with the same keys as affected_cuts_dict, where the values are "min", "max", "both" or "neither", indicating the limits
            that are included in the cut. 
            If None, or some variables in affected_cuts_dict are not in the keys, the default is "both".
        random_resampling_indices: pandas Index object or None. Default is None
            If a pandas Index object, it contains the indices of the stars in the dataframe used to compute the true value, i.e. after applying
                                     any of the affected cuts, which were selected after random downsampling. If any of those stars falls outside
                                     the affected cuts after the perturbation, we incorporate other stars to complete the sample.
            If None, no downsampling was performed.
        """
        
        self.perturbed_vars = perturbed_vars
        self.error_frac = error_frac
        self.repeats = repeats
        self.symmetric = symmetric
        self.random_resampling_indices = random_resampling_indices
        
        for k in self.affected_cuts_dict:
            if k not in self.affected_cuts_lims_dict:
                self.affected_cuts_lims_dict[k] = "both"

        # Set to {} here instead of in the default values of the arguments because default arguments are evaluated only once and hence are mutable.
        # See, e.g., https://www.30secondsofcode.org/python/s/mutable-default-arguments/
        self.affected_cuts_dict = {} if affected_cuts_dict is None else affected_cuts_dict
        self.affected_cuts_lims_dict = {} if affected_cuts_lims_dict is None else affected_cuts_lims_dict
        