import utils.miscellaneous_functions as MF

class BootstrapConfig():
    def __init__(self, symmetric=True, repeats=500, replacement=True, bootstrap_size=None):
        """
        Parameters
        ----------
        bootstrap_size: int, optional
            Size of bootstrap samples. According to https://stats.stackexchange.com/questions/263710, it should be the same as the original sample. 
            Default is None, which sets it to the original sample size in the compute_variables.get_std_bootstrap function.
        
        replacement: boolean, optional
            Whether to take the samples with replacement or not. Defaults to True.
        
        repeats: int, optional
            Number of bootstrap samples to take. 
            Default is 500.
            
        symmetric: boolean, optional
            Whether to have symmetric low and high errors.
            Default is True.
        """
        self.repeats = repeats
        self.replacement = replacement
        self.bootstrap_size = bootstrap_size
        self.symmetric = symmetric

class MonteCarloConfig():
    def __init__(self, perturbed_vars, affected_cuts_dict, repeats=500, affected_cuts_lims_dict=None, error_frac=None,symmetric=False,\
                  random_resampling_indices=None):
        """
        Parameters
        ----------
        perturbed_vars: list of strings
            List of variables to perturb.
        error_frac: float or None
            If float, it is the fractional uncertainty in the measurements.
            If None, take the error from the actual measurement uncertainties (in the observations).
        repeats: integer
            Number of MC repeats.
        symmetric: boolean
            If False, the MC method computes the standard deviation for values above and below the true value separately.
        random_resampling_indices: pandas Index object or None
            If a pandas Index object, it contains the indices of the stars in the dataframe used to compute the true_value, i.e. after applying
                                     any of the affected cuts, which were selected after random downsampling. If any of those stars falls outside
                                     the affected cuts after the perturbation, we incorporate other stars to complete the sample.
            If None, no downsampling was performed.
        affected_cuts_dict: dictionary or None
            Dictionary of cuts, where each key is a variable (str) and the values are a tuple containing the minimum and maximum values,
            which are affected by the perturbation of any of the perturbed variables. These cuts have to be applied after the MC perturbation
            as the stars might move across the limit (for example when applying a distance perturbation and wanting a cut on R). 
            In the calculation of MC values, we exclude/include the stars which fall outside/inside the limits after the perturbation respectively.
        affected_cuts_lims_dict: dictionary or None
            Dictionary with the same keys as affected_cuts_dict, where the values are "min", "max", "both" or "neither", indicating the limits
            that are included in the cut. 
            If None, or some variables in affected_cuts_dict are not in the keys, the default is "both".
        """
        
        self.perturbed_vars = perturbed_vars
        self.error_frac = error_frac
        self.repeats = repeats
        self.symmetric = symmetric
        self.random_resampling_indices = random_resampling_indices
        
        self.affected_cuts_dict = {} if affected_cuts_dict is None else affected_cuts_dict
        self.affected_cuts_lims_dict = {} if affected_cuts_lims_dict is None else affected_cuts_lims_dict
        
        for k in self.affected_cuts_dict:
            if k not in self.affected_cuts_lims_dict:
                self.affected_cuts_lims_dict[k] = "both"