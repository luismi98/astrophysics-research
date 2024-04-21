import miscellaneous_functions as MF

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
    def __init__(self, perturbed_var, affected_cuts_dict, repeats=500, affected_cuts_lims_dict=None, error_frac=None, random_resampling_indices=None):
        
        self.perturbed_var = perturbed_var
        self.error_frac = error_frac
        self.repeats = repeats
        
        # When random resampling (without replacement), we perform it before binning (to compute the true values) and here we keep track of
        # the result of that resampling when applying the MC. After the perturbation, we select the same stars as we did originally for the true values
        # if they have falled within the affected_cuts_dict. If there are any that now fall outside the range, we incorporate other stars
        # to complete the sample (i.e. have the same number of stars as in the unperturbed bin)
        self.random_resampling_indices = random_resampling_indices
        
        # Cuts affected by the perturbed variable have to be applied after the MC perturbation separate as the stars might move across the limit.
        # We exclude/include the stars which fall outside/inside the limits after the perturbation respectively
        self.affected_cuts_dict = affected_cuts_dict
        
        if affected_cuts_lims_dict is None:
            self.affected_cuts_lims_dict = {}
            
            for k in self.affected_cuts_dict:
                self.affected_cuts_lims_dict[k] = "both"
                
    def clean_value_cuts_dict(self, cuts_dict):
        if type(cuts_dict) != dict:
            cuts_dict = MF.merge_dictionaries(cuts_dict)
        
        cleaned_dict = {}
        
        for k,v in cuts_dict.items():
            if k in self.affected_cuts_dict:
                if v == self.affected_cuts_dict[k]:
                    continue
                else:
                    raise ValueError("An affected cut to the errors had different values in the dict for the values.")
                
            cleaned_dict[k] = v
        
        return cleaned_dict