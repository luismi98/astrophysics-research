import numpy as np
import os

def prevent_byteorder_error(df):
    for column in df.columns:
        if hasattr(df[column], 'dtype') and df[column].dtype.byteorder == '>':
            df[column] = df[column].values.newbyteorder().byteswap()

def is_grey(rgba):
    is_grey = True
    for i in range(3):
        is_grey *= abs(rgba[i] - 0.86) < 0.02
    return is_grey

def moving_average(bins,window_size=2):
    return np.convolve(bins, np.ones(window_size)/window_size, mode='valid')

def in_circle(x, y, center_x, center_y, radius):
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2

def map_array_from_txt(map_values_string):
    '''
    Takes a map array txt as single string, e.g. map_txt = "nan 1.716 -35.676 22.508 -41.901 -15.329 -44.852 -41.731"
    Returns a 2D map_array with float types
    '''
    map_array = np.array(map_values_string.split(" "), dtype=float)
    
    dimension_size = int(np.sqrt(len(map_array)))
    map_array = map_array.reshape((dimension_size,dimension_size))
    return map_array

def get_number_decimals(x):
    x_string = str(x)
    after_dot = 0
    count_bool = False
    for i in range(len(x_string)):
        if count_bool:
            after_dot += 1
        if x_string[i] == '.':
            count_bool = True
    return after_dot

def format_number_with_commas(number):
    number_str = str(number)
    
    if "," in number_str:
        raise ValueError(f"The given number `{number_str}` is already formatted using commas!")
    
    if '.' in number_str:
        number_str = number_str.split('.')[0]
    
    if len(number_str) <= 3:
        return str(number_str)
    
    number_str_commas = ""
    
    for i,digit in enumerate(number_str[::-1]):
        if i != 0 and i % 3 == 0:
            number_str_commas = "," + number_str_commas
        
        number_str_commas = digit + number_str_commas
        
    return number_str_commas

def mc_perturb(data, error):
    #Data should be something like the v_r component of stars in an ellipse distribution. Each with an associated uncertainty in the error array
    #It returns an array of the same shape, where the elements are numbers drawn from a gaussian distribution centered in the datapoint, with standard deviation of the associated error
    return np.random.normal(data, error)

def create_dir(save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        print("Created successfully")
        
def check_int(x):
    return int(x) if int(x) == x else x
    
def flatten_list(x):
    return list(np.array(x).flat)

def string_to_float(x):
    if not x[0].isdigit():
        return -float(x[1:])
    else:
        return float(x)
    
def return_int_or_dec_for_array(array,dec=2,extra_dec=0):
    return np.array([return_int_or_dec(val,dec,extra_dec) for val in array])

def return_int_or_dec(val, dec=1, extra_dec=0):
    if val == 0: return val
    
    rounded_val = round(val,dec)
    while rounded_val == 0:
        dec += 1
        rounded_val = round(val,dec)
        
    if extra_dec > 0:
        dec += extra_dec
        rounded_val = round(val,dec)
    
    return check_int(rounded_val)

def is_negative(value):
    return not abs(value) == value

def get_sign_string(value):
    return r"$-$" if is_negative(value) else ""
    
def get_exponential(x):
    for index, i in enumerate(str(x)):
        if i == '.':
            break
    if index < 2:
        return x
    result_str = fr"${str(x)[0]}\cdot 10^{index}$" if str(x)[0] != '1' else fr"$10^{index}$"
    return result_str

def get_exponent(x):
    for index, i in enumerate(str(x)):
        if i == '.':
            return index - 1
    return index

def get_gaussian(x, mean, std):
    return np.exp(-(x-mean)**2 / (2 * std**2)) / np.sqrt(2 * np.pi * std**2)

def get_mean_and_std(x, gaussian):
    mean = x[np.where(gaussian == gaussian.max())][0]
    
    half_max = gaussian.max() / 2
    for i in range(len(x) - 1):
        if gaussian[i] < half_max and gaussian[i+1] > half_max:
            break
    half_max_x = (x[i] + x[i+1]) / 2
    FWHM = 2 * (mean - half_max_x)

    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    
    return mean, sigma

def round_one_significant(x):
    x_string = str(x) if x>0 else str(x)[1:]

    for index, i in enumerate(x_string):
        if i == '.':
            index -= 1
            break
    first_digit = int(x_string[0])

    if len(x_string) > 1:
        second_digit = int(x_string[1]) if x_string[1] != '.' else int(x_string[2])
        if second_digit >= 5:
            first_digit += 1
    else:
        if first_digit >= 5:
            first_digit = 1
            index += 1

    number = first_digit * 10**index
    return number * (1 if x>0 else -1)

def merge_dictionaries(dict_list):

    all_keys = [k for d in dict_list for k in d.keys()]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("There were duplicated keys across the given dictionaries. Please use unique keys.")

    return {k:v for d in dict_list for k,v in d.items()}

def apply_cuts_to_df(df,cuts_dict,lims_dict=None):
    """
    Apply cuts to dataframe.
    
    Parameters
    ----------
    df: pandas dataframe
        The dataframe to apply the selection on.
    cuts_dict: dictionary, or list of dictionaries
        Key value-pairs: string (variable) and list (of [min,max] values to use as selection).
    lims_dict: dictionary, or list of dictionaries
        Key-value pairs: string (variable) and string (defining the limits to include in the selection: "neither", "min", "max" or both").
        If None, defaults to "both" for all cuts.
    """
    
    if isinstance(cuts_dict,list):
        cuts_dict = merge_dictionaries(cuts_dict)
    if isinstance(lims_dict,list):
        lims_dict = merge_dictionaries(lims_dict)

    if lims_dict is None:
        lims_dict = {k:"both" for k in cuts_dict.keys()}
    
    assert cuts_dict.keys() == lims_dict.keys(), "The keys should be equal for the cuts and limits dictionaries."
        
    for key in cuts_dict.keys():
        assert key in df, f"`{key}` is not a valid key in the dataframe."

        cuts,lims = cuts_dict[key],lims_dict[key]
        
        minimum,maximum = cuts[0],cuts[1]
        
        if lims == "neither":
            df = df[(df[key]>minimum)&(df[key]<maximum)]
        elif lims == "min":
            df = df[(df[key]>=minimum)&(df[key]<maximum)]
        elif lims == "max":
            df = df[(df[key]>minimum)&(df[key]<=maximum)]
        elif lims == "both":
            df = df[(df[key]>=minimum)&(df[key]<=maximum)]
        else:
            raise ValueError(f"`{lims}` is not a valid limit. Use 'neither', 'min', 'max' or 'both'.")
            
    return df

def print_nested_dict_recursive(dictionary):
    if not isinstance(dictionary,dict):
        print(dictionary)
        return
    
    for key in dictionary.keys():
        print(key)
        print_nested_dict_recursive(dictionary[key])

def print_nested_dict_iterative(nested_dict, all_key_tracks, print_limit=None):
    print_count = 0
    
    for key_track in all_key_tracks:
        lower_level_dict = nested_dict
        
        *keys, last_key = key_track # unpack all but the last key
        
        for key in keys:
            print(key)
            lower_level_dict = lower_level_dict[key] # we are setting up references to the overall_map_dict elements
            
        print(last_key)
        
        if isinstance(lower_level_dict[last_key], dict):
            for final_key in lower_level_dict[last_key]:
                print(f"'{final_key}':",lower_level_dict[last_key][final_key])
        else:
            print(lower_level_dict[last_key])
    
        print_count += 1
        
        if print_count == print_limit:
            if print_limit < len(all_key_tracks):
                print("(...)")
            return
        
def get_anti_diagonal(arr):
    """
    Get the ani-diagonal elements of a 2D array
    """
    
    return np.fliplr(arr).diagonal()

def extract_str_from_cuts_dict(cuts_dict):
    s = ""
    
    for k in cuts_dict:
        s += f"{return_int_or_dec(cuts_dict[k][0],2)}{k}{return_int_or_dec(cuts_dict[k][1],2)}_"
        
    return s.strip("_")