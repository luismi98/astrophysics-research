import numpy as np
import os
import json
import scipy.stats as stats

import src.compute_variables as CV

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
    
def get_exponential_form(x):
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

def any_duplicate_dict_keys(dict_list):
    all_keys = [k for d in dict_list for k in d.keys()]
    return len(all_keys) != len(set(all_keys))

def merge_dictionaries(dict_list):
    if any_duplicate_dict_keys(dict_list):
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
        Defaults to "both" for any cut whose lim is not specified.
    """
    
    if isinstance(cuts_dict,list):
        cuts_dict = merge_dictionaries(cuts_dict)
    if isinstance(lims_dict,list):
        lims_dict = merge_dictionaries(lims_dict)

    if lims_dict is None:
        lims_dict = {} # initialise to empty here because of the issue with mutable default arguments - https://stackoverflow.com/questions/9158294

    for k in cuts_dict:
        if k not in lims_dict:
            lims_dict[k] = "both"
    
    assert cuts_dict.keys() == lims_dict.keys(), f"The keys should be equal for the cuts and limits dictionaries but were `{list(cuts_dict.keys())}` and `{list(lims_dict.keys())}` respectively."
        
    for key in cuts_dict.keys():
        if key not in df: raise ValueError(f"`{key}` is not a valid key in the dataframe.")

        cuts,lims = cuts_dict[key],lims_dict[key]
        
        minimum,maximum = cuts[0],cuts[1]
        
        condition = build_lessgtr_condition(df[key],minimum,maximum,include=lims)

        df = df[condition]

    return df

def build_lessgtr_condition(array,low,high,include="both"):
    
    if include not in ["neither","min","max","both"]:
        raise ValueError(f"`{include}` is not a valid limit. Use 'neither', 'min', 'max' or 'both'.")

    lower_end = array>=low if include in ["min","both"] else array>low
    higher_end = array<=high if include in ["max","both"] else array<high

    return lower_end&higher_end

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

def save_dic_as_json(dic, filename):
    with open(filename+".json", 'w') as f:
        f.write(json.dumps(dic))
        
def load_dic_from_json(filename):
    with open(filename) as f:
        dic = json.loads(f.read())
    return dic

def clean_cuts_from_dict(cuts_dict, cuts_to_remove):
    """
    Return the provided dictionary after removing the given cuts, if any.
    """

    if type(cuts_dict) != dict:
        cuts_dict = merge_dictionaries(cuts_dict)
    
    cleaned_dict = {}
    
    for k,v in cuts_dict.items():
        if k in cuts_to_remove:
            if v == cuts_to_remove[k]:
                continue
            else:
                raise ValueError("A cut to remove had different values in the full dict.")
            
        cleaned_dict[k] = v
    
    return cleaned_dict

def combine_multiple_cut_dicts_into_str(all_cuts, cut_separator="_", order_separator="/"):
    """
    Given a dict or list of dicts containing spatial cuts, it builds a string like so:
    f"<height cut>/<depth cut>/<width cut>/<pop cut>" (if any), with the separator given by order_separator,
    and where: 
    - height is "b" or "z"
    - depth is "R", "d" or "x"
    - width is "l" or "y"
    - pop is "age" or "FeH"
    If multiple cuts of a given order are given, they are separated by cut_separator.
    """
    
    def add_path_parts(cut, orders, path_parts):
        for key, ranges in cut.items():
            for order, keys in orders.items():
                if key in keys:
                    range_string = f"{ranges[0]}{key}{ranges[1]}"
                    path_parts[order].add(range_string)
                    break
    
    orders = {
        0: ["b", "z"],
        1: ["d", "R", "x"],
        2: ["l", "y"],
        3: ["age", "FeH"]
    }
    
    path_parts = {o: set() for o in orders}
    
    if type(all_cuts) == list:
        for cut in all_cuts:
            add_path_parts(cut, orders, path_parts)
    else:
        add_path_parts(all_cuts, orders, path_parts)
    
    segments = []
    for order in sorted(path_parts):
        if path_parts[order]:
            segments.append(cut_separator.join(sorted(path_parts[order])))
    
    full_str = order_separator.join(segments)
    
    return full_str

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