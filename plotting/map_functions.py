import numpy as np
from scipy.ndimage import gaussian_filter

import utils.miscellaneous_functions as MF

def get_map_string_lists(fractional_errors=False):
    full_map_string_list = ["number",\
                            "mean_vx",              "mean_vx_error_low",            "mean_vx_error_high",\
                            "mean_vy",              "mean_vy_error_low",            "mean_vy_error_high",\
                            "std_vx",               "std_vx_error_low",             "std_vx_error_high",\
                            "std_vy",               "std_vy_error_low",             "std_vy_error_high",\
                            "anisotropy",           "anisotropy_error_low",         "anisotropy_error_high",\
                            "correlation",          "correlation_error_low",        "correlation_error_high",\
                            "tilt_abs",             "tilt_abs_error_low",           "tilt_abs_error_high", \
                            "tilt",                 "tilt_error_low",               "tilt_error_high", \
                            "spherical_tilt",       "spherical_tilt_error_low",     "spherical_tilt_error_high",
                            "abs_spherical_tilt",   "abs_spherical_tilt_error_low", "abs_spherical_tilt_error_high"]

    if fractional_errors:
        for map_string in full_map_string_list:
            if "error" in map_string and "fractionalerror" not in map_string:
                full_map_string_list.append(map_string.split("error")[0] + "fractionalerror" + map_string.split("error")[1])

    divergent_map_list = ["mean_vx","mean_vy","anisotropy","correlation","tilt","tilt_abs","spherical_tilt"]

    return full_map_string_list, divergent_map_list

####################################################################################################

def get_kinematic_label(map_string, kinematic_symbol_dict, kinematic_units_dict):
    label = kinematic_symbol_dict[map_string]
    
    if kinematic_units_dict[map_string] != "":
        label += r"$~[%s]$"%kinematic_units_dict[map_string]
        
    return label

def get_symbol(var):
    if var in ["d",]:
        return r"$%s$"%var
    elif var == "FeH":
        return "[Fe/H]"
    elif var == "vr":
        return r"$v_r$"
    elif var == "pmra":
        return r"$\mu_{\alpha}$"
    elif var == "pmdec":
        return r"$\mu_{\delta}$"
    elif "_error" in var:
        return r"ϵ$($"+get_symbol(var.removesuffix("_error"))+r"$)$"
    elif "_fractionalerror" in var:
        var_symbol = get_symbol(var.removesuffix("_fractionalerror")).removeprefix('$').removesuffix('$')
        return r"ϵ$($" + var_symbol + r"$)/|$" + var_symbol + r"$|$"
    else:
        raise ValueError(f"Variable `{var}` not recognised")
    
def get_units(var):
    if var in ["d","R"]:
        return "kpc"
    elif var in ["FeH","correlation","anisotropy"]:
        return ""
    elif var in ["pmra","pmdec","pmlcosb","pml","pmb"]:
        return "mas/yr"
    elif var in ["vr","vl","vb","vx","vy","vz","vR","vphi","vM","vm"]:
        return "km/s"
    elif var in ["tilt_abs","tilt","vertex","vertex_abs"]:
        return r"$^\circ$"
    elif "_error" in var:
        return get_units(var.removesuffix("_error"))
    elif "_fractionalerror" in var:
        return ""
    else:
        raise ValueError(f"Variable `{var}` not recognised")

def get_kinematic_symbols_dict(vel_x_variable="r",vel_y_variable="l",x_variable="l",y_variable="b",diff=False):
    """
    Get dictionaries for the kinematic variable's symbols.

    Parameters
    ----------
    vel_x_variable: string
        Horizontal velocity component.
    vel_y_variable: string
        Vertical velocity component.
    x_variable: string
        Horizontal position component. Needed to get the spherical tilt symbol.
    y_variable: string
        Vertical position component.
    diff: bool
        Whether you want the dictionary to contain the symbols for the kinematic variable differences (for difmaps).

    Returns
    -------
    dict
        Dictionary of kinematic symbols
    """

    def get_tilt_string(vel_string,absolute=True):
        # 11/01/2023 - I HAVE SWITCHED AROUND THE SYMBOLS
        if vel_string == "rl":
            return "\\tilde{l}_{\mathrm{v}}" if not absolute else "l_{\mathrm{v}}"
        else:
            return "\\tilde{l}_{\mathrm{v}}^{%s}"%(vel_string) if not absolute else "l_{\mathrm{v}}^{%s}"%(vel_string)

    spherical_tilt_vel_x = 'R' if x_variable+y_variable == 'xy' else x_variable
    spherical_tilt_vel_y = "\phi" if x_variable+y_variable == 'xy' else y_variable
        
    kinematic_symbol_dict = {
        "mean_vx" : r"$\langle v_{%s} \rangle$"%vel_x_variable,
        "mean_vy" : r"$\langle v_{%s} \rangle$"%vel_y_variable,
        "anisotropy" : r"$\beta_{%s}$"%(vel_x_variable+vel_y_variable),
        "std_vx" : r'$\sigma_{%s}$'%vel_x_variable,
        "std_vy" : r'$\sigma_{%s}$'%vel_y_variable,
        "tilt" : r"$%s$"%get_tilt_string(vel_x_variable+vel_y_variable,absolute=False),
        "tilt_abs" : r"$%s$"%get_tilt_string(vel_x_variable+vel_y_variable),
        "spherical_tilt" : r"$%s$"%get_tilt_string(spherical_tilt_vel_x+spherical_tilt_vel_y, absolute=False),
        "abs_spherical_tilt" : r"$\left|%s\right|$"%get_tilt_string(spherical_tilt_vel_x+spherical_tilt_vel_y, absolute=False),
        "correlation" : r"$\rho_{%s %s}$"%(vel_x_variable,vel_y_variable)
    }

    kinematic_variables_for_error = ["tilt","tilt_abs","spherical_tilt","anisotropy","correlation"]
    for map_string in kinematic_variables_for_error:
        variable_symbol = kinematic_symbol_dict[map_string].replace('$','')
        kinematic_symbol_dict[map_string+"_error"] = r"ϵ$(%s)$"%variable_symbol # using the symbol directly as it doesn't detect \upepsilon
        kinematic_symbol_dict[map_string+"_error_low"] = kinematic_symbol_dict[map_string+"_error"]
        kinematic_symbol_dict[map_string+"_error_high"] = kinematic_symbol_dict[map_string+"_error"]
        kinematic_symbol_dict[map_string+"_fractionalerror"] = r"ϵ$(%s)/|%s|$"%(variable_symbol,variable_symbol)
    
    if diff:
        kinematic_symbol_diff_dict = {}
        for symbol in list(kinematic_symbol_dict.keys()):
            variable_symbol = kinematic_symbol_dict[symbol].replace('$','')
            kinematic_symbol_diff_dict[symbol] = r"$\Delta %s$"%variable_symbol
        
        return kinematic_symbol_diff_dict
    else:
        return kinematic_symbol_dict 

def get_kinematic_units_dict(degree_symbol = '^\circ'):
    """
    Get dictionary for all the kinematic variable's units.

    Parameters
    ----------
    degree_symbol: string
        Which symbol to use for degrees.

    Returns
    -------
    dict
        Dictionary of kinematic variable's units
    """

    kinematic_units_dict = {}
    for angular_variable in ["tilt","tilt_abs","spherical_tilt","abs_spherical_tilt"]:
        kinematic_units_dict[angular_variable] = degree_symbol
    for vel_variable in ['mean_vx','mean_vy',"std_vx","std_vy"]:
        kinematic_units_dict[vel_variable] = "\mathrm{km ~ s^{-1}}"
    for unitless_variable in ["anisotropy","correlation"]:
        kinematic_units_dict[unitless_variable] = ''

    variables_for_error = ["tilt","tilt_abs","spherical_tilt","anisotropy","correlation"]
    for map_string in variables_for_error:
        kinematic_units_dict[map_string+"_error"] = kinematic_units_dict[map_string]
        kinematic_units_dict[map_string+"_error_low"] = kinematic_units_dict[map_string]
        kinematic_units_dict[map_string+"_error_high"] = kinematic_units_dict[map_string]
        kinematic_units_dict[map_string+"_fractionalerror"] = ''

    return kinematic_units_dict

def get_kinematic_titles_dict():
    """
    Get dictionaries for plot titles referring to full kinematic variable names, with uppercase initial.

    Parameters
    ----------
    vel_x_variable: string
        Horizontal velocity component.
    vel_y_variable: string
        Vertical velocity component.

    Returns
    -------
    dict
        Dictionary of titles
    """

    title_dict = {
        "mean_vx" : f"Mean velocity",
        "mean_vy" : f"Mean velocity",
        "anisotropy" : "Anisotropy",
        "anisotropy_error" : "Anisotropy error",
        "std_x" : "Standard deviation",
        "std_y" : "Standard deviation",
        "tilt": "Vertex deviation",
        "tilt_abs": "Vertex deviation",
        "spherical_tilt": "Spherical tilt",
        "abs_spherical_tilt": "Spherical tilt absolute",
        "vertex" : "Vertex deviation",
        "vertex_error" : "Vertex deviation error",
        "vertex_abs" : "Vertex deviation abs",
        "vertex_abs_error" : "Error in vertex deviation abs",
        "number": "Number of stars",
        "correlation" : "Correlation",
        "correlation_error" : "Correlation error"
    }
    title_dict["tilt_error"] = title_dict["tilt"]+" error"
    title_dict["tilt_abs_error"] = title_dict["tilt_abs"]+" error"

    return title_dict

def get_position_symbols_and_units_dict(zabs = True, degree_symbol = "^\circ"):
    """
    Get dictionaries for the symbols and units of all the position variables.

    Parameters
    ----------
    zabs: bool, default is True
        Whether the galaxy was mirrored above the plane, meaning b and z are now |b| and |z|.
    degree_symbol: string
        Which symbol to use for degrees.

    Returns
    -------
    tuple of dicts:
        A dictionary of symbols and another of units
    """

    symbol_dict = {
        "l" : r"$l$",
        "b" : r"$b$" if not zabs else r"$|b|$",
        "d" : r"$d$",
        "x" : r"$x$",
        "y" : r"$y$",
        "z" : r"$z$" if not zabs else r"$|z|$",
        "R" : r"$R_\mathrm{GC}$",
        "phi" : r"$\phi$"
    }
    
    units_dict = {
        "l" : degree_symbol,
        "b" : degree_symbol,
        "d" : "kpc",
        "x" : "kpc",
        "y" : "kpc",
        "z" : "kpc",
        "R" : "kpc",
        "phi" : degree_symbol
    }

    return symbol_dict, units_dict

####################################################################################################

def get_map_tick_range(vmin,vmax,step,include_lims=True,verbose=False):

    max_tick = vmax//step * step
    pos_ticks = np.arange(0, max_tick+step, step)
    
    min_tick = np.abs(vmin)//step * step # Need the absolute value otherwise the floor division gives weird stuff (eg -2//10 gives -1)
    min_tick *= -1 if vmin < 0 else 1
    neg_ticks = np.arange(min_tick,0,step)
    
    ticks = np.concatenate([neg_ticks,pos_ticks])
    
    if max_tick == vmax and not include_lims:
        ticks = ticks[:-1]
    if min_tick == vmin and not include_lims:
        ticks = ticks[1:]
    
    if verbose:
        print(f"vmin:{vmin}\nvmax:{vmax}\nstep:{step}\nmax_tick:{max_tick}\npos_ticks:{pos_ticks}\nmin_tick:{min_tick}\nneg_ticks{neg_ticks}\nall_ticks{ticks}")

    return ticks.tolist()

####################################################################################################
# vminvmax and colorbar sharing

def any_map_pair_is_shared(variable_list, shared_cbar_variables):
    """
    Use it to know if in the current plot (specified by `variable_list`) there are maps whose cbar is being shared
    """
    all_maps = MF.flatten_list(variable_list)
    for shared_maps in shared_cbar_variables:
        if shared_maps[0] in all_maps and shared_maps[1] in all_maps:
            return True
    return False

def share_vminvmax_given_vminvmax_lists(map_variables, shared_cbar_variables, vmin_list, vmax_list):
    """
    Currently used in the observational maps, where I compute the vmin and vmax lists beforehand so I can choose appropriate norms for each row

    Note: `map_variables` should be a 1-D list (typically the maps in a given block, with an entry per row)

    Also note: it modifies the vmin and vmax lists in-place.
    """

    for shared_vars in shared_cbar_variables:
        try:
            first_idx = map_variables.index(shared_vars[0])
            second_idx = map_variables.index(shared_vars[1])
            
            first_vmin = vmin_list[first_idx]
            second_vmin = vmin_list[second_idx]            
            vmin_list[first_idx],vmin_list[second_idx] = min([first_vmin,second_vmin]),min([first_vmin,second_vmin])
            
            first_vmax = vmax_list[first_idx]
            second_vmax = vmax_list[second_idx]
            vmax_list[first_idx],vmax_list[second_idx] = max([first_vmax,second_vmax]),max([first_vmax,second_vmax])
            
            print("Shared",shared_vars)
            
        except ValueError:
            continue

def get_vminvmax_from_map_dict(map_dict, map_variable, shared_cbar_variables=None, hardcode=False):
    """
    Currently used in the heatmaps for the simulation, where map_dict contains the values of all populations for each map_variable.
    
    It does not affect the contour plots because the levels are chosen manually so the vmin and vmax values have no effect.
    """

    if hardcode:
        if "fractionalerror" in map_variable:
            return 0,1
    
    if shared_cbar_variables is not None:
        for shared_vars in shared_cbar_variables:
            if map_variable in shared_vars:
                
                vmin,vmax=9999,-9999
                for shared in shared_vars:
                    vmin = min(vmin,np.nanmin(map_dict[shared]))
                    vmax = max(vmax,np.nanmax(map_dict[shared]))
                
                return vmin,vmax
                
    vmin = np.nanmin(map_dict[map_variable])
    vmax = np.nanmax(map_dict[map_variable])
    
    return vmin,vmax

#####################################################################################################
# Removing overlapping tick labels

def get_ticks(cax, vmin, vmax, axis="y"):
    """
    Unless the lowest and highest tick are exactly equal to the vmin and vmax, the ticks will go under and over by 1 (though they are hidden)
    """
    
    cbar_ticks = cax.get_yticks() if axis == "y" else cax.get_xticks()
    
    if cbar_ticks[0] != vmin:
        cbar_ticks = cbar_ticks[1:]
    if cbar_ticks[-1] != vmax:
        cbar_ticks = cbar_ticks[:-1]
    
    return cbar_ticks

def lowest_tick_is_low(ticks, vmin,vmax, frac=15):
    last_tick = ticks[0] # ax.get_yticks() (or x) gives the tick order from bottom to top
    return abs(vmin - last_tick) <= abs(vmax-vmin)/frac
    
def highest_tick_is_high(ticks, vmin,vmax, frac=15):
    first_tick = ticks[-1]
    return abs(vmax-first_tick) <= abs(vmax-vmin)/frac

def remove_ticklabel(ax, ticks, axis="y",which="bottom"):
    if which not in ["bottom","top"]:
        raise ValueError("`which` must be 'bottom' or 'top'.")

    ax.set_yticks(ticks) if axis == "y" else ax.set_xticks(ticks)
    
    slicing = slice(1,None) if which == "bottom" else slice(None,-1)

    formatted_labels = ["%s"%(MF.return_int_or_dec(tick) if str(tick)[0].isdigit() else "−%s"%(str(MF.return_int_or_dec(tick))[1:]))
                        for tick in ticks[slicing]]
    
    formatted_labels = [""]+formatted_labels if which == "bottom" else formatted_labels+[""]
    
    # Check if there are already any labels set to ""
    tick_labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    current_labels = np.array([lab.get_text() for lab in tick_labels])
    formatted_labels = np.array(formatted_labels)
    formatted_labels[current_labels == ""] = ""
    
    ax.set_yticklabels(formatted_labels) if axis == "y" else ax.set_xticklabels(formatted_labels)

def remove_overlapping_ticks(ax, next_ax, vminvmax, next_vminvmax, axis="y", which="both", frac=15):
    if which not in ["bottom","top","both"]:
        raise ValueError("`which` must be 'bottom', 'top' or 'both'.")

    vmin,vmax = vminvmax[0],vminvmax[1]
    next_vmin, next_vmax = next_vminvmax[0],next_vminvmax[1]
    
    ticks = get_ticks(ax, vmin, vmax, axis=axis)
    next_ticks = get_ticks(next_ax, next_vmin, next_vmax, axis=axis)

    if lowest_tick_is_low(ticks,vmin,vmax,frac=frac) and highest_tick_is_high(next_ticks,next_vmin,next_vmax,frac=frac):
        if which in ["bottom","both"]:
            remove_ticklabel(ax, ticks, axis=axis, which="bottom")
        if which in ["top","both"]:
            remove_ticklabel(next_ax, next_ticks, axis=axis, which="top")

###################################################################################################################

def get_2d_hist_counts(df, x_variable, y_variable, bins, extent, stellar_mass=9.5*10**3, gauss_sigma=1, density=True, mass_density_bool=True):
    """
    Parameters
    ----------
    stellar_mass: float. 
        Mass of each particle, needed if wanting to compute a mass density.
        Default is 9.5*10**3. See bottom left of page 8 in Debattista 2017.
    """

    counts,_,_ = np.histogram2d(df[x_variable],df[y_variable],bins=bins,range=extent,density=density)
    counts = counts.T*len(df) #Multiply by total number to get a surface density rather than probability density
    if mass_density_bool:
        counts *= stellar_mass
    
    return gaussian_filter(counts, gauss_sigma)