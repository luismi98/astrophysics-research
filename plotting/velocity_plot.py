import warnings
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.compute_variables as CV
import src.bootstrap_errors as bootstrap
from src.errorconfig import BootstrapConfig
import plotting.plotting_helpers as PH
import plotting.map_functions as mapf
import utils.error_helpers as EH

def velocity_plot(vx, vy, ax=None, bins=30, vel_x_var=None, vel_y_var=None, vel_lims=None, bootstrap_repeats=500, ellipse_bool=True,
                  contours_bool=True, contour_kws=None, contour_smoothing=1, seaborn_contours=False, contour_cbar_bool=False, 
                  c=None, scatter_colour_cbar=False, scatter_colour_var = None, scatter_cmap = "coolwarm",
                  text_bool=False, variables_and_plot_positions=None, text_kws=None, axis_labels="neither"):

    vx = np.array(vx)
    vy = np.array(vy)

    if len(vx) != len(vy):
        raise ValueError("`vx` and `vy` must have the same length!")
    if len(vx) == 0:
        raise ValueError("There are no stars to plot...")
    
    if ax is None:
        _, ax = plt.subplots()
        
    if c is None:
        ax.scatter(vx, vy, marker='.', color='grey')
    else:
        scat = ax.scatter(vx,vy,marker='.',s=15,c=c,cmap=scatter_cmap)
        
        if scatter_colour_cbar:
            divider = make_axes_locatable(ax)
            scatter_cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar_colour = plt.colorbar(scat,cax=scatter_cax)
            cbar_colour.set_label(label=f"{mapf.get_symbol(scatter_colour_var)} [{mapf.get_units(scatter_colour_var)}]")
    
    if contours_bool:
        default_contour_kws = {"alpha":0.7, "levels": 7, "cmap": "coolwarm"}
        if contour_kws is None:
            contour_kws = {}
        for kw in default_contour_kws:
            if kw not in contour_kws:
                contour_kws[kw] = default_contour_kws[kw]

        if contour_cbar_bool:
            divider = make_axes_locatable(ax)
            contour_cax = divider.append_axes("right", size="5%", pad=0.05)

        if seaborn_contours:
            cbar_label = r'Probability density [$\rm s^{2} \hspace{0.3} km^{-2}$]'

            cut_lim = 30

            sns.kdeplot(vx, vy, fill=True, shade_lowest=True, cut=cut_lim, ax=ax, **contour_kws,
                        cbar=contour_cbar_bool, cbar_ax = contour_cax, cbar_kws={'label': cbar_label})

            sns.kdeplot(vx, vy, fill=False, shade_lowest=True, cut=cut_lim, alpha=1, ax=ax, linewidths=2, **contour_kws,
                        cbar=contour_cbar_bool, cbar_ax=contour_cax,cbar_kws={'label': cbar_label})
        else:

            if vel_lims is None:
                vel_lims = [[-400,400],[-400,400]]

            h = np.histogram2d(vx,vy,bins=bins,range=vel_lims)
            cont = ax.contourf(gaussian_filter(h[0].T,contour_smoothing), **contour_kws, 
                               extent=[vel_lims[0][0],vel_lims[0][1],vel_lims[1][0],vel_lims[1][1]])

            if contour_cbar_bool:
                contour_cbar_bool = plt.colorbar(cont,cax=contour_cax)
                contour_cbar_bool.set_label(label=r"$N$",labelpad=20,rotation=0)

    if ellipse_bool:
        plot_ellipse(ax,vx,vy)

    if axis_labels in ["x","both"]:
        if vel_x_var is None:
            warnings.warn("Cannot show the xlabel if `vel_vx_var` is None.")
        else:
            ax.set_xlabel(r"$v_%s$ [km $\rm s^{-1}$]"%vel_x_var)
    if axis_labels in ["y","both"]:
        if vel_y_var is None:
            warnings.warn("Cannot show the ylabel if `vel_vy_var` is None.")
        else:
            ax.set_ylabel(r"$v_%s$ [km $\rm s^{-1}$]"%(vel_y_var if vel_y_var!="phi" else "\phi"))
    
    if text_bool:
        
        if variables_and_plot_positions is None:
            variables_and_plot_positions = {
                "number": [0.73, 0.97],
                "anisotropy": [0.025, 0.09],
                "correlation": [0.53, 0.09],
                "tilt_abs": [0.025, 0.97],
            }

        show_variable_values_text(ax=ax,vel_x_var=vel_x_var,vel_y_var=vel_y_var,vx=vx,vy=vy,variables_and_plot_positions=variables_and_plot_positions,
                                  bootstrap_repeats=bootstrap_repeats, kws=text_kws)
    
    return ax

def show_variable_values_text(ax, vel_x_var, vel_y_var, vx, vy, variables_and_plot_positions, bootstrap_repeats, kws):

    def build_text_kws(text_kws):
        default_textbox_kws = dict(boxstyle='round', facecolor='linen', alpha=1)
        default_text_kws = dict(fontsize="small", verticalalignment='top', bbox=default_textbox_kws)

        if text_kws is None:
            text_kws = {}
        
        for kw in default_text_kws:
            if kw not in text_kws:
                text_kws[kw] = default_text_kws[kw]
        for box_kw in default_textbox_kws:
            if box_kw not in text_kws["bbox"]:
                text_kws["bbox"][box_kw] = default_textbox_kws[box_kw]

        return text_kws
    
    kws = build_text_kws(text_kws=kws)

    bootstrapconfig = BootstrapConfig(repeats=bootstrap_repeats,symmetric=True)
    kinematic_symbols_dict = mapf.get_kinematic_symbols_dict(vel_x_variable=vel_x_var,vel_y_variable=vel_y_var)
    kinematic_units_dict = mapf.get_kinematic_units_dict()

    for variable in variables_and_plot_positions:
        if "spherical" in variable:
            raise ValueError("Did not expect a spherical tilt. It needs information about the bin edges.")
        
        x,y = variables_and_plot_positions[variable]

        if variable == "number":
            ax.text(x,y,fr"$N={len(vx)}$",transform=ax.transAxes,**kws)
            continue

        symbol,units = kinematic_symbols_dict[variable], kinematic_units_dict[variable]

        func, vx, vy, R_hat, tilt, absolute = EH.get_function_parameters(variable, vx=vx, vy=vy)

        value = EH.apply_function(function=func, vx=vx,vy=vy,tilt=tilt,absolute=absolute,R_hat=R_hat)
        error = bootstrap.get_std_bootstrap(function=func,config=bootstrapconfig, vx=vx,vy=vy,tilt=tilt,absolute=absolute).confidence_interval[0]

        text = fr"{symbol}$={value:.2f} \pm {error:.2f} {units}$"

        ax.text(x,y,text,transform=ax.transAxes,**kws)

def plot_ellipse(ax,vx,vy,radius_factor=2,max_vector_factor=2.5):

    cov = np.cov(vx, vy)
    eigenvalues = np.linalg.eig(cov)[0]
    radius = radius_factor*np.sqrt(max(eigenvalues))
    ratio = np.sqrt(min(eigenvalues)/max(eigenvalues))
    centre = [np.mean(vx), np.mean(vy)]
    x_ellipse, y_ellipse = PH.get_ellipse_coords(radius, ratio, CV.calculate_tilt(vx,vy,absolute=False))
    ax.plot(x_ellipse+centre[0], y_ellipse+centre[1], lw=3,color='red')
    eigvectors = np.linalg.eig(cov)[1]
    raw_max_vector = eigvectors[:,eigenvalues.argmax()]
    max_vector = raw_max_vector*max_vector_factor*np.sqrt(max(eigenvalues))

    vector_plot_data = [[max_vector[0], -max_vector[0]],[max_vector[1], -max_vector[1]]]
    vector_plot_data[0] += centre[0]
    vector_plot_data[1] += centre[1]
    ax.plot(vector_plot_data[0],vector_plot_data[1], lw=3, color='red')