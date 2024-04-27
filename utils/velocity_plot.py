import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import compute_variables as CV
import compute_errors as CE
import miscellaneous_functions as MF
import plotting_helpers as PH

# This function is a monstruosity I wrote long ago. When I have some time I will break it down into pieces
def velocity_plot(df,vx_component,vy_component, ax=None,vel_lims=[[-400,400],[-400,400]], bootstrap_repeat=500, tilt_abs=False,bins=30, cmap='coolwarm', alpha=0.7, smoothing=-1, contour_levels=7, 
                  seaborn=False, cbar=False, contours_bool=True,colour_var = None, c=None,colour_cmap='viridis', tickstep=100, size_ticks=20, 
                  size_axislabels=25, size_variables=20, size_title=20, title_str= '', population_string = '', title_limits=None, fileindex=None, 
                  show=True, save_path="/Users/Luismi/Desktop/MRes_UCLan/", save=False, dpi=200, fileformat='.png', filename=None, return_dict = False):
    
    vx,vy = df[f"v{vx_component}"].values,df[f"v{vy_component}"].values

    star_number = len(vx)
    if star_number == 0: raise ValueError("There are no stars to plot...")
    if np.sum(np.isnan(vx)) or np.sum(np.isnan(vy)): raise ValueError("There are some NaNs in the velocities, get rid of them first")
    
#------------------------------------------------------------------------------------------------------------------------
#---- PLOT --------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
    
    if smoothing == -1: # default
        smoothing = 1 if star_number > 300 else 2
        print("Using smoothing",smoothing)
    elif smoothing == -2:
        smoothing = 2 if star_number > 300 else 3
        print("Using smoothing",smoothing)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    if colour_var is None:
        ax.scatter(vx, vy, marker='.', color='grey')
    else:
        if c is None:
            raise ValueError("`colouring` is set to True but no colours were provided in the `c` variable")
        scat = ax.scatter(vx,vy,marker='.',s=15,c=c,cmap=colour_cmap)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cbar_label = {
            'd': r"$d$ [kpc]",
            'l': r"$l$ $[^\circ]$",
            "phi": r"$\phi$ $[^\circ]$",
            "R": r"$R$ [kpc]",
        }
        
        cbar_colour = plt.colorbar(scat,cax=cax)
        cbar_colour.set_label(label=cbar_label[colour_var])
    
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if contours_bool:
        if seaborn:
            cbar_label = r'Probability density [$\rm s^{2} \hspace{0.3} km^{-2}$]'

            cut_lim = 30

            sns.kdeplot(vx, vy, cmap=cmap, fill=True, shade_lowest=True, cut=cut_lim, \
                    alpha=alpha, ax=ax, cbar=cbar, cbar_ax = cax, cbar_kws={'label': cbar_label})

            sns.kdeplot(vx, vy, cmap=cmap, fill=False, shade_lowest=True, cut=cut_lim, \
                alpha=1, ax=ax, linewidths=2, cbar=cbar, cbar_ax=cax,cbar_kws={'label': cbar_label}) #; ax.collections[1].set_alpha(0)
        else:
            h = np.histogram2d(vx,vy,bins=bins,range=vel_lims)
            cont = ax.contourf(gaussian_filter(h[0].T,smoothing), levels=contour_levels, extent=[vel_lims[0][0],vel_lims[0][1],vel_lims[1][0],vel_lims[1][1]], alpha=alpha, cmap=cmap)

            if cbar:
                cbar = plt.colorbar(cont,cax=cax)
                cbar.set_label(label=r"$N$",labelpad=20,rotation=0)

#------------------------------------------------------------------------------------------------------------------------
#---- VALUES ------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

    variables_dict = {
        "vertex": CV.calculate_tilt(vx,vy,absolute=False),
        "vertex_error": CE.get_std_bootstrap(CV.calculate_tilt,vx,vy,tilt=True,absolute=False,repeat=bootstrap_repeat),
        "vertex_abs": CV.calculate_tilt(vx,vy,absolute=True),
        "vertex_abs_error": CE.get_std_bootstrap(CV.calculate_tilt,vx,vy,tilt=True,absolute=True,repeat=bootstrap_repeat),
        "anisotropy": CV.calculate_anisotropy(vx,vy),
        "anisotropy_error": CE.get_std_bootstrap(CV.calculate_anisotropy,vx,vy,repeat=bootstrap_repeat),
        "correlation": CV.calculate_correlation(vx,vy),
        "correlation_error": CE.get_std_bootstrap(CV.calculate_correlation,vx,vy,repeat=bootstrap_repeat)
    }
    #for key in variables_dict:
    #    print(variables_dict[key])
    #    dec = 1 if key in ['vertex','vertex_error'] else 2
    #    variables_dict[key] = return_int_or_dec(variables_dict[key],dec=dec)
    #    print(variables_dict[key],'\n')
        
#------------------------------------------------------------------------------------------------------------------------
#---- ELLIPSE -----------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------        

    cov = np.cov(vx, vy)
    eigenvalues = np.linalg.eig(cov)[0]
    radius = 2*np.sqrt(max(eigenvalues)) #radius of 2 sigma, i.e. the 95% confidence level 
    ratio = np.sqrt(min(eigenvalues)/max(eigenvalues))
    centre = [np.mean(vx), np.mean(vy)]
    x_ellipse, y_ellipse = PH.get_ellipse_coords(radius, ratio, variables_dict['vertex'])#, centre = centre)
    ax.plot(x_ellipse+centre[0], y_ellipse+centre[1], lw=4,color='red')
    eigvectors = np.linalg.eig(cov)[1]
    raw_max_vector = eigvectors[:,eigenvalues.argmax()]
    max_vector = raw_max_vector*2.5*np.sqrt(max(eigenvalues))

    vector_plot_data = [[max_vector[0], -max_vector[0]],[max_vector[1], -max_vector[1]]]
    vector_plot_data[0] += centre[0]
    vector_plot_data[1] += centre[1]
    ax.plot(vector_plot_data[0],vector_plot_data[1], lw=4, color='red')
        
#------------------------------------------------------------------------------------------------------------------------
#---- TEXT AND LABELS ---------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

    ax.set_xlabel(r"$v_%s$ [km $\rm s^{-1}$]"%vx_component, size=size_axislabels)
    ax.set_ylabel(r"$v_%s$ [km $\rm s^{-1}$]"%(vy_component if vy_component!="phi" else "\phi"), size=size_axislabels)
    
    if tilt_abs:
        vertex_text = r"$\tilde{l}_{\mathrm{v}}=%.1f$ $\pm$ $%.1f^\circ$"%(variables_dict['vertex_abs'],variables_dict['vertex_abs_error'])
    else:
        vertex_text = r"$l_{\mathrm{v}}=%.1f$ $\pm$ $%.1f^\circ$"%(variables_dict['vertex'],variables_dict['vertex_error'])
    ani_text = r"$\beta_{rl}=%.2f$ $\pm$ $%.2f $"%(variables_dict['anisotropy'],variables_dict['anisotropy_error'])
    corr_text = r"$\rho_{rl}=%.2f$ $\pm$ $%.2f $"%(variables_dict['correlation'],variables_dict['correlation_error'])
    number_text = fr"$N={star_number}$"

    text_box = dict(boxstyle='round', facecolor='linen', alpha=1)
    ax.text(0.025, 0.97, vertex_text, transform=ax.transAxes, fontsize=size_variables, verticalalignment='top', bbox=text_box)
    #ax.text(left_coord, vertex_y_coord, vertex_text, fontsize=size_variables, verticalalignment='top', bbox=text_box)
    ax.text(0.75, 0.97, number_text, transform=ax.transAxes, fontsize=size_variables, verticalalignment='top', bbox=text_box)
    ax.text(0.025, 0.07, ani_text, transform=ax.transAxes, fontsize=size_variables, verticalalignment='top', bbox=text_box)
    ax.text(0.64, 0.07, corr_text, transform=ax.transAxes, fontsize=size_variables, verticalalignment='top', bbox=text_box)

    if title_limits is not None:
        title = fr"${title_str}$\n{np.float16(title_limits[0])} $\leq$ $b[°]$ $<$ {np.float16(title_limits[1])}"
        ax.set_title(title, size_title=size_title)
        if filename is None:
            filename = "velocity_"+population_string+"_i"+str(fileindex)+'_'+str(MF.check_int(np.float16(title_limits[0])))+'b'+str(MF.check_int(np.float16(title_limits[1])))
    else:
        if title_str != '':
            ax.set_title(title_str, size=size_title)
        if filename is None:
            filename = population_string
        
#------------------------------------------------------------------------------------------------------------------------
#---- AXES ------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
           
    ax.set_xlim(vel_lims[0])
    ax.set_ylim(vel_lims[1])
    #ax.minorticks_on()

    ax.set_xticks(np.arange(vel_lims[0][0], vel_lims[0][1], tickstep)) #(-300,400,100)
    ax.set_yticks(np.arange(vel_lims[1][0], vel_lims[1][1], tickstep))
    #ax.tick_params(axis="x", labelsize=size_ticks)
    #ax.tick_params(axis="y", labelsize=size_ticks)

    ax.axvline(0, color='grey', linestyle='--')
    ax.axhline(0, color='grey', linestyle='--')
    
    ax.set_aspect("equal")

#------------------------------------------------------------------------------------------------------------------------
#---- SHOW AND SAVE -----------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

    if save:
        plt.savefig(save_path+filename+fileformat, bbox_inches='tight',dpi=dpi)
        print("Saved:\t"+save_path+filename+fileformat)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    if return_dict:
        return variables_dict