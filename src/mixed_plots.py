import numpy as np
import os
import scipy.stats as stat

import matplotlib.pyplot as plt
from matplotlib import colormaps as mplcmaps
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import matplotlib.cm as cm

import miscellaneous_functions as MF
import plotting_helpers as PH
import map_functions as mapf

STELLAR_MASS = 9.5*10**3 # stellar masses - see bottom left of page 8 in Debattista 2017

def visualise_cmap_idx_color(cmap,idx=0):
    fig,ax=plt.subplots(figsize=(1,1))
    ax.scatter([0],[0],c=cmap(idx),s=2000)
    ax.set(xticks=[],yticks=[])
    plt.show()

def show_color(rgba):
    fig,ax=plt.subplots(figsize=(0.5,0.5))
    ax.scatter([0],[0],color=rgba,s=500)
    ax.set(yticks=[],xticks=[])
    plt.show()

def illustrate_centered_cmap_usage(vmin,vmax,cmap = None, verbose=False):
    """
    Show a coloured scatter plot where the colorbar range is centered on 0.

    There are several test cases that can be (un)commented below. Note it only works if vmin and vmax have different sign or one of them is 0
    """

    if cmap is None:
        cmap = mplcmaps["coolwarm"]
    
    fig,(ax,cax)=plt.subplots(ncols=2,gridspec_kw={"wspace":0.05,"width_ratios":[1,0.04]})

    x = np.random.normal(size=50)
    y = np.random.normal(size=50)
    c = np.random.uniform(vmin,vmax,50)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cmap = PH.get_centered_cmap_from_vminvmax(vmin,vmax,cmap=cmap)

    plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
    ax.scatter(x,y,c=cmap(norm(c)),s=200)

    if verbose:
        print(vmin,norm(vmin))
        print(vmax,norm(vmax))

    plt.show()

def plot_circle(ax,radius=1,centre=[0,0],phirange=[-180,180],color="k",lw=1.5,label="",linestyle=None):
    circle = PH.get_ellipse_coords(centre=centre,radius=radius,phirange=phirange)
    ax.plot(circle[0],circle[1],label=label,color=color,lw=lw,linestyle=linestyle)

def plot_angled_line(ax,xmin,ymin,xmax,angle,degrees=True,color="k",linestyle=None,lw=1):
    if degrees:
        angle *= np.pi/180
    ax.plot([xmin,xmax],[ymin,ymin+(xmax-xmin)*np.tan(angle)],color=color,linestyle=linestyle,lw=lw)

def visualise_bulge_selection(cuts_dict=None,given_axs=None,zabs=True,save_bool=False,save_path="",y_max_plot=999,R0=8.1,projection="both",plot_sun=False):
    """ Generate plot showing the bulge cuts in xy and xz views of the galaxy

    Parameters
    ----------
    cuts: dictionary (optional)
        It can contain any or all of the following keys: {"d","l","b","R","y","l"}. The dictionary values are lists. 
        The values for the following variables will be assumed to be maxima of symmetric cuts:
            Over the origin: "d"
            Over the x-axis: "l", "y"
            Over the Galactic plane: "b" (if zabs==False).
        Defaults to empty lists.
    axs: matplotlib Axes object (optional)
        If given, it will just add the plot elements to them. Otherwise it will create a new fig and axs, and it will show the plot.
    save_bool: boolean (optional)
        Only takes effect if `axs` are not given. If True, it will save the plot to the path specified by `save_path`. Defaults to False.
    save_path: string (optional)
        Only takes effect if `axs` are not given and `save_bool` is True.
        If not given, it will default to "graphs/other_plots/visualise_bulge_cuts/" if it exists, otherwise the current directory.
    y_max_plot: float
        The y axis will not surpass this number. Defaults to 999.
    R0: float
        Distance Sun-GC in kpc. Defaults to 8.1.
    projection: string
        Set to "xy" or "xz" to keep only that projection and delete the other. Defaults to "both".
    """

    def get_cut_labels(cuts_dict,zabs):
        labels_dict = {}

        symbol_dict,units_dict = mapf.get_position_symbols_and_units_dict(zabs=zabs,degree_symbol='°')

        symbol_dict["l"] = r"$|l|$"
        symbol_dict["y"] = r"$|y|$"
        for k in units_dict:
            if units_dict[k] == "kpc":
                units_dict[k] = " kpc"

        for cut_key in cuts_dict:
            cut_list = cuts_dict[cut_key]
            
            if len(cut_list) > 0:
                val_string = str(MF.return_int_or_dec(cut_list[0],2))
                for cut in cut_list[1:]:
                    val_string += f", {str(MF.return_int_or_dec(cut,2))}"

                labels_dict[cut_key] = symbol_dict[cut_key] + "$=%s$"%val_string + units_dict[cut_key]
            else:
                labels_dict[cut_key] = None

        return labels_dict
    
    default_cuts = {"l":[],"y":[],"b":[],"z":[],"d":[],"R":[]}

    if cuts_dict is None:
        cuts_dict = default_cuts
    else:
        for key in default_cuts.keys():
            if key not in cuts_dict:
                cuts_dict[key] = default_cuts[key]
    
    if given_axs is None:
        fig,axs=plt.subplots(nrows=2,sharex=True,gridspec_kw={"hspace":0})
    else:
        axs = given_axs
    
    dmin_list = [MF.return_int_or_dec(R0 - (dmax - R0)) for dmax in cuts_dict["d"]]
    cuts_dict["d"] += dmin_list
    cuts_dict["d"] = sorted(cuts_dict["d"])

    Rgc_max = max(cuts_dict["R"]) if len(cuts_dict["R"])>0 else 3.5

    labels_dict = get_cut_labels(cuts_dict,zabs)

    if projection in ["xy","both"]:
        for lmax in cuts_dict["l"]:
            axs[0].plot([-R0,Rgc_max+0.5],[0,(R0+Rgc_max+0.5)*np.tan(lmax*np.pi/180)],color="red",label=labels_dict["l"] if lmax == cuts_dict["l"][0] else None)
            axs[0].plot([-R0,Rgc_max+0.5],[0,-(R0+Rgc_max+0.5)*np.tan(lmax*np.pi/180)],color="red")

        for ymax in cuts_dict["y"]:
            axs[0].axhline(ymax,color="purple",label=labels_dict["y"] if ymax==cuts_dict["y"][0] else None)
            axs[0].axhline(-ymax,color="purple")

        phirange = [0,360] if len(cuts_dict["l"]) == 0 or len(cuts_dict["l"]) > 1 else [-cuts_dict["l"][0],cuts_dict["l"][0]]
        for d in cuts_dict["d"]:
            plot_circle(axs[0],radius=d,centre=[-R0,0],phirange=phirange,color="orange",label=labels_dict["d"] if d==cuts_dict["d"][0] else None)

        for Rgc in cuts_dict["R"]:
            plot_circle(axs[0],radius=Rgc,lw=1,label=labels_dict["R"] if Rgc==cuts_dict["R"][0] else None)

        axs[0].set(xlabel=r"$x$ [kpc]",ylabel=r"$y$ [kpc]")
    
    if projection in ["xz","both"]:
        ax_xz = axs[1]
    
        for Rgc in cuts_dict["R"]:
            ax_xz.axvline(x=-Rgc,color="k",lw=1)
            ax_xz.axvline(x=Rgc,color="k",lw=1)

        for bmax in cuts_dict["b"]:
            ax_xz.plot([-R0,Rgc_max+0.5],[0,(R0+Rgc_max+0.5)*np.tan(bmax*np.pi/180)],color="red",label=labels_dict["b"] if bmax==cuts_dict["b"][0] else None)
            if not zabs:
                ax_xz.plot([-R0,Rgc_max+0.5],[0,-(R0+Rgc_max+0.5)*np.tan(bmax*np.pi/180)],color="green")

        for zmax in cuts_dict["z"]:
            ax_xz.axhline(zmax,color="cyan", label=labels_dict["z"] if zmax==cuts_dict["z"][0] else None)
            if not zabs:
                ax_xz.axhline(-zmax,color="cyan")

        phirange = [0,360] if len(cuts_dict["b"]) == 0 or len(cuts_dict["b"]) > 1 else [-cuts_dict["b"][0],cuts_dict["b"][0]]
        for d in cuts_dict["d"]:
            plot_circle(ax_xz,radius=d,centre=[-R0,0],phirange=phirange,color="orange",label=labels_dict["d"] if d==cuts_dict["d"][0] else None)
        
        ax_xz.set(xlabel=r"$x$ [kpc]",ylabel=r"$%s$ [kpc]"%'|z|' if zabs else 'z')
    
    ylim_max = min(y_max_plot,Rgc_max+0.05)

    for ax in axs:
        if plot_sun:
            sun_label = r"$R_0=%s$ kpc"%(MF.return_int_or_dec(R0,2)) if (projection in ["xy","both"] and ax == axs[0]) or (projection == "xz" and ax == axs[1]) else None
            ax.scatter([-R0],[0],marker="*",s=70,color="cyan",label=sun_label,zorder=10)
        ax.scatter([0],[0],color="k",s=30,marker="x",zorder=10)
        ax.plot([-R0,Rgc_max+0.5],[0,0],color="grey",linestyle="--")
        
        if given_axs is None:
            ax.set_xlim(-R0-0.5,Rgc_max+0.5)

            ax.set_ylim(-ylim_max,ylim_max)
            ax.set_aspect("equal")
            ax.legend()

    if given_axs is None and zabs:
        axs[1].set_ylim(-0.05)
    
    if True: # filename
        filename = f"visualise_cuts"
        
        for key,val_list in cuts_dict.items():
            if projection == "xy" and key not in ["l","y","R","d"]:
                continue
            if projection == "xz" and key not in ["z","b","d","R"]:
                continue

            for v in val_list:
                filename += f"_{key}{MF.return_int_or_dec(v,2)}"

    if given_axs is None:
        if projection != "both":
            fig.delaxes(axs[0] if "xz" else axs[1])

        if save_bool:
            if save_path == "" and os.path.isdir("graphs/other_plots/visualise_bulge_cuts/"):
                save_path = "graphs/other_plots/visualise_bulge_cuts/"

            print("Saving in:",save_path)
            for fileformat in [".pdf",".png"]:
                plt.savefig(save_path+filename+fileformat, dpi=200,bbox_inches="tight")
                print(fileformat)

        plt.show()
    else:
        return filename,cuts_dict

def quick_show_xy(df,xmin=None,xmax=None,ymin=None,ymax=None,bins_x=100,aspect="equal",alpha=1,ax=None,show=True,norm=LogNorm(),density=True):
    """
    Generate mass density counts in of a dataframe in xz space within the given limits. If show==True, show an imshow of it.
    """
    
    if xmin is None:
        xmin = np.min(df["x"])
    if xmax is None:
        xmax = np.max(df["x"])
    if ymin is None:
        ymin = np.min(df["y"])
    if ymax is None:
        ymax = np.max(df["y"])

    bins_y = int(bins_x * (ymax-ymin)/(xmax-xmin))

    c,_,_ = np.histogram2d(df["x"],df["y"],bins=[bins_x,bins_y],range=[[-xmax,xmax],[-ymax,ymax]],density=density)

    #Multiply by total number to get a surface density rather than probability density, and by stellar mass to get a mass density
    c = c.T * (len(df)*STELLAR_MASS if density else 1)

    if show:
        assert ax is not None, "Show option was set so expected ax to be given."

        ax.imshow(c,norm=norm,alpha=alpha,origin="lower",extent=[-xmax,xmax,-ymax,ymax])

        ax.set(xlabel=r"$x$ [kpc]",ylabel=r"$y$ [kpc]")
        ax.set_aspect(aspect)

    return c

def quick_show_xz(df,xmin=None,xmax=None,zmin=None,zmax=None,bins_x=100,aspect="equal",alpha=1,ax=None,show=True,norm=LogNorm(),density=True):
    """
    Generate mass density counts in of a dataframe in xz space within the given limits. If show==True, show an imshow of it.
    """
    
    if xmin is None:
        xmin = np.min(df["x"])
    if xmax is None:
        xmax = np.max(df["x"])
    if zmin is None:
        zmin = np.min(df["z"])
    if zmax is None:
        zmax = np.max(df["z"])

    bins_z = int(bins_x * (zmax-zmin)/(xmax-xmin))

    c,_,_ = np.histogram2d(df["x"],df["z"],bins=[bins_x,bins_z],range=[[-xmax,xmax],[zmin,zmax]],density=density)

    #Multiply by total number to get a surface density rather than probability density, and by stellar mass to get a mass density
    c = c.T * (len(df)*STELLAR_MASS if density else 1)

    if show:
        assert ax is not None, "Show option was set so expected ax to be given."

        ax.imshow(c,norm=norm,alpha=alpha,origin="lower",extent=[-xmax,xmax,zmin,zmax])

        ax.set(xlabel=r"$x$ [kpc]",ylabel=r"$|z|$ [kpc]" if zmin==0 else r"$z$ [kpc]")
        ax.set_aspect(aspect)

    return c
 
def visually_inspect_bar_angle(df, xymax=4, zmin=0, age_lims=None, bins=50, bar_angle=None):
    fig,ax=plt.subplots()

    cut_df = df[(np.abs(df["x"])<xymax)&(np.abs(df["y"])<xymax)]

    if zmin != 0:
        cut_df = cut_df[np.abs(cut_df["z"])>zmin]
    
    if age_lims is not None:
        cut_df = cut_df[(cut_df["age"]>age_lims[0])&(cut_df["age"]<age_lims[1])]

    ax.hist2d(cut_df["x"],cut_df["y"],bins=bins,norm=LogNorm())
    
    if bar_angle is not None:
        slope = np.tan(bar_angle*np.pi/180)
        ax.plot([-4,4],[4*slope,-4*slope],color="red",label=r"Bar at $\alpha=%i^\circ$"%bar_angle)
        plt.legend()
        
    ax.set_aspect("equal")

    plt.show()

def visualise_1D_binning(value_array, bin_edges_min,bin_edges_max=None, given_ax=None, hist_bins=100,log=False,xlabel="",filename_prefix="",save_bool=False,save_path=None,show_bool=True):
    """
    Currently used to visualise the latitude binnings in the latitude plot.
    """

    if given_ax is None:
        fig,ax=plt.subplots(figsize=(8,4))
    else:
        ax = given_ax
    
    ax.hist(value_array,bins=hist_bins,color="grey",log=log)
    for edge in bin_edges_min:
        ax.axvline(x=edge,color="red")
    
    if bin_edges_max is not None:
        for edge in bin_edges_max:
            ax.axvline(x=edge,color="blue")
    
    major_tick_locator = MF.round_one_significant(MF.return_int_or_dec(max(value_array)-min(value_array)))//10
    if major_tick_locator == 0: major_tick_locator = 0.5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_locator))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(major_tick_locator/2))
    
    if given_ax is None:
        ax.set_ylabel(r"$N$",rotation=0,labelpad=20)
        ax.set_xlabel(xlabel)

        if save_bool and save_path is not None:
            log_string = "_log" if log else ""
            prefix_string = f"{filename_prefix}_" if filename_prefix != "" else ""
            nbins_string = f"_{hist_bins}bins"

            filename = f"{prefix_string}chosenRanges{log_string}{nbins_string}"
            
            filepath = save_path+filename+".png"

            plt.savefig(filepath,dpi=200,bbox_inches="tight")
            print("Saved",filepath)
        
        if show_bool:
            plt.show()

def show_text(text):
    fig,ax=plt.subplots(figsize=(0.1,0.1));ax.set_yticks([]);ax.set_xticks([])
    for spine in ['top', 'right',"bottom","left"]:
        ax.spines[spine].set_visible(False)
    plt.text(x=0,y=0,s=text)

def visualise_GC_distances(xymax=3.5,xybins=21,black_contour_level=3.5):
    fig,ax=plt.subplots()

    c = np.zeros(shape=(xybins,xybins))
    for i in range(xybins):
        for j in range(xybins):
            c[i,j] = np.sqrt(abs(i-xybins//2)**2 + abs(j-xybins//2)**2) * 2*xymax/xybins

    h=ax.contourf(c,extent=[-xymax,xymax,-xymax,xymax],cmap="jet",levels=xybins)
    if black_contour_level is not None:
        ax.contour(c,extent=[-xymax,xymax,-xymax,xymax],levels=[black_contour_level],colors="k")
    ax.set_aspect("equal")
    plt.colorbar(h)
    plt.show()

def plot_velocity_histograms_single_stat(df,vel_x_variable,vel_y_variable,bins=100,colour_var="x",colour_stat="mean",cmap=None,\
                                         save_bool=False,save_path="",suffix="",verbose=False,show=False):
    fig,axs=plt.subplots(figsize=(13,7),ncols=4,gridspec_kw={"width_ratios":[1,1,-0.3,0.1],"wspace":0.4})
    
    if True: # plot
        N1,_,patches1 = axs[0].hist(df[f"v{vel_x_variable}"],bins=bins)
        N2,_,patches2 = axs[1].hist(df[f"v{vel_y_variable}"],bins=bins)
        
        scipyh_vx_x,_,_ = stat.binned_statistic(df[f"v{vel_x_variable}"],values=df[colour_var],statistic=colour_stat,bins=bins)
        scipyh_vy_x,_,_ = stat.binned_statistic(df[f"v{vel_y_variable}"],values=df[colour_var],statistic=colour_stat,bins=bins)
        
        min_x = min(min(scipyh_vx_x),min(scipyh_vy_x))
        max_x = max(max(scipyh_vx_x),max(scipyh_vy_x))
        norm = plt.Normalize(min_x,max_x)
        
        if cmap is None:
            cmap = mplcmaps["coolwarm"]

        if colour_stat in ["mean"]:
            cmap = PH.get_centered_cmap_from_vminvmax(cmap=cmap,vmin=min_x,vmax=max_x)
        elif colour_stat in ["std"]:
            cmap = PH.get_truncated_colormap(cmap=cmap,minval=0.5,maxval=1)
        
        for i in range(bins):
            patches1[i].set_facecolor(cmap(norm(scipyh_vx_x[i])))
            patches2[i].set_facecolor(cmap(norm(scipyh_vy_x[i])))

        axs[0].set_xlabel(r"$v_%s$ [km/s]"%vel_x_variable)
        axs[1].set_xlabel(r"$v_%s$ [km/s]"%vel_y_variable)

    #     for ax in [axs[1],axs[2]]:
    #         ax.set_ylim(0,np.max([N1,N2]))

        axs[0].set_ylabel(r"$N$",rotation=0,labelpad=20)

        plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax=axs[3])
        axs[3].set_ylabel(r"$\langle %s \rangle$ [kpc]"%colour_var)#,rotation=0,labelpad=15)

        fig.delaxes(axs[2])
    
    if True: # save, show
        colour_str = f"_{colour_stat}{colour_var}"
        suffix = f"_{suffix}" if suffix != "" else ""

        filename = "velhists" + colour_str + suffix
        
        if verbose:
            print(filename)
    
        if save_bool:
            assert save_path != "", "Please set a valid `save_path`."

            plt.savefig(save_path+filename+".png",dpi=250,bbox_inches="tight")

            if verbose:
                print("Saved in",save_path)

        plt.show() if show else plt.close()

def plot_velocity_histograms_both_stats(df,vel_x_variable,vel_y_variable,bins=100,colour_var="x",cmap=None,\
                                        save_bool=False,save_path="",suffix="",verbose=False,show=False):
    """
    Plots a 2x2 figure where the first/second row show the velocity histograms for given velocity components.
    The left/right columns show the same histograms but coloured by the mean and standard deviation of `colour_var` respectively.
    """

    if True: # fix, ax
        fig,axs=plt.subplots(figsize=(13,10),nrows=4,ncols=2,gridspec_kw={"height_ratios":[0.1,0.02,1,1],"wspace":0,"hspace":0})

        cax_mean = axs[0,0]
        cax_std = axs[0,1]

        ax_vx_mean = axs[2,0]
        ax_vy_mean = axs[3,0]

        ax_vx_std = axs[2,1]
        ax_vy_std = axs[3,1]
    
    if True: # plot
        
        _,_,patches00 = ax_vx_mean.hist(df[f"v{vel_x_variable}"],bins=bins)
        _,_,patches10 = ax_vy_mean.hist(df[f"v{vel_y_variable}"],bins=bins)
        
        _,_,patches01 = ax_vx_std.hist(df[f"v{vel_x_variable}"],bins=bins)
        _,_,patches11 = ax_vy_std.hist(df[f"v{vel_y_variable}"],bins=bins)
        
        scipyh_vx_mean,_,_ = stat.binned_statistic(df[f"v{vel_x_variable}"],values=df[colour_var],statistic="mean",bins=bins)
        scipyh_vy_mean,_,_ = stat.binned_statistic(df[f"v{vel_y_variable}"],values=df[colour_var],statistic="mean",bins=bins)
        scipyh_vx_std,_,_ = stat.binned_statistic(df[f"v{vel_x_variable}"],values=df[colour_var],statistic="std",bins=bins)
        scipyh_vy_std,_,_ = stat.binned_statistic(df[f"v{vel_y_variable}"],values=df[colour_var],statistic="std",bins=bins)

        min_mean = min(min(scipyh_vx_mean),min(scipyh_vy_mean))
        max_mean = max(max(scipyh_vx_mean),max(scipyh_vy_mean))
        norm_mean = plt.Normalize(min_mean,max_mean)
        
        min_std = min(min(scipyh_vx_std),min(scipyh_vy_std))
        max_std = max(max(scipyh_vx_std),max(scipyh_vy_std))
        norm_std = plt.Normalize(min_std,max_std)

        if cmap is None:
            cmap = mplcmaps["coolwarm"]
        
        cmap_mean = PH.get_centered_cmap_from_vminvmax(cmap=cmap,vmin=min_mean,vmax=max_mean)
        cmap_std = PH.get_truncated_colormap(cmap=cmap,minval=0.5,maxval=1)
        
        for i in range(bins):
            patches00[i].set_facecolor(cmap_mean(norm_mean(scipyh_vx_mean[i])))
            patches10[i].set_facecolor(cmap_mean(norm_mean(scipyh_vy_mean[i])))
            
            patches01[i].set_facecolor(cmap_std(norm_std(scipyh_vx_std[i])))
            patches11[i].set_facecolor(cmap_std(norm_std(scipyh_vy_std[i])))

    if True: # axes
#         ax_vx_mean.set_xlabel(r"$v_%s$ [km/s]"%vel_x_variable)
#         ax_vx_std.set_xlabel(r"$v_%s$ [km/s]"%vel_x_variable)
#         ax_vy_mean.set_xlabel(r"$v_%s$ [km/s]"%vel_y_variable)
#         ax_vy_std.set_xlabel(r"$v_%s$ [km/s]"%vel_y_variable)

        ax_vy_mean.set_xlabel("Velocity [km/s]")
        ax_vy_std.set_xlabel("Velocity [km/s]")
        
        ax_vx_mean.text(x=0.1,y=0.8,s=r"$v_%s$"%vel_x_variable,transform=ax_vx_mean.transAxes,bbox={"lw":0.5,"facecolor":"white"})
        ax_vx_std.text(x=0.1,y=0.8,s=r"$v_%s$"%vel_x_variable,transform=ax_vx_std.transAxes,bbox={"lw":0.5,"facecolor":"white"})
        ax_vy_mean.text(x=0.1,y=0.8,s=r"$v_%s$"%vel_y_variable,transform=ax_vy_mean.transAxes,bbox={"lw":0.5,"facecolor":"white"})
        ax_vy_std.text(x=0.1,y=0.8,s=r"$v_%s$"%vel_y_variable,transform=ax_vy_std.transAxes,bbox={"lw":0.5,"facecolor":"white"})

        ax_vx_mean.set_ylabel(r"$N$",rotation=0,labelpad=20)
        ax_vy_mean.set_ylabel(r"$N$",rotation=0,labelpad=20)
        
#         ax_vx_mean.set_yticks(ax_vx_mean.get_yticks()[1:])
        ax_vx_mean.set_xticklabels([]);ax_vx_std.set_xticklabels([])
        ax_vx_std.set_yticklabels([]);ax_vy_std.set_yticklabels([])
        
        min_v = min([np.min(df[f"v{vel_x_variable}"]), np.min(df[f"v{vel_y_variable}"])])
        max_v = max([np.max(df[f"v{vel_x_variable}"]), np.max(df[f"v{vel_y_variable}"])])
        for ax in [ax_vx_mean,ax_vy_mean,ax_vx_std,ax_vy_std]:
            ax.set_xlim(min_v,max_v)
            
        fig.delaxes(axs[1,0])
        fig.delaxes(axs[1,1])
        
    if True: # cbar

        plt.colorbar(cm.ScalarMappable(norm=norm_mean,cmap=cmap_mean),cax=cax_mean,orientation="horizontal")
        plt.colorbar(cm.ScalarMappable(norm=norm_std,cmap=cmap_std),cax=cax_std,orientation="horizontal")
        cax_mean.set_xlabel(r"$\langle %s \rangle$ [kpc]"%colour_var,labelpad=15)
        cax_std.set_xlabel(r"$\sigma_%s$ [kpc]"%colour_var,labelpad=15)
        cax_mean.xaxis.tick_top();cax_mean.xaxis.set_label_position("top")
        cax_std.xaxis.tick_top();cax_std.xaxis.set_label_position("top")
        
        cax_std_ticks = mapf.get_ticks(cax=cax_std,vmin=min_std,vmax=max_std,axis="x")
        mapf.remove_ticklabel(ax=cax_std,ticks=cax_std_ticks,axis="x",which="bottom")
    
    if True: # save, show
        colour_str = f"_meanstd{colour_var}"
        suffix = f"_{suffix}" if suffix != "" else ""

        filename = "velhists" + colour_str + suffix
        
        if verbose:
            print(filename)
    
        if save_bool:
            assert save_path != "", "Please set a valid `save_path`."

            for fileformat in [".png",".pdf"]:
                plt.savefig(save_path+filename+fileformat,dpi=250,bbox_inches="tight")

            if verbose:
                print("Saved in",save_path)

        plt.show() if show else plt.close()

def visualise_spherical_tilt_calculation(beta_array, phi_vector):
    fig,ax = plt.subplots()

    for i,ang in enumerate(beta_array):
        ang *= np.pi/180
        ax.plot([0,np.cos(ang)],[0,np.sin(ang)],"k--")
        ax.text(np.cos(ang),np.sin(ang),s=str(i),color="r")
    
    phi_vector *= np.pi/180
    
    ax.plot([0,np.cos(phi_vector)],[0,np.sin(phi_vector)],color="g")

    ax.set_aspect("equal")
    plt.show()