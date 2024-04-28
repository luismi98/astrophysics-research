import numpy as np
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.path as mpltPath
from shapely.geometry import Polygon

def voronoi_finite_polygons_2d(vor): 
    # Full credit to Steven
    
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def generate_2Dvor(x_values, y_values, min_number_stars, remove_unbound_regions=False, pixelsize=0.5, vorbinplot=False):
    N = len(x_values)
    
    signal = np.ones(N)
    noise = np.ones(N)
    
    target_sn = np.sqrt(min_number_stars)
    
    _,_,_, x_bar, y_bar, sn, *_ = voronoi_2d_binning(
        x_values, y_values, signal, noise, target_sn, pixelsize=pixelsize, plot=vorbinplot)
    
    x_bar = x_bar[sn >= target_sn]
    y_bar = y_bar[sn >= target_sn]

    # Build Voronoi tessellation
    points = np.column_stack((x_bar, y_bar))
    vor = Voronoi(points)
    
    if remove_unbound_regions:
        # Clean cell at infinity
        if [] in vor.regions:
            vor.regions.remove([])

        # Remove unbound cells
        idx = 0
        while idx < len(vor.regions):
            if -1 in vor.regions[idx]:
                vor.regions.remove(vor.regions[idx])
            else:
                idx += 1
    else:
        vor.regions, vor.vertices = voronoi_finite_polygons_2d(vor) # Steven's function!
    
    return vor, points

def calculate_point_region(vor, x_values, y_values, remove_unbound_regions=False):
    # Build point_region array (maps each input datapoint to a Voronoi region)
    
    N = len(x_values)
    input_points = np.column_stack([x_values,y_values])
    
    point_region = np.full(shape=(N),fill_value=-9999.0)

    for i, cell_vertices in enumerate(vor.regions):
        if -1 in cell_vertices:
            raise ValueError("Unexpected -1 in region vertices")
        polygon = vor.vertices[cell_vertices]

        path = mpltPath.Path(polygon)
        points_inside = path.contains_points(input_points)

        point_region[points_inside] = i

    # if not remove_unbound_regions:
    #     assert not -9999.0 in point_region, "There are some -9999 left"
        
    return point_region

def compute_values(regions, df, point_region, compute_variable_function="counts", **compute_variable_kwgs):
    computed_values = np.full(shape=(len(regions)),fill_value=-9999.0)
    
    for i, cell_vertices in enumerate(regions):
        if -1 in cell_vertices:
            raise ValueError("Unexpected -1 in region vertices")
        
        if compute_variable_function == "counts":
            computed_values[i]=np.sum([point_region == i])
        else:
            computed_values[i]=compute_variable_function(df.iloc[point_region == i], **compute_variable_kwgs)
            
    assert not -9999.0 in computed_values, "There are some -9999 left"
    
    return computed_values
    
def plot_voronoi_map(vor, region_values, ax, cbar=False,cmap=cm.viridis, value_text=False, symmetric_lims=False, replace_plot_scipy=False):
    
    if replace_plot_scipy:
        voronoi_plot_2d(vor,ax)
    else:
        
        if symmetric_lims:
            symmetric_lim = np.max(np.abs([region_values.min(),region_values.max()]))
            norm = plt.Normalize(-symmetric_lim, symmetric_lim)
        else:
            norm = plt.Normalize(region_values.min(), region_values.max())
        
        if value_text:
            xmin,xmax,ymin,ymax = np.min(vor.points[:,0]),np.max(vor.points[:,0]),np.min(vor.points[:,1]),np.max(vor.points[:,1])
        
        for (cell_vertices,value) in zip(vor.regions,region_values):
            if -1 in cell_vertices:
                raise ValueError("Unexpected -1 in region vertices")
            polygon = vor.vertices[cell_vertices]

            poly = plt.Polygon(polygon, edgecolor="k", lw=0.5, facecolor=cmap(norm(value)))
            ax.add_patch(poly)

            if value_text:
                centroid = Polygon(polygon).centroid
                if centroid.x > xmin-1 and centroid.x < xmax+1 and centroid.y > ymin-1 and centroid.y < ymax+1:
                    ax.text(centroid.x,centroid.y,color="k",s="%i"%value if value==int(value) else "%.2f"%value)

        if cbar:
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Star number',shrink=0.7)
            
def apply_and_plot_dynamical_binning_map(df, x_var, y_var, min_number_stars, ax, symmetric_lims=False, remove_unbound_regions=False, pixelsize=0.5, vorbinplot=False, replace_plot_scipy=False, cmap=cm.viridis, cbar=False, value_text=False, compute_variable_function="counts", **compute_variable_kwgs):
    
    x_values = df[x_var].values
    y_values = df[y_var].values
    
    vor = generate_2Dvor(x_values, y_values, min_number_stars, remove_unbound_regions=remove_unbound_regions, pixelsize=pixelsize, vorbinplot=vorbinplot)
    
    point_region = calculate_point_region(vor, x_values, y_values, remove_unbound_regions=remove_unbound_regions)
    
    computed_values = compute_values(vor.regions, df, point_region, compute_variable_function=compute_variable_function, **compute_variable_kwgs)
    
    plot_voronoi_map(vor, computed_values, ax, cmap=cmap, value_text=value_text, cbar=cbar, symmetric_lims=symmetric_lims, replace_plot_scipy=replace_plot_scipy)
    
    return vor, computed_values