import numpy as np
from astropy.table import Table
import coordinates
import miscellaneous_functions as MF

Z0_CONST = coordinates.get_solar_height()
R0_CONST = coordinates.get_solar_radius()

def select_dr2_variables(allStar, error_bool=False):
    """
    #Read https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/allStar.html#hdu1
    
    VHELIO_AVG: "Average solar system barycentric radial velocity, weighted by S/N, using RVs determined from cross-correlation of individual 
                spectra with combined spectrum". Explanation by GPT4:
        "Average solar system barycentric radial velocity"
            This refers to the average velocity at which an object (e.g., a star, a planet) is moving away from or towards the 
            center of mass of the solar system (the barycenter), when viewed along the line of sight. This is measured in the radial direction 
            (directly away from or towards the observer).
        "Weighted by S/N"
            This implies that the velocities are averaged in a way that gives more weight to measurements with higher signal-to-noise (S/N) ratios.
            The signal-to-noise ratio is a measure of the clarity of the signal in the presence of noise.
            In this context, higher S/N measurements are more reliable, so they get more weight in the average.
        "RVs determined from cross-correlation of individual spectra with combined spectrum"
            RVs stands for Radial Velocities. The way these velocities are determined is by cross-correlating individual spectra with a combined spectrum.
            In astronomy, a spectrum (plural: spectra) is the range of colors (including invisible ones) that light from an astronomical object can have. 
            The cross-correlation mentioned here is a measure of similarity between the individual spectrum of an object and a combined or reference spectrum.
            This comparison can help in determining the radial velocity of the object.
    VERR: "Uncertainty in VHELIO_AVG from the S/N-weighted individual RVs"
    
    Note to use for example the equatorial proper motion errors, we'd need to convert them to galactic proper motions (and then change the 
    units to velocity). The conversion from equatorial to galactic requires propagation of errors, which would need looking into the bovy 
    equations and taking derivatives. An alternative is to use Monte Carlo to feed the bovy equatorial-to-galactic proper motion conversion 
    values of pmra and pmdec perturbed by the corresponding errors, and compute the standard deviation of the resulting pml and pmbcosb values.
    """
    
    # DR2
    kinematic_list = ["VHELIO_AVG"]
    kinematic_errors = ["VERR"]
    position_list = ["GLON","GLAT","RA","DEC"]
    properties_list = ["FE_H"]
    properties_error = ["FE_H_ERR"]

    #other_position_list = ["GAIA_R_LO","GAIA_R_HI"]
    #other_kinematic_list = ["GAIA_PMRA", "GAIA_PMDEC", "OBSVHELIO_AVG","VSCATTER", "OBSVSCATTER", "SYNTHVSCATTER", "SYNTHVHELIO_AVG", "GAIA_RADIAL_VELOCITY"]
    #other_properties_list = ['RV_FEH','FE_H_FLAG']
    #all_errors_list = ["VERR","VERR_MED","OBSVERR","OBSVERR_MED","SYNTHVERR","SYNTHVERR_MED","GAIA_PMRA_ERROR", "GAIA_PMDEC_ERROR", "GAIA_RADIAL_VELOCITY_ERROR","FE_H_ERR"]
    
    columns_to_keep_dr2 = np.concatenate([kinematic_list,position_list, properties_list])
    if error_bool:
        columns_to_keep_dr2 = np.concatenate([columns_to_keep_dr2, properties_error, kinematic_errors])
        
    all_keys_dr2 = allStar.keys()
    delete_columns_dr2 = [column for column in all_keys_dr2 if column not in columns_to_keep_dr2]
    del allStar[delete_columns_dr2]

def select_dr3_variables(gaiaDR3, error_bool=False):
    # DR3
    kinematic_list = ["GAIA_PMRA", "GAIA_PMDEC"]
    quality_list = ["GAIA_RUWE", "good_delta_Gmag"]
    kinematic_errors = ["GAIA_PMRA_ERROR", "GAIA_PMDEC_ERROR"]
    #other_kinematic_list = ["GAIA_RADIAL_VELOCITY", "GAIA_RADIAL_VELOCITY_ERROR"]
    
    columns_to_keep_dr3 = np.concatenate([kinematic_list, quality_list])
    if error_bool:
        columns_to_keep_dr3 = np.concatenate([columns_to_keep_dr3, kinematic_errors])
    
    all_keys_dr3 = gaiaDR3.keys() 
    delete_columns_dr3 = [column for column in all_keys_dr3 if column not in columns_to_keep_dr3]
    del gaiaDR3[delete_columns_dr3]

def get_dataframe(allStar, spectrophoto, gaiaDR3, error_bool=False):
    # DR2
    dr2 = allStar.to_pandas()
    dr2.rename(columns={'GLON':'l', 'GLAT':'b', "RA":"ra", "DEC": "dec", 'FE_H':'FeH', "VHELIO_AVG":"vr"}, inplace=True)
    if error_bool:
        dr2.rename(columns = {'FE_H_ERR':'FeH_error', "VERR":"vr_error"},inplace=True)
    del allStar
    
    # DR3
    dr3 = gaiaDR3.to_pandas()
    dr3.rename(columns={"GAIA_PMRA":"pmra","GAIA_PMDEC":"pmdec"},inplace=True)
    if error_bool:
        dr3.rename(columns={"GAIA_PMRA_ERROR":"pmra_error","GAIA_PMDEC_ERROR":"pmdec_error"},inplace=True)
    #for column in dr3.columns:
    #    dr3.rename(columns={column:"dr3_"+column},inplace=True)
    del gaiaDR3
        
    # JOIN
    data = dr2.join(dr3)
    del dr2,dr3
        
    # DISTANCE
    data['d'] = spectrophoto['D2_med']
    if error_bool:
        data['d_error'] = spectrophoto['sig2']
    del spectrophoto

    return data

def load_data(data_path="/Users/Luismi/Desktop/MRes_UCLan/data/Observational_data/", error_bool = True):
    
    allStar = Table.read(data_path+'gaia_DR2/allStar_bulgeSample_ARA2020paper.fits', format='fits',hdu=1)
    spectrophoto = Table.read(data_path+'gaia_DR2/spectroPhotomDists_bulgeSample_ARA2020paper.fits', format='fits')
    gaiaDR3 = Table.read(data_path+'gaia_DR3/GaiaDR3_data_allStar_bulgeSample_ARA2020paper_order.fits', format='fits',hdu=1)
    
    assert len(allStar) == len(spectrophoto) == len(gaiaDR3), "All data must have the same dimensions!"
    print(f"Found {len(allStar)} total stars.")
          
    select_dr2_variables(allStar, error_bool=error_bool)
    select_dr3_variables(gaiaDR3, error_bool=error_bool)
    
    return get_dataframe(allStar, spectrophoto, gaiaDR3, error_bool=error_bool)

def clean_up_bad_data(data, verbose=False):
    # Values not available in dr2 are flagged with values around -9999
    for variable in data.keys():
        data.loc[data[variable] <= -9999, variable] = np.nan
        
    if verbose:
        theresnan = np.sum(np.isnan(data))
        print(theresnan)

    MF.prevent_byteorder_error(data)
    
    # Eliminate nans
    no_metal = np.isnan(data["FeH"])
    no_pm = np.isnan(data["pmra"]) | np.isnan(data["pmdec"])
    no_distance = np.isnan(data["d"])
    no_vr = np.isnan(data["vr"])
    bad_ruwe = data["GAIA_RUWE"] > 1.4
    bad_match = data["good_delta_Gmag"] == 0

    to_remove = no_metal | no_pm | no_distance | no_vr | bad_ruwe | bad_match
    data.drop(np.where(to_remove)[0],inplace=True)
    print(f"Removed {np.sum(to_remove)} bad indices, leaving {len(data)} stars.")

def convert_positions_and_velocities(data,zabs=True,GSR=True,R0=R0_CONST,Z0=Z0_CONST):
    
    data.loc[data.l > 180, 'l'] -= 360 # Longitude between -180 and 180

    coordinates.pmrapmdec_to_pmlpmb(data)

    # Reflect above plane. Note pmb needs to be first
    if zabs:
        data.loc[data.b < 0, 'pmb'] *= -1
        data.loc[data.b < 0, 'b'] *= -1

    coordinates.lbd_to_xyz(data, GSR=GSR, R0=R0, Z0=Z0)
    coordinates.xyz_to_Rphiz(data)

    # Phi between 0 and 360
    data.loc[data.phi < 0, 'phi'] += 360

    v_sun = coordinates.get_solar_velocity(changing_reference_frame=GSR)
    if GSR: assert v_sun != [0,0,0], "`v_sun` should not be zero, as observed velocities are given from the perspective of the Sun and need to be corrected."

    coordinates.vrpmlpmb_to_vxvyvz(data,v_sun=v_sun,R0=R0,Z0=Z0)
    coordinates.vxvyvz_to_vRvphivz(data)

    if GSR:
        coordinates.vxvyvz_to_vrpmlpmb(data)
    coordinates.pmlpmb_to_vlvb(data)

def load_and_process_data(data_path="/Users/Luismi/Desktop/MRes_UCLan/Observational_data/", error_bool = False, zabs=True, 
                          GSR=True, R0=R0_CONST, verbose = False, drop_unused=True):
    
    print(f"Working with zabs == {zabs}; GSR == {GSR}.")

    data = load_data(data_path=data_path, error_bool=error_bool)
    
    clean_up_bad_data(data, verbose=verbose)

    convert_positions_and_velocities(data, zabs=zabs, GSR=GSR, R0=R0)

    if drop_unused:
        columns_to_drop = ['ra','dec','pmra','pmdec','GAIA_RUWE','good_delta_Gmag','pmb','pmlcosb']
        data.drop(columns = columns_to_drop, inplace=True)

    return data