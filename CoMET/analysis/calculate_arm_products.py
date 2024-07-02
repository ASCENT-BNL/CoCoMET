#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:55:18 2024

@author: thahn
"""

# =============================================================================
# This file contains the functions used to calculate statistics and data related to ARM products
# =============================================================================



# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np
    
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx



"""
Inputs:
    analysis_object: A CoMET-UDAF standard analysis object containing at least UDAF_tracks
    path_to_files: A glob-like path to the VDISQUANTS ARM product output
    verbose: Determins if output should be printed during processing or not
Outputs:
    vap_info: A pandas dataframe with the following rows: frame, tracking_time, vdisquants_time, time_delta, closest_feature_id (km), rain_rate (mm/hr), total_droplet_concentration (1/m^3),
            sband_estimated_reflectivity (dBZ), cband_estimated_reflectivity (dBZ), xband_estimated_reflectivity (dBZ)
"""
def calculate_arm_vdisquants(analysis_object, path_to_files, verbose=False, **args):
    import numpy as np
    import xarray as xr
    import pandas as pd
    from vincenty import vincenty
    
    # Open video disdrometer product
    vap = xr.open_mfdataset(path_to_files, coords='all', concat_dim='time', combine='nested')
    
    # First get the position of the video disdrometer
    lat_pos = vap.lat.values[0]
    lon_pos = vap.lon.values[0]
    
    vap_info = {
        "frame": [],
        "tracking_time": [],
        "vdisquants_time": [],
        "time_delta": [],
        "closest_feature_id": [],
        "closest_cell_id": [],
        "distance_to_closest_feature": [],
        "rain_rate": [],
        "total_droplet_concentration": [],
        "sband_estimated_reflectivity": [],
        "cband_estimated_reflectivity": [],
        "xband_estimated_reflectivity": []
    }
    
    # Loop over frames
    for ii, frame in enumerate(analysis_object['UDAF_linking'].groupby("frame")):
        
        if (verbose): print(f"=====Calculating VIDSQUANT Products. {'%.2f' % ((ii+1)/len(np.unique(analysis_object['UDAF_linking'].frame))*100)}% Complete=====")
        
        # Get VAP at current time step
        time_idx = find_nearest(vap.time.values, frame[1].time.values[0])
        time_delta = abs(vap.time.values[time_idx]-frame[1].time.values[0])
        
        feature_id = []
        cell_distance = []
        
        # Loop over features
        for feature in frame[1].groupby("feature_id"):   
        
            # Get distance from each feature to the video disdrometer
            dis_to_vdis = vincenty((lat_pos,lon_pos), (feature[1].latitude.values,feature[1].longitude.values))
            cell_distance.append(dis_to_vdis)
            feature_id.append(feature[0])
        
        closest_feature_id = feature_id[np.where(cell_distance == np.nanmin(cell_distance))[0][0]]
        closest_feature = frame[1].query("feature_id==@closest_feature_id")
        
        
        vap_info["frame"].append(frame[0])
        vap_info["tracking_time"].append(frame[1].time.values[0])
        vap_info["vdisquants_time"].append(vap.time.values[time_idx])
        vap_info["time_delta"].append(time_delta)
        vap_info["closest_feature_id"].append(closest_feature.feature_id.values[0])
        vap_info["closest_cell_id"].append(closest_feature.cell_id.values[0])
        vap_info["distance_to_closest_feature"].append(np.nanmin(cell_distance))
        vap_info["rain_rate"].append(vap.rain_rate.values[time_idx])
        vap_info["total_droplet_concentration"].append(vap.total_droplet_concentration.values[time_idx])
        vap_info["sband_estimated_reflectivity"].append(vap.reflectivity_factor_sband20c.values[time_idx])
        vap_info["cband_estimated_reflectivity"].append(vap.reflectivity_factor_cband20c.values[time_idx])
        vap_info["xband_estimated_reflectivity"].append(vap.reflectivity_factor_xband20c.values[time_idx])
        
        
    return(pd.DataFrame(vap_info))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        