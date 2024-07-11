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
    output_data: An xarray Dataset with the following: frame, tracking_time, vdisquants_time, time_delta, closest_feature_id (km), rain_rate (mm/hr), total_droplet_concentration (1/m^3),
            sband_estimated_reflectivity (dBZ), cband_estimated_reflectivity (dBZ), xband_estimated_reflectivity (dBZ)
"""
def calculate_arm_vdisquants(analysis_object, path_to_files, verbose=False, **args):
    import numpy as np
    import xarray as xr
    from tqdm import tqdm
    from vincenty import vincenty
    
    # Open video disdrometer product
    vap = xr.open_mfdataset(path_to_files, coords="all", concat_dim="time", combine="nested")
    
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
    
    frame_groups = analysis_object["UDAF_linking"].groupby("frame")
    
    # Loop over frames
    for ii, frame in tqdm(enumerate(frame_groups), desc="=====Calculating VIDSQUANTS Data=====",total=frame_groups.ngroups):
        
        # Get VAP at current time step
        time_idx = find_nearest(vap.time.values, frame[1].time.values[0])
        time_delta = abs(vap.time.values[time_idx]-frame[1].time.values[0])
        
        # Get the position of the video disdrometer
        lat_pos = vap.lat.values[time_idx]
        lon_pos = vap.lon.values[time_idx]
        
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
        
    
    # Create output Dataset
    output_data = xr.Dataset(
        coords = dict(
            frame=("time", vap_info["frame"]),
            tracking_time=("time", vap_info["tracking_time"]),
            vdisquants_time=("time", vap_info["vdisquants_time"]),
        ),
        data_vars=dict(
            time_delta=("time", vap_info["time_delta"]),
            closest_feature_id=("time", vap_info["closest_feature_id"]),
            closest_cell_id=("time",vap_info["closest_cell_id"]),
            distance_to_closest_feature=("time",vap_info["distance_to_closest_feature"]),
            rain_rate=("time",vap_info["rain_rate"]),
            total_droplet_concentration=("time",vap_info["total_droplet_concentration"]),
            sband_estimated_reflectivity=("time",vap_info["sband_estimated_reflectivity"]),
            cband_estimated_reflectivity=("time",vap_info["cband_estimated_reflectivity"]),
            xband_estimated_reflectivity=("time",vap_info["xband_estimated_reflectivity"]),
        ),
        attrs=dict(
            description="Tracking Linked VDISQUANTS Data"
        )
    )
    
    return(output_data)



"""
Inputs:
    analysis_object: A CoMET-UDAF standard analysis object containing at least UDAF_tracks
    path_to_files: A glob-like path to the VDISQUANTS ARM product output
    verbose: Determins if output should be printed during processing or not
Outputs:
    output_data: An xarray Dataset with the following: frame, tracking_time, vdisquants_time, height (m above MSL), time_delta, closest_feature_id (km), temperature (C), relative_humidity (%),
                                                        barometric_pressure (hPA), wind_speed (m/s), wind_direction (degrees), northward_wind (m/s), eastward_wind (m/s)
"""
def calculate_arm_interpsonde(analysis_object, path_to_files, verbose=False, **args):
    import numpy as np
    import xarray as xr
    from tqdm import tqdm
    from vincenty import vincenty
    
    # Open video disdrometer product
    sonde = xr.open_mfdataset(path_to_files, coords="all", concat_dim="time", combine="nested")
    
    sonde_info = {
        "frame": [],
        "tracking_time": [],
        "sonde_time": [],
        "time_delta": [],
        "closest_feature_id": [],
        "closest_cell_id": [],
        "distance_to_closest_feature": [],
        "temperature": [],
        "relative_humidity": [],
        "barometric_pressure": [],
        "wind_speed": [],
        "wind_direction": [],
        "northward_wind": [],
        "eastward_wind": []
    }
    
    frame_groups = analysis_object["UDAF_linking"].groupby("frame")
    
    # Loop over frames
    for ii, frame in tqdm(enumerate(frame_groups), desc="=====Calculating INTERPSONDE Data=====",total=frame_groups.ngroups):
        
        # Get VAP at current time step
        time_idx = find_nearest(sonde.time.values, frame[1].time.values[0])
        time_delta = abs(sonde.time.values[time_idx]-frame[1].time.values[0])
        
        # Get the position of the sonde
        lat_pos = sonde.lat.values[time_idx]
        lon_pos = sonde.lon.values[time_idx]
        
        feature_id = []
        cell_distance = []
        
        # Loop over features
        for feature in frame[1].groupby("feature_id"):   
        
            # Get distance from each feature to the video disdrometer
            dis_to_sonde = vincenty((lat_pos,lon_pos), (feature[1].latitude.values,feature[1].longitude.values))
            cell_distance.append(dis_to_sonde)
            feature_id.append(feature[0])
        
        closest_feature_id = feature_id[np.where(cell_distance == np.nanmin(cell_distance))[0][0]]
        closest_feature = frame[1].query("feature_id==@closest_feature_id")
        
        
        sonde_info["frame"].append(frame[0])
        sonde_info["tracking_time"].append(frame[1].time.values[0])
        sonde_info["sonde_time"].append(sonde.time.values[time_idx])
        sonde_info["time_delta"].append(time_delta)
        sonde_info["closest_feature_id"].append(closest_feature.feature_id.values[0])
        sonde_info["closest_cell_id"].append(closest_feature.cell_id.values[0])
        sonde_info["distance_to_closest_feature"].append(np.nanmin(cell_distance))
        sonde_info["temperature"].append(sonde.temp.values[time_idx])
        sonde_info["relative_humidity"].append(sonde.rh.values[time_idx])
        sonde_info["barometric_pressure"].append(sonde.bar_pres.values[time_idx] * 10)
        sonde_info["wind_speed"].append(sonde.wspd.values[time_idx])
        sonde_info["wind_direction"].append(sonde.wdir.values[time_idx])
        sonde_info["northward_wind"].append(sonde.v_wind.values[time_idx])
        sonde_info["eastward_wind"].append(sonde.u_wind.values[time_idx])
    
    
    # Create output Dataset
    output_data = xr.Dataset(
        coords = dict(
            frame=("time", sonde_info["frame"]),
            tracking_time=("time", sonde_info["tracking_time"]),
            sonde_time=("time", sonde_info["sonde_time"]),
            height=sonde.height.values * 1000 #TODO: Adjust to be AGL instead of above MSL
        ),
        data_vars=dict(
            time_delta=("time", sonde_info["time_delta"]),
            closest_feature_id=("time", sonde_info["closest_feature_id"]),
            closest_cell_id=("time",sonde_info["closest_cell_id"]),
            distance_to_closest_feature=("time",sonde_info["distance_to_closest_feature"]),
            temperature=(["time","height"],sonde_info["temperature"]),
            relative_humidity=(["time","height"],sonde_info["relative_humidity"]),
            barometric_pressure=(["time","height"],sonde_info["barometric_pressure"]),
            wind_speed=(["time","height"],sonde_info["wind_speed"]),
            wind_direction=(["time","height"],sonde_info["wind_direction"]),
            northward_wind=(["time","height"],sonde_info["northward_wind"]),
            eastward_wind=(["time","height"],sonde_info["eastward_wind"])
        ),
        attrs=dict(
            description="Tracking Linked INTERPSONDE Data"
        )
    )
    
    return (output_data)
