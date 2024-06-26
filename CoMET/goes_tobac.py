#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:05:16 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running tobac on GOES data processed using goes_load.py
# =============================================================================



"""
Inputs:
    cube: iris cube containing the variable to be tracked
    tracking_type: ['IC','MCS','CP'] which type of tracking is being performed--either isolated convection, meso-scale convective systems, or cold pools
    CONFIG: User configuration file
Outputs:
    goes_geopd: geodataframe containing all default tobac feature id outputs
"""
def goes_tobac_feature_id(cube, tracking_type, CONFIG):
    import tobac
    import geopandas as gpd
    
    if (tracking_type.lower() == 'ic' or tracking_type.lower() == 'mcs'):
        
        # Get horozontal spacings in km then convert to m
        res = float(cube.attributes['spatial_resolution'].split("km")[0])*1000
        
        dxy = tobac.get_spacings(cube, grid_spacing=res)[0]
        
        # Perform tobac feature identification and then convert to a geodataframe before returning
        goes_radar_features = tobac.feature_detection.feature_detection_multithreshold(cube, dxy=dxy, **CONFIG['goes']['tobac']['feature_id'])
        
        if (type(goes_radar_features) == None):
            return None
        
        goes_geopd = gpd.GeoDataFrame(
            goes_radar_features, geometry = gpd.points_from_xy(goes_radar_features.longitude, goes_radar_features.latitude), crs="EPSG:4326"
        )

        return(goes_geopd)
    
    
    
    elif (tracking_type.lower() == 'cp'):
        print('==========================In-Progress==========================')
        return
    
    else:
        raise Exception(f'!=====Invalid Tracking Type. You Entered: {tracking_type.lower()}=====!')
        


"""
Inputs:
    cube: iris cube containing the variable to be tracked
    tracking_type: ['IC','MCS','CP'] which type of tracking is being performed--either isolated convection, meso-scale convective systems, or cold pools
    radar_features: tobac radar features from goes_tobac_feature_id output
    CONFIG: User configuration file
Outputs:
    goes_geopd_tracks: geodataframe containing all default tobac feature id outputs
"""
def goes_tobac_linking(cube, tracking_type, radar_features, CONFIG):
    import tobac
    import numpy as np
    import geopandas as gpd
    
    if (tracking_type.lower() == 'ic' or tracking_type.lower() == 'mcs'):
        
        # Get horozontal spacings in km then convert to m
        res = float(cube.attributes['spatial_resolution'].split("km")[0])*1000
        
        dxy = tobac.get_spacings(cube, grid_spacing=res)[0]
        
        # Get time spacing
        diffs = []
        for ii in range(cube.coord('time').points.shape[0]-1):
            diffs.append(cube.coord('time').points[ii+1] - cube.coord('time').points[ii])
        dt = np.nanmean(diffs) * 60
        
        # Do tracking then convert output dataframe to a geodataframe
        goes_tracks = tobac.linking_trackpy(radar_features,cube,dt=dt,dxy=dxy,**CONFIG['goes']['tobac']['linking'])
        
        if (type(goes_tracks) == None):
            return None
        
        goes_geopd_tracks = gpd.GeoDataFrame(
            goes_tracks, geometry = gpd.points_from_xy(goes_tracks.longitude, goes_tracks.latitude), crs="EPSG:4326"
        )
        
        return (goes_geopd_tracks)
    
    # Cold Pool Tracking is not yet implemented--wait for Beta version
    elif (tracking_type.lower() == 'cp'):
        print('==========================In-Progress==========================')
        return
        
    else:
        raise Exception(f'!=====Invalid Tracking Type. You Entered: {tracking_type.lower()}=====!')



"""
Inputs:
    cube: iris cube containing the variable to be tracked
    tracking_type: ['IC','MCS','CP'] which type of tracking is being performed--either isolated convection, meso-scale convective systems, or cold pools
    radar_features: tobac radar features from goes_tobac_feature_id output
    CONFIG: User configuration file
Outputs:
    (segment_array, segment_features): xarray DataArray containing segmented data and geodataframe with ncells row
"""
def goes_tobac_segmentation(cube, tracking_type, radar_features, CONFIG):
    import tobac
    import xarray as xr
    
    if (tracking_type.lower() == 'ic' or tracking_type.lower() == 'mcs'):
        
        # Check tracking var
        if (cube.name().lower() != 'toa_brightness_temperature'):
            raise Exception(f'!=====Invalid Tracking Variable. Your Cube Has: {cube.name().lower()}=====!')
            return
        
        # Get horozontal spacings in km then convert to m
        res = float(cube.attributes['spatial_resolution'].split("km")[0])*1000
        
        dxy = tobac.get_spacings(cube, grid_spacing=res)[0]
    
        # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
        segment_cube, segment_features = tobac.segmentation_2D(radar_features, cube, dxy=dxy, **CONFIG['goes']['tobac']['segmentation_2d'])
            
        # Convert iris cube to xarray and return
        return ((xr.DataArray.from_iris(segment_cube), segment_features))
            
    
    
    # Cold Pool Tracking is not yet implemented--wait for Beta version
    elif (tracking_type.lower() == 'cp'):
        print('==========================In-Progress==========================')
        return
        
    else:
        raise Exception(f'!=====Invalid Tracking Type. You Entered: {tracking_type.lower()}=====!')