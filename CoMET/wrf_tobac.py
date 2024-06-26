#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:26:17 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running tobac on WRF data processed using wrf_load.py
# =============================================================================



# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np
    
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx

"""
Inputs:
    cube: iris cube containing the variable to be tracked
    tracking_type: ['IC','MCS','CP'] which type of tracking is being performed--either isolated convection, meso-scale convective systems, or cold pools
    CONFIG: 
Outputs:
    wrf_geopd: geodataframe containing all default tobac feature id outputs
"""
def wrf_tobac_feature_id(cube, tracking_type, CONFIG):
    import tobac
    import geopandas as gpd
    
    if (tracking_type.lower() == 'ic' or tracking_type.lower() == 'mcs'):
        
        if ("height" in CONFIG['wrf']['tobac']['feature_id']): del CONFIG['wrf']['tobac']['feature_id']['height']
        
        # Get horozontal spacings
        dxy = tobac.get_spacings(cube)[0]
        
        # Perform tobac feature identification and then convert to a geodataframe before returning
        wrf_radar_features = tobac.feature_detection.feature_detection_multithreshold(cube, dxy=dxy, **CONFIG['wrf']['tobac']['feature_id'])
        
        if (wrf_radar_features is None):
            return None
        
        wrf_geopd = gpd.GeoDataFrame(
            wrf_radar_features, geometry = gpd.points_from_xy(wrf_radar_features.longitude, wrf_radar_features.latitude), crs="EPSG:4326"
        )

        return(wrf_geopd)
    
    
    
    elif (tracking_type.lower() == 'cp'):
        print('==========================In-Progress==========================')
        return
    
    else:
        raise Exception(f'!=====Invalid Tracking Type. You Entered: {tracking_type.lower()}=====!')
   
    
    
"""
Inputs:
    cube: iris cube containing the variable to be tracked
    tracking_type: ['IC','MCS','CP'] which type of tracking is being performed--either isolated convection, meso-scale convective systems, or cold pools
    radar_features: tobac radar features from wrf_tobac_feature_id output
    CONFIG: 
Outputs:
    wrf_geopd_tracks: geodataframe containing all default tobac tracking outputs
"""
def wrf_tobac_linking(cube, tracking_type, radar_features, CONFIG):
    import tobac
    import geopandas as gpd
    
    if (tracking_type.lower() == 'ic' or tracking_type.lower() == 'mcs'):
        
        dxy,dt = tobac.get_spacings(cube)
        
        # Do tracking then convert output dataframe to a geodataframe
        wrf_tracks = tobac.linking_trackpy(radar_features,cube,dt=dt,dxy=dxy,vertical_coord='altitude',**CONFIG['wrf']['tobac']['linking'])
        
        if (wrf_tracks is None):
            return None
        
        wrf_geopd_tracks = gpd.GeoDataFrame(
            wrf_tracks, geometry = gpd.points_from_xy(wrf_tracks.longitude, wrf_tracks.latitude), crs="EPSG:4326"
        )
        
        return (wrf_geopd_tracks)
    
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
    radar_features: tobac radar features from wrf_tobac_feature_id output
    segmentation_type: ['2D', '3D'], whether to perform 2d segmentation or 3d segmentation
    CONFIG: 
    segmentation_height: height, in meters, to perform the updraft or reflectivity segmentation if 2d selected and tracking_var != tb
Outputs:
    (segment_array, segment_features): xarray DataArray containing segmented data and geodataframe with ncells row
"""
def wrf_tobac_segmentation(cube, tracking_type, radar_features, segmentation_type, CONFIG, segmentation_height = None):
    import tobac
    import xarray as xr
    
    if (tracking_type.lower() == 'ic' or tracking_type.lower() == 'mcs'):
        
        # Enforce 2D tracking only for brightness temperature tracking
        if (cube.name().lower() == 'tb' and not segmentation_type.lower() == '2d'):
            raise Exception(f'!=====Invalid Segmentation Type. You Entered: {segmentation_type.lower()}. TB Tracking Restricted to 2D Segmentation=====!')
            return
        
        dxy = tobac.get_spacings(cube)[0]
    
        # 2D and 3D segmentation have different requirements so they are split up here
        if (segmentation_type.lower() == '2d'):
            
            if ("height" in CONFIG['wrf']['tobac']['segmentation_2d']): del CONFIG['wrf']['tobac']['segmentation_2d']['height']
            
            # If tracking var is tb, bypass height
            if (cube.name().lower() == 'tb'):
                # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
                segment_cube, segment_features = tobac.segmentation_2D(radar_features, cube, dxy=dxy,**CONFIG['wrf']['tobac']['segmentation_2d'])
                
                # Convert iris cube to xarray and return
                return ((xr.DataArray.from_iris(segment_cube), segment_features))
            
            
            # Ensure segmentation_height is a proper number before running
            if (segmentation_height == None or type(segmentation_height) == str or type(segmentation_height) == bool):
                raise Exception(f'!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height.lower()}=====!')
                return
            if (segmentation_height > cube.coord('altitude').points.max() or segmentation_height < cube.coord('altitude').points.min()):
                raise Exception(f'!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height.lower()}=====!')
                return
                
            
            # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
            height_index = find_nearest(cube.coord('altitude').points, segmentation_height)
            
            # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
            segment_cube, segment_features = tobac.segmentation_2D(radar_features, cube[:,height_index], dxy=dxy,**CONFIG['wrf']['tobac']['segmentation_2d'])
            
            # Convert iris cube to xarray and return
            return ((xr.DataArray.from_iris(segment_cube), segment_features))
        
        elif (segmentation_type.lower() == '3d'):
            
            # Similarly, perform 3d segmentation then return products
            segment_cube, segment_features = tobac.segmentation_3D(radar_features, cube, dxy=dxy,**CONFIG['wrf']['tobac']['segmentation_3d'])
               
            ## Convert iris cube to xarray and return
            return ((xr.DataArray.from_iris(segment_cube), segment_features))
    
        else:
            raise Exception(f'!=====Invalid Segmentation Type. You Entered: {segmentation_type.lower()}=====!')
            return
    
    
    
    # Cold Pool Tracking is not yet implemented--wait for Beta version
    elif (tracking_type.lower() == 'cp'):
        print('==========================In-Progress==========================')
        return
        
    else:
        raise Exception(f'!=====Invalid Tracking Type. You Entered: {tracking_type.lower()}=====!')