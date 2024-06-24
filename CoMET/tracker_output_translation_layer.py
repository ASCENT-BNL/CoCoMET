#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:34:22 2024

@author: thahn
"""

# =============================================================================
# Modifies tracking output into the CoMET-UDAF
# =============================================================================



"""
Inputs:
    features: The output from the feature detection step of a given tracker
    tracker: ["tobac"] The tracker used to generate the features
Outputs:
    UDAF_features: A geodataframe matching the format of the CoMET-UDAF feature detection specification
"""
def feature_id_to_UDAF(features, tracker):
    
    if (tracker.lower() == "tobac"):
        import numpy as np
        import geopandas as gpd
        from datetime import datetime
        
        # Extract values from features
        frames = features.frame.values
        times = np.array([datetime.fromisoformat(f.isoformat()) for f in features.time.values])
        feature_ids = features.feature.values-1
        north_souths = features.south_north.values
        east_wests = features.west_east.values
        latitudes = features.latitude.values
        longitudes = features.longitude.values
        projection_x = features.projection_x_coordinate.values
        projection_y = features.projection_y_coordinate.values
        geometries = features.geometry.values
        
        # Include 3D coordinates if present. If not, set all alt values as NaN
        if ("altitude" in features and "vdim" in features):
            altitudes = features.altitude.values
            up_downs = features.vdim.values
        
        else:
            altitudes = np.repeat(np.nan, features.shape[0])
            up_downs = np.repeat(np.nan, features.shape[0])
        
        
        # Create GeoDataFrame according to UDAF specification
        UDAF_features = gpd.GeoDataFrame(data = {
            "frame": frames,
            "time": times,
            "feature_id": feature_ids,
            "north_south": north_souths,
            "east_west": east_wests,
            "up_down": up_downs,
            "latitude": latitudes,
            "longitude": longitudes,
            "projection_x": projection_x,
            "projection_y": projection_y,
            "altitude": altitudes,
            "geometry": geometries
            })
        
        return UDAF_features
    
    else:
        raise Exception(f"!=====Invalid Tracker, You Entered: {tracker.lower()}=====!")



"""
Inputs:
    tracks: The output from the linking/tracking step of a given tracker
    tracker: ["tobac"] The tracker used to generate the features
Outputs:
    UDAF_tracks: A geodataframe matching the format of the CoMET-UDAF linking specification
"""
def linking_to_UDAF(tracks, tracker):
    
    if (tracker.lower() == "tobac"):
        import numpy as np
        import geopandas as gpd
        from datetime import datetime
        
        # Extract values from features
        frames = tracks.frame.values
        times = np.array([datetime.fromisoformat(f.isoformat()) for f in tracks.time.values])
        feature_ids = tracks.feature.values-1
        cell_ids = tracks.cell.values-1
        # Correct any -2 values, created as a result of shifting, back to -1
        cell_ids[cell_ids == -2]= -1
        north_souths = tracks.south_north.values
        east_wests = tracks.west_east.values
        latitudes = tracks.latitude.values
        longitudes = tracks.longitude.values
        projection_x = tracks.projection_x_coordinate.values
        projection_y = tracks.projection_y_coordinate.values
        geometries = tracks.geometry.values
        
        # Include 3D coordinates if present. If not, set all alt values as NaN
        if ("altitude" in tracks and "vdim" in tracks):
            altitudes = tracks.altitude.values
            up_downs = tracks.vdim.values
        
        else:
            altitudes = np.repeat(np.nan, tracks.shape[0])
            up_downs = np.repeat(np.nan, tracks.shape[0])
        
        
        # Create GeoDataFrame according to UDAF specification
        UDAF_tracks = gpd.GeoDataFrame(data = {
            "frame": frames,
            "time": times,
            "feature_id": feature_ids,
            "cell_id": cell_ids,
            "north_south": north_souths,
            "east_west": east_wests,
            "up_down": up_downs,
            "latitude": latitudes,
            "longitude": longitudes,
            "projection_x": projection_x,
            "projection_y": projection_y,
            "altitude": altitudes,
            "geometry": geometries
            })
        
        return UDAF_tracks
    
    else:
        raise Exception(f"!=====Invalid Tracker, You Entered: {tracker.lower()}=====!")



"""
Inputs:
    segmentation: The output from the segmentation step of a given tracker
    tracker: ["tobac"] The tracker used to generate the features
Outputs:
    UDAF_segmentation: An xarray DataArray matching the format of the CoMET-UDAF segmentation specification
"""
def segmentation_to_UDAF(segmentation, tracker):
    print("=====In Progress=====")