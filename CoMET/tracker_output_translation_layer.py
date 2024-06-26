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
    
    if (features is None): return None
    
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
            "south_north": north_souths,
            "west_east": east_wests,
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
    
    if (tracks is None): return None
    
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
            "south_north": north_souths,
            "west_east": east_wests,
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
    UDAF_tracks: UDAF standard tracking output
    tracker: ["tobac"] The tracker used to generate the features
Outputs:
    UDAF_segmentation: An xarray DataArray matching the format of the CoMET-UDAF segmentation specification
"""
def segmentation_to_UDAF(segmentation, UDAF_tracks, tracker):
    
    if (segmentation is None or UDAF_tracks is None): return None
    
    if (tracker.lower() == "tobac"):
        import numpy as np
        import xarray as xr
        from copy import deepcopy
        
        feature_segmentation = (segmentation - 1).rename("Feature_Segmentation")    
        cell_segmentation = (deepcopy(feature_segmentation) - 1).rename("Cell_Segmentation")
        
        # Loop over tracks, replacing feature_id values with cell_id values in the cell_segmenation DataArray
        for frame in UDAF_tracks.groupby("frame"):
            
            # Loop over each feature in that frame
            for feature in frame[1].iterrows():
                
                # Replace the feature_id with the cell_id
                cell_segmentation[frame[0]].values[cell_segmentation[frame[0]].values == feature[1].feature_id] = feature[1].cell_id
        
        # Combine into one xarray Dataset and return
        
        # To check for both prescense of altitude and shape of altitude without throwing DNE error
        altitude_check_bool = False
        if ("altitude" in feature_segmentation.coords):
            
            if (feature_segmentation.altitude.shape != ()): altitude_check_bool = True
        
        # Check if altitude is present
        if (altitude_check_bool):    

            # Change altitude values to indices
            # return_ds = xr.combine_by_coords([feature_segmentation,cell_segmentation]).assign_coords({"up_down":np.arange(0,feature_segmentation.altitude.shape[0])})
            return_ds = xr.combine_by_coords([feature_segmentation,cell_segmentation]).assign_coords(altitude = ("altitude", np.arange(0,feature_segmentation.altitude.shape[0])))
            
            
            # Remove extra coordinates and rename altitude
            return_ds = return_ds.rename_dims({"altitude": "up_down"}).drop_vars(["model_level_number","x","y","altitude"])
            
            return(return_ds[["time","up_down","south_north","west_east","Feature_Segmentation","Cell_Segmentation"]])
        
        else:
            
            # Concat into one dataset and remove superflous coordinates
            return_ds = xr.combine_by_coords([feature_segmentation,cell_segmentation])
            return_ds = return_ds.drop_vars(["x","y"])
            
            return(return_ds)
            
    
    else:
        raise Exception(f"!=====Invalid Tracker, You Entered: {tracker.lower()}=====!")