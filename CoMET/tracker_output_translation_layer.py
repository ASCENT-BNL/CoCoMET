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
        from tqdm import tqdm
        import geopandas as gpd
        from datetime import datetime
        
        # Extract values from features
        frames = tracks.frame.values
        times = np.array([datetime.fromisoformat(f.isoformat()) for f in tracks.time.values])
        lifetimes = tracks.time_cell.values
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
        
        
        lifetime_percents = []
        
        # Loop over rows
        for row in tqdm(tracks.iterrows(), desc="=====Performing tobac Linking to UDAF====", total=tracks.shape[0]):
            
            cell_max_life = tracks.query("cell==@row[1].cell").time_cell.values.max()
            
            # If only tracked one time, add -1 to lifetime_percent
            if cell_max_life == 0:
                lifetime_percents.append(-1)
            else:
                lifetime_percents.append(row[1].time_cell/cell_max_life)
                
        
        # Create GeoDataFrame according to UDAF specification
        UDAF_tracks = gpd.GeoDataFrame(data = {
            "frame": frames,
            "time": times,
            "lifetime": lifetimes,
            "lifetime_percent": lifetime_percents,
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
        from tqdm import tqdm
        from copy import deepcopy
        
        feature_segmentation = (segmentation - 1).rename("Feature_Segmentation")    
        cell_segmentation = deepcopy(feature_segmentation).rename("Cell_Segmentation")
        
        
        frame_groups = UDAF_tracks.groupby("frame")
        
        # Loop over tracks, replacing feature_id values with cell_id values in the cell_segmenation DataArray
        for frame in tqdm(frame_groups, desc="=====Performing tobac Segmentation to UDAF=====",total=frame_groups.ngroups):
            
            # Loop over each feature in that frame
            for feature in frame[1].iterrows():
                
                # Replace the feature_id with the cell_id
                cell_segmentation[frame[0]].values[cell_segmentation[frame[0]].values == feature[1].feature_id] = feature[1].cell_id
        
        # Combine into one xarray Dataset and return
        
        # To check for both prescense of altitude and shape of altitude without throwing DNE error
        altitude_check_bool = False
        if ("altitude" in segmentation.coords):
            
            if (segmentation.altitude.shape != ()): altitude_check_bool = True
        
        if ("model_level_number" in segmentation.coords):
        
            if (segmentation.model_level_number.shape != ()): altitude_check_bool = True
        
        # Check if altitude is present
        if (altitude_check_bool):    
            
            return_ds = xr.combine_by_coords([feature_segmentation,cell_segmentation])

            # Check for NEXRAD, and rename accordingly
            if ('y' in segmentation.dims and 'x' in segmentation.dims and "z" in segmentation.dims and 'lat' in segmentation.coords and 'lon' in segmentation.coords):
                
                return_ds = return_ds.assign_coords(altitude = ("z", feature_segmentation.z.values), up_down = ("z", np.arange(0,feature_segmentation.z.shape[0])))
                return_ds = return_ds.swap_dims({"z": "up_down", "y": "south_north", "x": "west_east"}).rename({"lat": "latitude", "lon": "longitude"})
                return_ds = return_ds.drop_vars(["z","x","y","model_level_number"])
                
                return(return_ds[["time","up_down","south_north","west_east","Feature_Segmentation","Cell_Segmentation"]])
            
            # For WRF case
            else:
                
                # Change altitude values to indices
                # return_ds = xr.combine_by_coords([feature_segmentation,cell_segmentation]).assign_coords({"up_down":np.arange(0,feature_segmentation.altitude.shape[0])})
                return_ds = return_ds.assign_coords(up_down = ("altitude", np.arange(0,feature_segmentation.altitude.shape[0])))
                
                
                # Remove extra coordinates and rename altitude
                return_ds = return_ds.swap_dims({"altitude": "up_down"}).drop_vars(["model_level_number","x","y"])
                return(return_ds[["time","up_down","south_north","west_east","Feature_Segmentation","Cell_Segmentation"]])
        
        else:
            
            # Concat into one dataset and remove superflous coordinates
            return_ds = xr.combine_by_coords([feature_segmentation,cell_segmentation])


            # Check for GOES and rename accordingly
            if ("sensor_band_bit_depth" in segmentation.attrs):
                
                # Rename t to time and rename x and y values to south_north and west_east, respectively. Rename lat and lon to latitude and longitude
                return_ds = return_ds.assign_coords(time = ("t", return_ds.t.values)).swap_dims({"t": "time"}).drop_vars(["t"])
                return_ds = return_ds.swap_dims({"y": "south_north", "x": "west_east"}).rename({"lat": "latitude", "lon": "longitude"})
                
                return_ds = return_ds.drop_vars(["x","y"])
                return(return_ds[["time","south_north","west_east","Feature_Segmentation","Cell_Segmentation"]])
             
                
            # Check for NEXRAD, and rename accordingly
            elif ('y' in segmentation.dims and 'x' in segmentation.dims and 'lat' in segmentation.coords and 'lon' in segmentation.coords):
                
                return_ds = return_ds.swap_dims({"y": "south_north", "x": "west_east"}).rename({"lat": "latitude", "lon": "longitude"})
                 
                return_ds = return_ds.drop_vars(["x","y"])
                return(return_ds[["time","south_north","west_east","Feature_Segmentation","Cell_Segmentation"]])
            
            # For WRF case
            else:
                
                return_ds = return_ds.drop_vars(["x","y"])
                return(return_ds[["time","south_north","west_east","Feature_Segmentation","Cell_Segmentation"]])
            
    
    else:
        raise Exception(f"!=====Invalid Tracker, You Entered: {tracker.lower()}=====!")