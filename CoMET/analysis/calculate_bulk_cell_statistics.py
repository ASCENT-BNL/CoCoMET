#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:00:59 2024

@author: thahn
"""

# =============================================================================
# This contains all of the functions to calculate basic bulk cell statistics (ETH, area, etc.) from CoMET-UDAF data
# =============================================================================

import logging

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from copy import deepcopy


# Calculate nearest item in list to given pivot
def find_nearest(array: np.ndarray, pivot) -> int:

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx

def calculate_ETH(
        analysis_object: dict, 
        threshold: float,
        variable: str | None = None,
        cell_footprint_height: float = 2,
        quantile: float = 0.95,
        **args: dict,
) -> pd.DataFrame | None:
    """
    The same thing as calculate_max_height, but must segment on dbz.
    """

    if 'DBZ' not in analysis_object['segmentation_xarray']:
        raise ValueError('!=====Need DBZ calculation for ETH=====!')
    
    if len(args) != 0:
        return calculate_max_height(analysis_object, threshold, variable, cell_footprint_height, quantile, True, **args)
    else: 
        return calculate_max_height(analysis_object, threshold, variable, cell_footprint_height, quantile, isETH = True, )
    
def calculate_max_height(
    analysis_object: dict,
    threshold: float,
    variable: str | None = None,
    cell_footprint_height: float = 2,
    quantile: float = 0.95,
    isETH: bool = False,
    **args: dict,
) -> pd.DataFrame | None:
    """

    
    Parameters
    ----------
    analysis_object : dict
        A  CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_2d or UDAF_segmentation_3d, and segmentation_xarray.
    threshold : float
        The value which needs to be exceeded to count towards the var top height. I.e. 15 for reflectivity.
    variable : str, optional
        The variable from the input segmentation_xarray which should be used for calculating var_max_height. The default is None.
    cell_footprint_height : float, optional
        The height used to calculate the cell area to determine where to calculate var_max_heights. The default is 2km.
    quantile : float, optional
        The percentile of calculated max heights to return. The default is 0.95.
    isETH: float, optional
        A boolean indicating whether the calculation is for ETH. The default is False
    **args : dict
        Throw away inputs.

    Raises
    ------
    Exception
        Exception if missing segmentation data from the analysis object.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, max_height or eth where max_height or eth is in km.

    """

    # If input variable field is 2D return None. Also, if DataArray, use those values for calculations. If Dataset, use tracking_var to get variable
    if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:

        if len(analysis_object["segmentation_xarray"].shape) != 4:
            logging.warning("!=====Input Variable is not 3D=====!")
            return None

        else:
            variable_field = analysis_object["segmentation_xarray"]

    else:

        if len(analysis_object["segmentation_xarray"][variable].shape) != 4:
            logging.warning("!=====Input Variable is not 3D=====!")
            return None

        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

    # If 3D segmentation is available, use that to calculate cell footprint, otherwise use 2D segmentation
    if analysis_object["UDAF_segmentation_3d"] is not None:

        height_index = find_nearest(
            analysis_object["UDAF_segmentation_3d"].altitude.values,
            cell_footprint_height * 1000,
        )

        footprint_data = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation[
            :, height_index
        ]

    elif analysis_object["UDAF_segmentation_2d"] is not None:

        footprint_data = analysis_object["UDAF_segmentation_2d"].Feature_Segmentation

    else:
        raise Exception("!=====Missing Segmentation Input=====!")

    if isETH:
        printout_string = ('ETH', 'eth')
    else:
        printout_string = ('Max Heights', 'max_height')

    max_height_info = {"frame": [], "feature_id": [], "cell_id": [], printout_string[1]: []}  # in km

    frame_groups = analysis_object["UDAF_tracks"].groupby("frame")

    # Loop over frames
    for ii, frame in tqdm(
        enumerate(frame_groups),
        desc=f"=====Calculating {printout_string[0]}=====",
        total=frame_groups.ngroups,
    ):

        # Loop over each feature
        for feature in frame[1].groupby("feature_id"):

            # Get the indices of the cell footprint
            proper_indices = np.argwhere(footprint_data[frame[0]].values == feature[0])

            # Cells which have no segmented output should get a NaN
            if len(proper_indices) == 0:

                max_height_info["frame"].append(frame[0])
                max_height_info["feature_id"].append(feature[0])
                max_height_info["cell_id"].append(feature[1]["cell_id"].min())
                max_height_info[printout_string[1]].append(np.nan)
                continue

            max_height_set = []

            # Calculate ETH for each location
            for iy, ix in proper_indices:

                max_alt_index = np.argwhere(
                    variable_field[frame[0], :, iy, ix].values > threshold
                )

                # If there are no indices greater than threshold, skip
                if len(max_alt_index) != 0:
                    max_alt_index = max_alt_index.max()
                else:
                    max_height_set.append(np.nan)
                    continue

                max_alt = variable_field.altitude.values[max_alt_index]
                max_height_set.append(max_alt)

            max_height_info["frame"].append(frame[0])
            max_height_info["feature_id"].append(feature[0])
            max_height_info["cell_id"].append(feature[1]["cell_id"].min())

            # If all NaN slice, append just np.nan
            if np.isnan(max_height_set).all():
                max_height_info[printout_string[1]].append(np.nan)
            else:
                max_height_info[printout_string[1]].append(np.nanquantile(max_height_set, quantile) / 1000)

    return pd.DataFrame(max_height_info)

def calculate_max_intensity(
    analysis_object: dict, 
    variable: str | None = None, 
    cell_footprint_height: float = 2,
    quantile: float = 0.95,
    **args
    ) -> pd.DataFrame | None:
    """
    

    Parameters
    ----------
    analysis_object : dict
        A  CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_2d or UDAF_segmentation_3d, and segmentation_xarray.
    variable : str, optional
        The variable from the input segmentation_xarray which should be used for calculating var_max_height. The default is None.
    cell_footprint_height : float, optional
        The height used to calculate the cell area to determine where to calculate var_max_heights. The default is 2km.
    quantile : float, optional
        The percentile of calculated max heights to return. The default is 0.95.
    **args : dict
        Throw away inputs.

    Raises
    ------
    Exception
        Exception if missing segmentation data from the analysis object.
        Exception if there is more than one frame associated with a feature id

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, max_intensity where max_intensity is in the unit of the variable.

    """

    # If DataArray, use those values for calculations. If Dataset, use tracking_var to get variable
    if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
        variable_field = analysis_object["segmentation_xarray"]

    else:
        variable_field = analysis_object["segmentation_xarray"][variable]

    # If 3D segmentation is available, use that to calculate cell footprint, otherwise use 2D segmentation
    if analysis_object["UDAF_segmentation_3d"] is not None:

        height_index = find_nearest(            
            analysis_object["UDAF_segmentation_3d"].altitude.values,
            cell_footprint_height * 1000,
            )
        
        segmentation = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation
        features_across_footprint = np.unique(segmentation[:, height_index])[1:]

    elif analysis_object["UDAF_segmentation_2d"] is not None:

        segmentation = analysis_object["UDAF_segmentation_2d"].Feature_Segmentation
        features_across_footprint = np.unique(segmentation)[1:]

    else:
        raise Exception("!=====Missing Segmentation Input=====!")
    
    max_intensity_info = {"frame" : [], "feature_id" : [], "cell_id" : [], "max_intensity" : []}
    Tracks = analysis_object['UDAF_tracks']    
    
    # Loop over each frame, then apply a mask for the cell/feature, then within that find the highest val
    for feature_id, feat_group in tqdm(Tracks.groupby("feature_id"),
                                    desc="=====Calculating Max Intensity=====",
                                    total=len(Tracks.groupby("feature_id"))
    ):
        frame = feat_group['frame'].values[0]
        if len(feat_group['frame'].values) > 1:
            raise Exception("More than one frame found per feature")
        
        if feature_id in features_across_footprint:
            one_feat_array = np.where(segmentation[frame].values == feature_id, variable_field[frame].values, 0)
            max_intensity = np.nanmax(one_feat_array)
        else:
            max_intensity = np.nan

        max_intensity_info["frame"].append(frame)
        max_intensity_info["feature_id"].append(feature_id)
        max_intensity_info["cell_id"].append(feat_group['cell_id'].values[0])

        # TODO: add in quantile to reduce outlier skewing
        max_intensity_info["max_intensity"].append(max_intensity)

        # if np.array(max_intensity_list).all() is np.nan:
        #     max_intensity_info["max_intensity"].extend([np.nan] * len(max_intensity_list))
        # else:
        #     max_intensity_info["max_intensity"].append(np.nanquantile(max_intensity_list, quantile) / 1000)

    return pd.DataFrame(max_intensity_info)

def calculate_area(
    analysis_object: dict,
    height: float = 2,
    variable: str | None = None,
    threshold: float | None = None,
    **args: dict,
) -> pd.DataFrame:
    """


    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_2d or UDAF_segmentation_3d.
    height : float, optional
        The height which is used to calculate the area of cells. The default is 2km.
    variable : str, optional
        Variable to which we should apply the threshold. The default is None.
    threshold : float, optional
        Value of which the area should be greater than. The default is None.
    **args : dict
        Throw away variables.

    Raises
    ------
    Exception
        Exception if missing segmentation input from the analysis object.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, area where area is in km^2.

    """

    # If 3D segmentation is available, use that at given height, otherwise use 2D segmentation
    if analysis_object["UDAF_segmentation_3d"] is not None:

        height_index = find_nearest(
            analysis_object["UDAF_segmentation_3d"].altitude.values, height * 1000
        )

        mask = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation[
            :, height_index
        ]

    elif analysis_object["UDAF_segmentation_2d"] is not None:

        mask = analysis_object["UDAF_segmentation_2d"].Feature_Segmentation

    else:
        raise Exception("!=====Missing Segmentation Input=====!")

    # Enforce threshold if present
    if threshold is not None:
        # Load background varaible
        if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
            variable_field = analysis_object["segmentation_xarray"]
        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

        if len(variable_field.shape) == 4:
            height_index = find_nearest(
                analysis_object["segmentation_xarray"]["altitude"].values, height * 1000
            )

            variable_field = variable_field[:, height_index]

        mask.values[variable_field.values < threshold] = -1

    area_info = {"frame": [], "feature_id": [], "cell_id": [], "area": []}  # in km^2

    # We first calculate the area of each individual cell
    # First get the size of each dimension
    x_dim_sizes = np.abs(np.diff(mask.projection_x_coordinate))
    y_dim_sizes = np.abs(np.diff(mask.projection_y_coordinate))

    # These are one cell too small due to how diff works, so infer last cell size using the same cell size as the previous cell
    # x_dim_sizes.append(x_dim_sizes[-1])
    x_dim_sizes = np.append(x_dim_sizes, x_dim_sizes[-1])
    y_dim_sizes = np.append(y_dim_sizes, y_dim_sizes[-1])

    # Multiply each cell by the other to get an area for one individual cell
    cell_areas = np.outer(y_dim_sizes, x_dim_sizes)

    frame_groups = analysis_object["UDAF_tracks"].groupby("frame")

    # Loop over frames
    for ii, frame in tqdm(
        enumerate(frame_groups),
        desc="=====Calculating Areas=====",
        total=frame_groups.ngroups,
    ):

        # Loop over each feature
        for feature in frame[1].groupby("feature_id"):

            # Get valid indices of a given features
            proper_indices = np.argwhere(mask[frame[0]].values == feature[0])

            # Cells which have no segmented output should get a NaN
            if len(proper_indices) == 0:

                area_info["frame"].append(frame[0])
                area_info["feature_id"].append(feature[0])
                area_info["cell_id"].append(feature[1]["cell_id"].min())
                area_info["area"].append(np.nan)
                continue

            # Sum up all the areas that comprise it
            total = np.sum([cell_areas[iy, ix] for iy, ix in proper_indices])

            # Push info to dictionary
            area_info["frame"].append(frame[0])
            area_info["feature_id"].append(feature[0])
            area_info["cell_id"].append(feature[1]["cell_id"].min())
            area_info["area"].append(total / 1e6)

    return pd.DataFrame(area_info)


def calculate_volume(
    analysis_object: dict,
    variable: str | None = None,
    threshold: float | None = None,
    **args: dict,
) -> pd.DataFrame:
    """


    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_3d.
    variable : str, optional
        Variable to which we should apply the threshold. The default is None.
    threshold : float, optional
        Value of which the area should be greater than. The default is None.
    **args : dict
        Throw away variables.

    Raises
    ------
    Exception
        Exception if missing segmentation input from the analysis object.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, volume where area is in km^3.

    """

    if analysis_object["UDAF_segmentation_3d"] is None:
        raise Exception(
            "!=====3D Segmentation Data is Required for Volume Calculation=====!"
        )

    mask = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation

    if threshold is not None:
        # Load background varaible
        if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
            variable_field = analysis_object["segmentation_xarray"]
        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

        if len(variable_field.shape) != 4:
            raise Exception(
                "=====Must have a 3D segmentation_xarray for threshold application====="
            )

        # Enforce threshold
        mask.values[variable_field.values < threshold] = -1

    volume_info = {
        "frame": [],
        "feature_id": [],
        "cell_id": [],
        "volume": [],  # in km^3
    }

    # We first calculate the area of each individual cell
    # First get the size of each dimension
    x_dim_sizes = np.diff(mask.projection_x_coordinate)
    y_dim_sizes = np.diff(mask.projection_y_coordinate)
    z_dim_sizes = np.diff(mask.altitude)

    # These are one cell too small due to how diff works, so infer last cell size using the same cell size as the previous cell
    # x_dim_sizes.append(x_dim_sizes[-1])
    x_dim_sizes = np.append(x_dim_sizes, x_dim_sizes[-1])
    y_dim_sizes = np.append(y_dim_sizes, y_dim_sizes[-1])
    z_dim_sizes = np.append(z_dim_sizes, z_dim_sizes[-1])

    # use Einstein sum notation to get volume of cells
    cell_volumes = np.einsum("i,j,k->ijk", z_dim_sizes, y_dim_sizes, x_dim_sizes)

    frame_groups = analysis_object["UDAF_tracks"].groupby("frame")

    # Loop over frames
    for ii, frame in tqdm(
        enumerate(frame_groups),
        desc="=====Calculating Volumes=====",
        total=frame_groups.ngroups,
    ):

        # Loop over each feature
        for feature in frame[1].groupby("feature_id"):

            # Get valid indices of a given features
            proper_indices = np.argwhere(mask[frame[0]].values == feature[0])

            # Cells which have no segmented output should get a NaN
            if len(proper_indices) == 0:

                volume_info["frame"].append(frame[0])
                volume_info["feature_id"].append(feature[0])
                volume_info["cell_id"].append(feature[1]["cell_id"].min())
                volume_info["volume"].append(np.nan)
                continue

            # Sum up all the volumes that comprise it
            total = np.sum([cell_volumes[iz, iy, ix] for iz, iy, ix in proper_indices])

            # Push info to dictionary
            volume_info["frame"].append(frame[0])
            volume_info["feature_id"].append(feature[0])
            volume_info["cell_id"].append(feature[1]["cell_id"].min())
            volume_info["volume"].append(total / 1e9)

    return pd.DataFrame(volume_info)

def calculate_velocity(
        analysis_object: dict,
        variable: str | None = None,
        threshold: float | None = None,
        **args: dict,
    ) -> pd.DataFrame:
    """


    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks and tracking_xarray.
    variable : str, optional
        Variable to which we should apply the threshold. The default is None.
    threshold : float, optional
        Value of which the area should be greater than. The default is None.
    **args : dict
        Throw away variables.

    Raises
    ------
    Exception
        Exception if missing tracks input from the analysis object.
        Exception if the segmentation data is not 2D or 3D.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, velocity where area is in m/s or m/frame.

    """

    if analysis_object['UDAF_tracks'] is None:
        raise Exception(
            "!=====Tracks Data is Required for Velocity Calculation=====!"
            )
    
    # Check if the code is 3D
    if analysis_object["UDAF_segmentation_3d"]is not None:
        dim = 3
    elif analysis_object["UDAF_segmentation_2d"] is not None:
        dim = 2
    else:
        raise Exception("!=====Missing Segmentation Input=====!")

    # Get dt
    if ("tracking_xarray" in analysis_object) and ("DT" in analysis_object['tracking_xarray'].attrs):
        dt_from_tracking = analysis_object['tracking_xarray'].attrs["DT"] # in s
    else:
        dt_from_tracking = 1.
        logging.warning("Could not find DT in dataset, computing velocity in m/frame")

    def calculate_row_velocity(row1, row2, dt_arr, dim):
        #given 2 rows of the tracks dataframe, compute the distance between the 2 in 3 components:
        x1 = row1['projection_x']
        x2 = row2['projection_x']
        y1 = row1['projection_y']
        y2 = row2['projection_y']

        f2 = row1['frame']
        f1 = row2['frame']

        dt = np.sum(dt_arr[f1:f2]) # If a cell is not tracked for a frame, account for this (e.x. cell is in frame 1 and 3)

        distancecomps = ((y2-y1), (x2-x1))
        distance = np.sqrt(distancecomps[0]**2 + distancecomps[1]**2)

        velocity = ((y2-y1) / dt, (x2-x1) / dt)
        speed = np.sqrt((velocity[0])**2 + (velocity[1])**2)

        if dim == 3:
            z1 = row1['altitude']
            z2 = row2['altitude']

            distancecomps = ((z2-z1), (y2-y1), (x2-x1))
            distance = np.sqrt(distancecomps[0]**2 + distancecomps[1]**2 + distancecomps[2]**2)

            velocity = ((z2-z1) / dt, (y2-y1) / dt, (x2-x1) / dt)
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
        
        return (distancecomps, distance, velocity, speed)
    
    Tracks = deepcopy(analysis_object['UDAF_tracks'])
    velocity_info = {
        "frame": [],
        "feature_id": [],
        "cell_id": [],
        "velocity": [],  # in m / s
        "speed": [], # in m / s
    }

    # Make dt into an array
    if type(dt_from_tracking) == list or type(dt_from_tracking) == np.ndarray:
        # Check that the number of frames and the number of dt's are the same
        dt_array = list(dt_from_tracking) # append the last value from the list
        dt_array.append(dt_from_tracking[-1])
    else:
        dt_array = np.ones( len(np.unique(Tracks['frame'])) + 1) * dt_from_tracking
        
    for i in tqdm(Tracks.index,
                  desc='=====Calculating Cell Velocities=====',
                  total=len(Tracks.index)
        ):

        cell_id = Tracks.iloc[i]['cell_id']
        frame = Tracks.iloc[i]['frame']
        feature_id = Tracks.iloc[i]['feature_id']

        velocity_info['frame'].append(frame)
        velocity_info['feature_id'].append(feature_id)
        velocity_info['cell_id'].append(cell_id)

        if Tracks.iloc[i]['frame'] ==  np.min(Tracks['frame']) and cell_id != -1:
            velocity_info['velocity'].append(tuple(np.zeros(dim)))
            velocity_info['speed'].append(0)

        elif cell_id not in Tracks[:i]['cell_id'].values: #check if this is a new cell
            #if new, look for closest cell in same frame to copy velocity from
            same_frame_index_list = Tracks.index[Tracks['frame'] == Tracks.iloc[i]['frame']].tolist() 
            same_frame_index_list.remove(i)
            if len(same_frame_index_list) == 0:
                velocity_info['velocity'].append(tuple(np.zeros(dim)))
                velocity_info['speed'].append(0)
                continue

            min_distance = np.inf
            for j in same_frame_index_list:
                target_row = Tracks.iloc[j]
                if len(velocity_info['velocity']) < j:
                    continue
                else:
                    target_velocity = velocity_info['velocity'][j]
                    target_speed = velocity_info['speed'][j]

                distance = calculate_row_velocity(Tracks.iloc[i], target_row, dt_array, dim)[1]
                if distance < min_distance:
                    min_distance = distance
                    closest_vel = target_velocity
                    closest_speed = target_speed

            # If there are no other cells in the frame, append 0
            if min_distance == np.inf:
                velocity_info['velocity'].append(tuple(np.zeros(dim)))
                velocity_info['speed'].append(0)
            else:
                velocity_info['velocity'].append(closest_vel)
                velocity_info['speed'].append(closest_speed)

        else:
            #if the cell is not new, determine velocity from the change in position from previous frame
            index = np.argwhere(Tracks[:i]['cell_id'].values == cell_id)[-1][0] #index of previous row with same cell id
            velocity_speed = calculate_row_velocity(Tracks.iloc[i], Tracks.iloc[index], dt_array, dim)[2:]
            velocity_info['velocity'].append(velocity_speed[0])
            velocity_info['speed'].append(velocity_speed[1])

    for i in Tracks.index:
        cell_id = Tracks.iloc[i]['cell_id']
        if velocity_info['velocity'][i] == tuple(np.zeros(dim)): #go back and look at where velocity was set to zero and see if we can extrapolate backwards
            index_list = list(np.argwhere(Tracks['cell_id'].values == cell_id))
            index_list.remove(i)
            if len(index_list) == 0:
                continue
            velocity_info['velocity'][i] = velocity_info['velocity'][index_list[0][0]]
            velocity_info['speed'][i] = velocity_info['speed'][index_list[0][0]]

    velocity_info_df = pd.DataFrame(velocity_info)
    # Enforce the threshold
    if threshold is not None:
        # Load background varaible
        if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
            variable_field = analysis_object["segmentation_xarray"]
        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

        if len(variable_field.shape) != 4 and len(variable_field.shape) != 3:
            raise Exception(
                "=====The segmentation must be 2D or 3D====="
            )
        
        feature_mask = analysis_object[f"UDAF_segmentation_{dim}d"].Feature_Segmentation
        feature_mask.values[variable_field.values < threshold] = -1
        uncut_feature_list = np.unique(feature_mask)[1:] # don't use -1
        velocity_info_df = velocity_info_df[velocity_info_df['feature_id'].isin(uncut_feature_list)]

    return velocity_info_df

def calculate_cell_growth(
        analysis_object: dict,
        variable: str | None = None,
        threshold: float | None = None,
        **args: dict,
    ) -> pd.DataFrame:
    """


    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks.
    variable : str, optional
        Variable to which we should apply the threshold. The default is None.
    threshold : float, optional
        Value of which the area should be greater than. The default is None.
    **args : dict
        Throw away variables.

    Raises
    ------
    Exception
        Exception if missing tracks input from the analysis object.
        Exception if the segmentation data is not 3D.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, cell_growth where cell_growth is in m^3 / s.

    """

    if analysis_object['UDAF_tracks'] is None:
        raise Exception(
            "!=====Tracks Data is Required for Velocity Calculation=====!"
        )

    # Check if volume information is given in the arguments, if not then recalculate it
    if 'volume' in args:
        vol = args['volume']['volume']
    else:
        vol = calculate_volume(analysis_object, **args)['volume']

    Tracks = analysis_object['UDAF_tracks']
    Tracks_with_vol = deepcopy(Tracks).join(vol)

    # Cell Growth Rate
    cell_growth_info = {'frame' : [], 'feature_id' : [], 'cell_id' : [], 'cell_growth' : []}

    # List all of the unique cells and make sure there are no non-physical cells
    cell_groups = Tracks_with_vol.groupby("cell_id")

    for cell_id, cell_g in tqdm(cell_groups,
                            desc="=====Calculating Cell Growth Rates=====",
                            total=len(cell_groups)):
        
        cell_feature_arr = np.array(cell_g['feature_id']).tolist()
        cell_frame_arr = np.array(cell_g['frame']).tolist()
        cell_arr = [cell_id] * len(cell_frame_arr)
        cell_vol_arr = np.array(cell_g['volume'])
        cell_lifetime_arr = np.array(cell_g['lifetime'].astype(int) * 1e-9) # convert from ns to s

        # If the cell only lives for one frame, make sure each other array is also one frame, then append nan
        if len(cell_lifetime_arr) == 1:
            assert len(cell_frame_arr) == 1
            assert len(cell_feature_arr) == 1
            assert len(cell_arr) == 1
            
            cell_growth_info['frame'].append(cell_frame_arr[0])
            cell_growth_info['feature_id'].append(cell_feature_arr[0])
            cell_growth_info['cell_id'].append(cell_arr[0])
            cell_growth_info['cell_growth'].append(np.nan)
            continue

        # If the cell lasts for more than one frame, calculate delta eth and delta t
        deth = cell_vol_arr[1:] - cell_vol_arr[:-1]
        dt = cell_lifetime_arr[1:] - cell_lifetime_arr[:-1]

        # The eth/t slope is the growth rate
        slope = deth / dt
        slope = slope.tolist()
        slope.append(slope[-1])

        # Iterate through each element and append them to the info array
        cell_growth_info['frame'].extend(cell_frame_arr)
        cell_growth_info['feature_id'].extend(cell_feature_arr)
        cell_growth_info['cell_id'].extend(cell_arr)
        cell_growth_info['cell_growth'].extend(slope)

    cell_growth_df = pd.DataFrame(cell_growth_info)
    uncut_feature_list = np.unique(cell_growth_df['feature_id'].values)
    # Enforce the threshold
    if threshold is not None:
        # Load background varaible
        if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
            variable_field = analysis_object["segmentation_xarray"]
        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

        if len(variable_field.shape) != 4:
            raise Exception(
                "=====The segmentation must be 3D====="
            )
        
        feature_mask = analysis_object[f"UDAF_segmentation_3d"].Feature_Segmentation
        feature_mask.values[variable_field.values < threshold] = -1
        uncut_feature_list = np.unique(feature_mask)[1:] # don't use -1
        cell_growth_df = cell_growth_df[cell_growth_df['feature_id'].isin(uncut_feature_list)]

    cell_growth_df.sort_values('feature_id', inplace=True)
    cell_growth_df.set_index(np.arange(len(uncut_feature_list)), inplace=True)

    return cell_growth_df

def calculate_perimeter(
        analysis_object: dict,
        variable: str | None = None,
        threshold: float | None = None,
        **args: dict,
    ) -> pd.DataFrame:
    """


    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks.
    variable : str, optional
        Variable to which we should apply the threshold. The default is None.
    threshold : float, optional
        Value of which the area should be greater than. The default is None.
    **args : dict
        Throw away variables.

    Raises
    ------
    Exception
        Exception if missing tracks input from the analysis object.
        Exception if the segmentation data is not 2D or 3D.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, perimeter where perimeter is in km^(segmentation_dim - 1).

    """

    if analysis_object['UDAF_tracks'] is None:
        raise Exception(
            "!=====Tracks Data is Required for Velocity Calculation=====!"
        )

    # If 3D segmentation is available, use that at given height, otherwise use 2D segmentation
    if analysis_object["UDAF_segmentation_3d"] is not None:

        feature_seg_3d = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation
        dim = 3

    elif analysis_object["UDAF_segmentation_2d"] is not None:

        feature_seg_2d = analysis_object["UDAF_segmentation_2d"].Feature_Segmentation
        dim = 2

    else:
        raise Exception("!=====Missing Segmentation Input=====!")


    perimeter_info = {"frame" : [], "feature_id" : [], "cell_id" : [], "perimeter" : []}
    Tracks = analysis_object["UDAF_tracks"]

    # define some useful variables
    proj_x = analysis_object[f'UDAF_segmentation_{dim}d'].projection_x_coordinate
    proj_y = analysis_object[f'UDAF_segmentation_{dim}d'].projection_y_coordinate
    if dim == 3:
        altitude = analysis_object['UDAF_segmentation_3d'].altitude

    for ii in tqdm(range(len(Tracks)),
                            desc='=====Calculating Cell Perimeters=====',
                            total=len(Tracks),
                            ):
        feature_id = Tracks['feature_id'][ii]
        frame = Tracks['frame'][ii]
        cell_id = Tracks['cell_id'][ii]

        x_dim_sizes = np.diff(proj_x) / 1000 # m->km
        y_dim_sizes = np.diff(proj_y) / 1000 # m->km

        # These are one cell too small due to how diff works, so infer last cell size using the same cell size as the previous cell
        x_dim_sizes = np.append(x_dim_sizes, x_dim_sizes[-1])
        y_dim_sizes = np.append(y_dim_sizes, y_dim_sizes[-1])

        if dim == 3:
            z_dim_sizes = np.diff(altitude) / 1000 # m->km
            z_dim_sizes = np.append(z_dim_sizes, z_dim_sizes[-1])

        perims = 0 #this is to keep track of the amount of the perimeter that is shared with another cell
        #key is feature id of other cell, value is the shared perimeter

        if dim == 3:
            feature_seg_in_frame = feature_seg_3d[frame].values
            
            for nz, ny, nx in np.argwhere(feature_seg_in_frame == feature_id):

                for mx in (nx - 1, nx + 1):

                    if mx in range(feature_seg_in_frame.shape[2]) and feature_seg_in_frame[nz, ny, mx] != feature_id:
                        perims += y_dim_sizes[ny]*z_dim_sizes[nz]

                for my in (ny - 1, ny + 1):

                    if my in range(feature_seg_in_frame.shape[1]) and feature_seg_in_frame[nz, my, nx] != feature_id:
                        perims += x_dim_sizes[nx]*z_dim_sizes[nz]

                for mz in (nz - 1, nz + 1):

                    if mz in range(feature_seg_in_frame.shape[0]) and feature_seg_in_frame[mz, ny, nx] != feature_id:
                        perims += x_dim_sizes[nx]*y_dim_sizes[ny]

            perimeter_info["frame"].append(frame)
            perimeter_info['feature_id'].append(feature_id)
            perimeter_info['cell_id'].append(cell_id)
            perimeter_info['perimeter'].append(perims)
        
        if dim == 2:
            feature_seg_in_frame = feature_seg_2d[frame].values
            
            for ny, nx in np.argwhere(feature_seg_in_frame == feature_id):

                for mx in (nx - 1, nx + 1):

                    if mx in range(feature_seg_in_frame.shape[1]) and feature_seg_in_frame[ny, mx] != feature_id:
                        perims += y_dim_sizes[ny]

                for my in (ny - 1, ny + 1):

                    if my in range(feature_seg_in_frame.shape[0]) and feature_seg_in_frame[my, nx] != feature_id:
                        perims += x_dim_sizes[nx]

            perimeter_info["frame"].append(frame)
            perimeter_info['feature_id'].append(feature_id)
            perimeter_info['cell_id'].append(cell_id)
            perimeter_info['perimeter'].append(perims)

    perimeter_df = pd.DataFrame(perimeter_info)
    # Enforce the threshold
    if threshold is not None:
        # Load background varaible
        if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
            variable_field = analysis_object["segmentation_xarray"]
        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

        if len(variable_field.shape) != 4 and len(variable_field.shape) != 3:
            raise Exception(
                "=====The segmentation must be 2D or 3D====="
            )
        
        feature_mask = analysis_object[f"UDAF_segmentation_{dim}d"].Feature_Segmentation
        feature_mask.values[variable_field.values < threshold] = -1
        uncut_feature_list = np.unique(feature_mask)[1:] # don't use -1
        perimeter_df = perimeter_df[perimeter_df['feature_id'].isin(uncut_feature_list)]

    return perimeter_df

def calculate_irregularity(
        analysis_object: dict, 
        irregularity_metrics: str | list[str],
        variable: str | None = None,
        threshold: float | None = None,
        **args: dict,
    ) -> pd.DataFrame:
    """


    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_2d or UDAF_segmentation_3d.
    irregularity_metrics: str | list[str]
        A string or list of strings for which irregularity metric to calculate.
        The implemented metrics are:
            "sphericity",
            "convexity",
    variable : str, optional
        Variable to which we should apply the threshold. The default is None.
    threshold : float, optional
        Value of which the area should be greater than. The default is None.
    **args : dict
        Throw away variables.

    Raises
    ------
    KeyError
        KeyError if there are no found implemented irregularity metrics.
    Exception
        Exception if no segmentation data is found.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, irregularity.

    """
    from .irregularity import (
        convexity,
        sphericity,
    )
    

    if type(irregularity_metrics) != list:
        irregularity_metrics =[irregularity_metrics]
    
    # A list of the currently implemented irregularity metrics
    implemented_metrics = ["sphericity", "convexity"]

    overlap = set(implemented_metrics) & set(irregularity_metrics)
    if len(overlap) == 0:
        raise KeyError("Found No Implemented Irregularity Metric")
        
    list_of_irregularities = []
    if 'convexity' in irregularity_metrics:
        if 'perimeter' in args:
            perimeter_df = args['perimeter']['perimeter']
        else:
            perimeter_df = calculate_perimeter(analysis_object)['perimeter']

        list_of_irregularities.append(convexity.convexity(analysis_object, perimeter_df))

    if 'sphericity' in irregularity_metrics:
        # Get volume and perimeter dataframes
        if 'volume' in args:
            volume_df = args['volume']['volume']
        else:
            volume_df = calculate_volume(analysis_object)['volume']

        if 'perimeter' in args:
            perimeter_df = args['perimeter'] # takes the entire perimeter df, not just perimeter values
        else:
            perimeter_df = calculate_perimeter(analysis_object)

        list_of_irregularities.append(sphericity.sphericity(perimeter_df, volume_df))

    # Enforce the threshold

    if analysis_object["UDAF_segmentation_3d"] is not None:
        uncut_feature_list = np.unique(analysis_object[f"UDAF_segmentation_3d"].Feature_Segmentation)[1:] # don't use -1
    elif analysis_object["UDAF_segmentation_2d"] is not None:
        uncut_feature_list = np.unique(analysis_object["UDAF_segmentation_2d"].Feature_Segmentation[1:])
    else:
        raise Exception("!=====Segmentation Not Found=====!")
    
    if threshold is not None:
        # Load background varaible
        if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
            variable_field = analysis_object["segmentation_xarray"]
        else:
            variable_field = analysis_object["segmentation_xarray"][variable]
        
        feature_mask = analysis_object[f"UDAF_segmentation_3d"].Feature_Segmentation
        feature_mask.values[variable_field.values < threshold] = -1
        uncut_feature_list = np.unique(feature_mask)[1:] # don't use -1

    irregularity_info_df = list_of_irregularities[0]
    # if there are more irregularity metrics, add them as a column to the irregularity info dataframe
    if len(list_of_irregularities) > 1:
        for df in list_of_irregularities[1:]:
            irregularity_info_df = irregularity_info_df.join(df[df.keys()[-1]])
    
    irregularity_info_df = irregularity_info_df[irregularity_info_df['feature_id'].isin(uncut_feature_list)]
        
    return irregularity_info_df