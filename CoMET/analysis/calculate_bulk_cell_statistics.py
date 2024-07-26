#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:00:59 2024

@author: thahn
"""

# =============================================================================
# This contains all of the functions to calculate basic bulk cell statistics (ETH, area, etc.) from CoMET-UDAF data
# =============================================================================


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def calculate_ETH(
    analysis_object,
    threshold,
    variable=None,
    cell_footprint_height=2000,
    quantile=0.95,
    **args,
):
    """
    Inputs:
        analysis_object: A CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_2d or UDAF_segmentation_3d, and segmentation_xarray
        threshold: The value which needs to be exceeded to count towards the echo top height. I.e. 15 for reflectivity.
        variable: The variable from the input segmentation_xarray which should be used for calculating ETH
        cell_footprint_height: The height used to calculate the cell area to determine where to calculate ETHs
        quantile: The percentile of calculated ETHs to return
    Outputs:
        eth_info: A pandas dataframe with the following rows: frame, feature_id, cell_id, eth where eth is in km
    """

    import numpy as np
    import xarray as xr
    import pandas as pd
    from tqdm import tqdm

    # If input variable field is 2D return None. Also, if DataArray, use those values for calculations. If Dataset, use tracking_var to get variable
    if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:

        if len(analysis_object["segmentation_xarray"].shape) != 4:
            return None

        else:
            variable_field = analysis_object["segmentation_xarray"]

    else:

        if len(analysis_object["segmentation_xarray"][variable].shape) != 4:
            return None

        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

    # If 3D segmentation is available, use that to calculate cell footprint, otherwise use 2D segmentation
    if analysis_object["UDAF_segmentation_3d"] is not None:

        height_index = find_nearest(
            analysis_object["UDAF_segmentation_3d"].altitude.values,
            cell_footprint_height,
        )

        footprint_data = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation[
            :, height_index
        ]

    elif analysis_object["UDAF_segmentation_2d"] is not None:

        footprint_data = analysis_object["UDAF_segmentation_2d"].Feature_Segmentation

    else:
        raise Exception("!=====Missing Segmentation Input=====!")

    eth_info = {"frame": [], "feature_id": [], "cell_id": [], "eth": []}  # in km

    frame_groups = analysis_object["UDAF_linking"].groupby("frame")

    # Loop over frames
    for ii, frame in tqdm(
        enumerate(frame_groups),
        desc="=====Calculating Echo Top Heights=====",
        total=frame_groups.ngroups,
    ):

        # Loop over each feature
        for feature in frame[1].groupby("feature_id"):

            # Get the indices of the cell footprint
            proper_indices = np.argwhere(footprint_data[frame[0]].values == feature[0])

            # Cells which have no segmented output should get a NaN
            if len(proper_indices) == 0:

                eth_info["frame"].append(frame[0])
                eth_info["feature_id"].append(feature[0])
                eth_info["cell_id"].append(feature[1]["cell_id"].min())
                eth_info["eth"].append(np.nan)
                continue

            eth_set = []

            # Calculate ETH for each location
            for iy, ix in proper_indices:

                max_alt_index = np.argwhere(
                    variable_field[frame[0], :, iy, ix].values > threshold
                )

                # If there are no indices greater than threshold, skip
                if len(max_alt_index) != 0:
                    max_alt_index = max_alt_index.max()
                else:
                    eth_set.append(np.nan)
                    continue

                max_alt = variable_field.altitude.values[max_alt_index]
                eth_set.append(max_alt)

            eth_info["frame"].append(frame[0])
            eth_info["feature_id"].append(feature[0])
            eth_info["cell_id"].append(feature[1]["cell_id"].min())

            # If all NaN slice, append just np.nan
            if np.isnan(eth_set).all():
                eth_info["eth"].append(np.nan)
            else:
                eth_info["eth"].append(np.nanquantile(eth_set, quantile) / 1000)

    return pd.DataFrame(eth_info)


def calculate_area(analysis_object, height=2000, variable=None, threshold=None, **args):
    """
    Inputs:
        analysis_object: A CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_2d or UDAF_segmentation_3d
        height: The height which is used to calculate the area of cells
        threshold: Value of which the area should be greater than
    Outputs:
        area_info: A pandas dataframe with the following rows: frame, feature_id, cell_id, area where area is in km^2
    """

    import numpy as np
    import pandas as pd
    import xarray as xr
    from tqdm import tqdm

    # Load background varaible
    if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
        variable_field = analysis_object["segmentation_xarray"]
    else:
        variable_field = analysis_object["segmentation_xarray"][variable]

    if len(variable_field.shape) == 4:
        height_index = find_nearest(
            analysis_object["segmentation_xarray"]["altitude"].values, height
        )

        variable_field = variable_field[:, height_index]

    # If 3D segmentation is available, use that at given height, otherwise use 2D segmentation
    if analysis_object["UDAF_segmentation_3d"] is not None:

        height_index = find_nearest(
            analysis_object["UDAF_segmentation_3d"].altitude.values, height
        )

        mask = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation[
            :, height_index
        ]

    elif analysis_object["UDAF_segmentation_2d"] is not None:

        mask = analysis_object["UDAF_segmentation_2d"].Feature_Segmentation

    else:
        raise Exception("!=====Missing Segmentation Input=====!")

    # Enforce threshold
    if threshold is not None:
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

    frame_groups = analysis_object["UDAF_linking"].groupby("frame")

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


def calculate_volume(analysis_object, variable=None, threshold=None, **args):
    """
    Inputs:
        analysis_object: A CoMET-UDAF standard analysis object containing at least UDAF_tracks and UDAF_segmentation_3d
    Outputs:
        volume_info: A pandas dataframe with the following rows: frame, feature_id, cell_id, volume where area is in km^3
    """

    import numpy as np
    import pandas as pd
    import xarray as xr
    from tqdm import tqdm

    # Load background varaible
    if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:
        variable_field = analysis_object["segmentation_xarray"]
    else:
        variable_field = analysis_object["segmentation_xarray"][variable]

    if len(variable_field.shape) != 4:
        raise Exception("=====Must have a 3D segmentation _array=====")

    if analysis_object["UDAF_segmentation_3d"] is None:
        raise Exception(
            "!=====3D Segmentation Data is Required for Volume Calculation=====!"
        )

    mask = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation

    # Enforce threshold
    if threshold is not None:
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

    frame_groups = analysis_object["UDAF_linking"].groupby("frame")

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
