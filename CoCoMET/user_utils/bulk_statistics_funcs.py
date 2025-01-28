#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:37:22 2024

@author: thahn
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

__all__ = ["area_func", "volume_func", "var_max_height_func"]


# TODO: Update this to include all functions in the future
# Calculate nearest item in list to given pivot
def find_nearest(array: np.ndarray, pivot) -> int:

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def var_max_height_func(
    variable_field: xr.DataArray,
    tracks: gpd.GeoDataFrame,
    segmentation_2d: xr.DataArray,
    threshold: float,
    quantile: float = 0.95,
) -> pd.DataFrame | None:
    """


    Parameters
    ----------
    variable_field : xarray.core.dataarray.DataArray
        An xarray DataArray that has the values you are interested in with an altitude coordinate.
    tracks : geopandas.geodataframe.GeoDataFrame
        US_linking formated tracks from CoCoMET-like output.
    segmentation_2d : xarray.core.dataarray.DataArray
        2D Segmentation output with feature-wise labelling.
    threshold : float
        The value which needs to be exceeded to count towards the var top height. I.e. 15 for reflectivity.
    quantile : float, optional
        The percentile of calculated max heights to return. The default is 0.95.

    Raises
    ------
    Exception
        Exception if missing segmentation data from the analysis object.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, eth where eth is in km.

    """

    footprint_data = segmentation_2d

    eth_info = {"frame": [], "feature_id": [], "cell_id": [], "eth": []}  # in km

    frame_groups = tracks.groupby("frame")

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


def area_func(segmentation_2d: xr.DataArray, tracks: gpd.GeoDataFrame) -> pd.DataFrame:
    """


    Parameters
    ----------
    segmentation_2d : xarray.core.dataarray.DataArray
        2D Segmentation output with feature-wise labelling.
    tracks : geopandas.geodataframe.GeoDataFrame
        US_linking formated tracks from CoCoMET-like output.

    Raises
    ------
    Exception
        Exception if missing segmentation input from the analysis object.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, area where area is in km^2.

    """

    mask = segmentation_2d

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

    frame_groups = tracks.groupby("frame")

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


def volume_func(
    segmentation_3d: xr.DataArray, tracks: gpd.GeoDataFrame
) -> pd.DataFrame:
    """


    Parameters
    ----------
    segmentation_3d : xarray.core.dataarray.DataArray
        3D Segmentation output with feature-wise labelling.
    tracks : geopandas.geodataframe.GeoDataFrame
        US_linking formated tracks from CoCoMET-like output.

    Raises
    ------
    Exception
        Exception if missing segmentation input from the analysis object.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, volume where area is in km^3.

    """

    mask = segmentation_3d

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

    frame_groups = tracks.groupby("frame")

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
