#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:05:16 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running tobac on GOES data processed using goes_load.py
# =============================================================================

import logging

import geopandas as gpd
import iris
import iris.cube
import numpy as np
import tobac
import xarray as xr


def goes_tobac_feature_id(
    cube: iris.cube.Cube, CONFIG: dict
) -> gpd.GeoDataFrame | None:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    goes_geopd : geopandas.geodataframe.GeoDataFrame
        Geodataframe containing all default tobac feature id outputs.

    """

    # Get horozontal spacings in km then convert to m
    res = float(cube.attributes["spatial_resolution"].split("km")[0]) * 1000

    dxy = tobac.get_spacings(cube, grid_spacing=res)[0]

    # Perform tobac feature identification and then convert to a geodataframe before returning
    goes_radar_features = tobac.feature_detection.feature_detection_multithreshold(
        cube, dxy=dxy, **CONFIG["goes"]["tobac"]["feature_id"]
    )

    if goes_radar_features is None:
        return None

    goes_geopd = gpd.GeoDataFrame(
        goes_radar_features,
        geometry=gpd.points_from_xy(
            goes_radar_features.longitude, goes_radar_features.latitude
        ),
        crs="EPSG:4326",
    )

    return goes_geopd


def goes_tobac_linking(
    cube: iris.cube.Cube, radar_features: gpd.GeoDataFrame, CONFIG: dict
) -> gpd.GeoDataFrame | None:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    radar_features : geopandas.geodataframe.GeoDataFrame
        tobac radar features from goes_tobac_feature_id output.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    goes_geopd_tracks : geopandas.geodataframe.GeoDataFrame
        Geodataframe containing all default tobac feature id outputs.

    """

    if radar_features is None:
        return None

    # Mute tobac logging output
    logging.getLogger("trackpy").setLevel(level=logging.ERROR)

    # Get horozontal spacings in km then convert to m
    res = float(cube.attributes["spatial_resolution"].split("km")[0]) * 1000

    dxy = tobac.get_spacings(cube, grid_spacing=res)[0]

    # Get time spacing
    diffs = []
    for ii in range(cube.coord("time").points.shape[0] - 1):
        diffs.append(cube.coord("time").points[ii + 1] - cube.coord("time").points[ii])
    dt = np.nanmean(diffs) * 60

    # Do tracking then convert output dataframe to a geodataframe
    goes_tracks = tobac.linking_trackpy(
        radar_features, cube, dt=dt, dxy=dxy, **CONFIG["goes"]["tobac"]["linking"]
    )

    if goes_tracks is None:
        return None

    goes_geopd_tracks = gpd.GeoDataFrame(
        goes_tracks,
        geometry=gpd.points_from_xy(goes_tracks.longitude, goes_tracks.latitude),
        crs="EPSG:4326",
    )

    return goes_geopd_tracks


def goes_tobac_segmentation(
    cube: iris.cube.Cube, radar_features: gpd.GeoDataFrame, CONFIG: dict
) -> tuple[xr.DataArray, gpd.GeoDataFrame] | tuple[None, None]:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    radar_features : geopandas.geodataframe.GeoDataFrame
        tobac radar features from goes_tobac_feature_id output.
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if using invalid tracking variable.

    Returns
    -------
    segment_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing default tobac segmented data.
    segment_pd : geopandas.geodataframe.GeoDataFrame
        Geodataframe with ncells row.

    """

    if radar_features is None:
        return (None, None)

    # Check tracking var
    if cube.name().lower() != "toa_brightness_temperature":
        raise Exception(
            f"!=====Invalid Tracking Variable. Your Cube Has: {cube.name().lower()}=====!"
        )

    # Get horozontal spacings in km then convert to m
    res = float(cube.attributes["spatial_resolution"].split("km")[0]) * 1000

    dxy = tobac.get_spacings(cube, grid_spacing=res)[0]

    # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
    segment_cube, segment_features = tobac.segmentation_2D(
        radar_features, cube, dxy=dxy, **CONFIG["goes"]["tobac"]["segmentation_2d"]
    )

    # Convert iris cube to xarray and return
    return (xr.DataArray.from_iris(segment_cube), segment_features)
