#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:25:47 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running tobac on NEXRAD data processed using nexrad_load.py
# =============================================================================

import logging
from copy import deepcopy

import geopandas as gpd
import iris
import iris.cube
import numpy as np
import tobac
import xarray as xr


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def multi_nexrad_tobac_feature_id(
    cube: iris.cube.Cube, CONFIG: dict
) -> gpd.GeoDataFrame | None:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if height is out of bounds.

    Returns
    -------
    nexrad_geopd : geopandas.geodataframe.GeoDataFrame
        Geodataframe containing all default tobac feature id outputs.

    """

    feat_cube = deepcopy(cube)
    inCONFIG = deepcopy(CONFIG)

    if "height" in inCONFIG["multi_nexrad"]["tobac"]["feature_id"]:

        # Ensure segmentation_height is a proper number before running
        if (
            inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"] == None
            or type(inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"]) == str
            or type(CONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"]) == bool
        ):
            raise Exception(
                f"""!=====Feature Identification Height Out of Bounds. You Entered: {inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"]}=====!"""
            )
        if (
            inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"]
            > cube.coord("altitude").points.max()
            or inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"]
            < cube.coord("altitude").points.min()
        ):
            raise Exception(
                f"""!=====Feature Identification Height Out of Bounds. You Entered: {inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"]}=====!"""
            )

        # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
        height_index = find_nearest(
            cube.coord("altitude").points,
            inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"],
        )

        feat_cube = feat_cube[:, height_index]
        feat_cube.remove_coord("altitude")
        feat_cube.remove_coord("model_level_number")

        del inCONFIG["multi_nexrad"]["tobac"]["feature_id"]["height"]

    # Get horozontal spacings
    dxy = tobac.get_spacings(cube)[0]

    # Perform tobac feature identification and then convert to a geodataframe before returning
    nexrad_radar_features = tobac.feature_detection.feature_detection_multithreshold(
        feat_cube, dxy=dxy, **inCONFIG["multi_nexrad"]["tobac"]["feature_id"]
    )

    if nexrad_radar_features is None:
        return None

    nexrad_geopd = gpd.GeoDataFrame(
        nexrad_radar_features,
        geometry=gpd.points_from_xy(
            nexrad_radar_features.longitude, nexrad_radar_features.latitude
        ),
        crs="EPSG:4326",
    )

    return nexrad_geopd


def multi_nexrad_tobac_linking(
    cube: iris.cube.Cube, radar_features: gpd.GeoDataFrame, CONFIG: dict
) -> gpd.GeoDataFrame | None:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    radar_features : gpd.GeoDataFrame
        tobac radar features from nexrad_tobac_feature_id output.
    CONFIG : dict
        Geodataframe containing all default tobac feature id outputs.

    Returns
    -------
    nexrad_geopd_tracks : geopandas.geodataframe.GeoDataFrame
        Default tobac linking outputs.

    """

    if radar_features is None:
        return None

    # Mute tobac logging output
    logging.getLogger("trackpy").setLevel(level=logging.ERROR)

    dxy = tobac.get_spacings(cube)[0]

    # Get time spacing
    diffs = []
    for ii in range(cube.coord("time").points.shape[0] - 1):

        diffs.append(cube.coord("time").points[ii + 1] - cube.coord("time").points[ii])

    dt = np.nanmedian(diffs) * 60

    # Do tracking then convert output dataframe to a geodataframe
    nexrad_tracks = tobac.linking_trackpy(
        radar_features,
        cube,
        dt=dt,
        dxy=dxy,
        vertical_coord="altitude",
        **CONFIG["multi_nexrad"]["tobac"]["linking"],
    )

    if nexrad_tracks is None:
        return None

    nexrad_geopd_tracks = gpd.GeoDataFrame(
        nexrad_tracks,
        geometry=gpd.points_from_xy(nexrad_tracks.longitude, nexrad_tracks.latitude),
        crs="EPSG:4326",
    )

    return nexrad_geopd_tracks


def multi_nexrad_tobac_segmentation(
    cube: iris.cube.Cube,
    radar_features: gpd.GeoDataFrame,
    segmentation_type: str,
    CONFIG: dict,
    segmentation_height: float = None,
) -> tuple[xr.DataArray, gpd.GeoDataFrame] | tuple[None, None]:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    radar_features : geopandas.geodataframe.GeoDataFrame
        tobac radar features from multi_nexrad_tobac_feature_id output.
    segmentation_type : str
        ["2D", "3D"], whether to perform 2d segmentation or 3d segmentation.
    CONFIG : dict
        User configuration file.
    segmentation_height : float, optional
        Height, in meters, to perform the updraft or reflectivity segmentation if 2d selected. The default is None.

    Raises
    ------
    Exception
        Exception for invalid segmentation type or height.

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
    if cube.name().lower() != "equivalent_reflectivity_factor":
        raise Exception(
            f"!=====Invalid Tracking Variable. Your Cube Has: {cube.name().lower()}=====!"
        )

    inCONFIG = deepcopy(CONFIG)

    dxy = tobac.get_spacings(cube)[0]

    # 2D and 3D segmentation have different requirements so they are split up here
    if segmentation_type.lower() == "2d":

        if "height" in inCONFIG["multi_nexrad"]["tobac"]["segmentation_2d"]:
            del inCONFIG["multi_nexrad"]["tobac"]["segmentation_2d"]["height"]

        # Ensure segmentation_height is a proper number before running
        if (
            segmentation_height is None
            or type(segmentation_height) == str
            or type(segmentation_height) == bool
        ):
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height}=====!"
            )
        if (
            segmentation_height > cube.coord("altitude").points.max()
            or segmentation_height < cube.coord("altitude").points.min()
        ):
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height}=====!"
            )

        # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
        height_index = find_nearest(cube.coord("altitude").points, segmentation_height)

        # Remove 1 dimensional coordinates cause by taking only one altitude
        seg_cube = deepcopy(cube[:, height_index])
        seg_cube.remove_coord("altitude")
        seg_cube.remove_coord("model_level_number")

        # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
        segment_cube, segment_features = tobac.segmentation_2D(
            radar_features,
            seg_cube,
            dxy=dxy,
            **inCONFIG["multi_nexrad"]["tobac"]["segmentation_2d"],
        )

        # Convert iris cube to xarray and return
        # Add projection x and y back to xarray DataArray
        outXarray = xr.DataArray.from_iris(segment_cube).assign_coords(
            projection_x_coordinate=(
                "x",
                segment_cube.coord("projection_x_coordinate").points,
            ),
            projection_y_coordinate=(
                "y",
                segment_cube.coord("projection_y_coordinate").points,
            ),
        )

        return (outXarray, segment_features)

    elif segmentation_type.lower() == "3d":

        # Similarly, perform 3d segmentation then return products
        segment_cube, segment_features = tobac.segmentation_3D(
            radar_features,
            cube,
            dxy=dxy,
            **inCONFIG["multi_nexrad"]["tobac"]["segmentation_3d"],
        )

        # Convert iris cube to xarray and return
        # Add projection x and y back to xarray DataArray
        outXarray = xr.DataArray.from_iris(segment_cube).assign_coords(
            projection_x_coordinate=(
                "x",
                segment_cube.coord("projection_x_coordinate").points,
            ),
            projection_y_coordinate=(
                "y",
                segment_cube.coord("projection_y_coordinate").points,
            ),
        )

        return (outXarray, segment_features)

    else:
        raise Exception(
            f"!=====Invalid Segmentation Type. You Entered: {segmentation_type.lower()}=====!"
        )
