#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:26:17 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running tobac on WRF data processed using wrf_load.py
# =============================================================================

import logging
from copy import deepcopy

import geopandas as gpd
import iris.cube
import numpy as np
import tobac
import xarray as xr


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def wrf_tobac_feature_id(cube: iris.cube.Cube, CONFIG: dict) -> gpd.GeoDataFrame | None:
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
        Exception if out-of-bounds height.

    Returns
    -------
    wrf_geopd : geopandas.geodataframe.GeoDataFrame
        Geodataframe containing all default tobac feature id outputs.

    """

    feat_cube = deepcopy(cube)
    inCONFIG = deepcopy(CONFIG)

    if "height" in inCONFIG["wrf"]["tobac"]["feature_id"]:

        # Ensure segmentation_height is a proper number before running
        if (
            inCONFIG["wrf"]["tobac"]["feature_id"]["height"] == None
            or type(inCONFIG["wrf"]["tobac"]["feature_id"]["height"]) == str
            or type(CONFIG["wrf"]["tobac"]["feature_id"]["height"]) == bool
        ):
            raise Exception(
                f"""!=====Tracking Height Out of Bounds. You Entered: {inCONFIG['wrf']['tobac']['feature_id']['height'] .lower()}=====!"""
            )
        if (
            inCONFIG["wrf"]["tobac"]["feature_id"]["height"]
            > cube.coord("altitude").points.max()
            or inCONFIG["wrf"]["tobac"]["feature_id"]["height"]
            < cube.coord("altitude").points.min()
        ):
            raise Exception(
                f"""!=====Tracking Height Out of Bounds. You Entered: {inCONFIG['wrf']['tobac']['feature_id']['height'] .lower()}=====!"""
            )

        # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
        height_index = find_nearest(
            cube.coord("altitude").points,
            inCONFIG["wrf"]["tobac"]["feature_id"]["height"],
        )

        feat_cube = feat_cube[:, height_index]
        feat_cube.remove_coord("altitude")
        feat_cube.remove_coord("model_level_number")

        del inCONFIG["wrf"]["tobac"]["feature_id"]["height"]

    # Get horozontal spacings
    dxy = tobac.get_spacings(cube)[0]

    # Perform tobac feature identification and then convert to a geodataframe before returning
    wrf_radar_features = tobac.feature_detection.feature_detection_multithreshold(
        feat_cube, dxy=dxy, **inCONFIG["wrf"]["tobac"]["feature_id"]
    )

    if wrf_radar_features is None:
        return None

    wrf_geopd = gpd.GeoDataFrame(
        wrf_radar_features,
        geometry=gpd.points_from_xy(
            wrf_radar_features.longitude, wrf_radar_features.latitude
        ),
        crs="EPSG:4326",
    )

    return wrf_geopd


def wrf_tobac_linking(
    cube: iris.cube.Cube, radar_features: gpd.GeoDataFrame, CONFIG: dict
) -> gpd.GeoDataFrame | None:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    radar_features : gpd.GeoDataFrame
        Tobac radar features from wrf_tobac_feature_id output.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    wrf_geopd_tracks : geopandas.geodataframe.GeoDataFrame
        Geodataframe containing all default tobac tracking outputs.

    """

    if radar_features is None:
        return None

    # Mute tobac logging output
    logging.getLogger("trackpy").setLevel(level=logging.ERROR)

    dxy, dt = tobac.get_spacings(cube)

    # Do tracking then convert output dataframe to a geodataframe
    wrf_tracks = tobac.linking_trackpy(
        radar_features,
        cube,
        dt=dt,
        dxy=dxy,
        vertical_coord="altitude",
        **CONFIG["wrf"]["tobac"]["linking"],
    )

    if wrf_tracks is None:
        return None

    wrf_geopd_tracks = gpd.GeoDataFrame(
        wrf_tracks,
        geometry=gpd.points_from_xy(wrf_tracks.longitude, wrf_tracks.latitude),
        crs="EPSG:4326",
    )

    return wrf_geopd_tracks


def wrf_tobac_segmentation(
    cube: iris.cube.Cube,
    radar_features: gpd.GeoDataFrame,
    segmentation_type: str,
    CONFIG: dict,
    segmentation_height: float | None = None,
) -> tuple[xr.DataArray, gpd.GeoDataFrame]:
    """


    Parameters
    ----------
    cube : iris.cube.Cube
        Iris cube containing the variable to be tracked.
    radar_features : gpd.GeoDataFrame
        tobac radar features from wrf_tobac_feature_id output.
    segmentation_type : str
        ['2D', '3D'], whether to perform 2d segmentation or 3d segmentation.
    CONFIG : dict
        User configuration file.
    segmentation_height : float | None, optional
        height, in meters, to perform the updraft or reflectivity segmentation if 2d selected and tracking_var is not 2D. The default is None.

    Raises
    ------
    Exception
        Exception if out-of-bounds height.

    Returns
    -------
    segment_array : xarray.core.dataarray.DataArray
        Xarray DataArray containing segmented data.
    segment_features : geopandas.geodataframe.GeoDataFrame
        Geodataframe with ncells column.

    """

    if radar_features is None:
        return (None, None)

    # Enforce 2D tracking only for 2D variables
    if (len(cube.shape) == 3) and not segmentation_type.lower() == "2d":
        raise Exception(
            f"!=====Invalid Segmentation Type. You Entered: {segmentation_type.lower()}. TB and PR Tracking Restricted to 2D Segmentation=====!"
        )

    dxy = tobac.get_spacings(cube)[0]
    inCONFIG = deepcopy(CONFIG)

    # 2D and 3D segmentation have different requirements so they are split up here
    if segmentation_type.lower() == "2d":

        if "height" in inCONFIG["wrf"]["tobac"]["segmentation_2d"]:
            del inCONFIG["wrf"]["tobac"]["segmentation_2d"]["height"]

        # If altitude and/or model level number is present, remove it

        # If tracking var is 2d, bypass height
        if len(cube.shape) == 3:
            # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
            segment_cube, segment_features = tobac.segmentation_2D(
                radar_features,
                cube,
                dxy=dxy,
                **inCONFIG["wrf"]["tobac"]["segmentation_2d"],
            )

            # Convert iris cube to xarray and return
            # Add projection x and y back to xarray DataArray
            outXarray = xr.DataArray.from_iris(segment_cube).assign_coords(
                projection_x_coordinate=(
                    "west_east",
                    segment_cube.coord("projection_x_coordinate").points,
                ),
                projection_y_coordinate=(
                    "south_north",
                    segment_cube.coord("projection_y_coordinate").points,
                ),
            )

            return (outXarray, segment_features)

        # Ensure segmentation_height is a proper number before running
        if (
            segmentation_height == None
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
            **inCONFIG["wrf"]["tobac"]["segmentation_2d"],
        )

        # Convert iris cube to xarray and return
        # Add projection x and y back to xarray DataArray
        outXarray = xr.DataArray.from_iris(segment_cube).assign_coords(
            projection_x_coordinate=(
                "west_east",
                segment_cube.coord("projection_x_coordinate").points,
            ),
            projection_y_coordinate=(
                "south_north",
                segment_cube.coord("projection_y_coordinate").points,
            ),
        )

        return (outXarray, segment_features)

    if segmentation_type.lower() == "3d":

        # Similarly, perform 3d segmentation then return products
        segment_cube, segment_features = tobac.segmentation_3D(
            radar_features, cube, dxy=dxy, **inCONFIG["wrf"]["tobac"]["segmentation_3d"]
        )

        # Convert iris cube to xarray and return
        # Add projection x and y back to xarray DataArray
        outXarray = xr.DataArray.from_iris(segment_cube).assign_coords(
            projection_x_coordinate=(
                "west_east",
                segment_cube.coord("projection_x_coordinate").points,
            ),
            projection_y_coordinate=(
                "south_north",
                segment_cube.coord("projection_y_coordinate").points,
            ),
        )

        return (outXarray, segment_features)

    raise Exception(
        f"!=====Invalid Segmentation Type. You Entered: {segmentation_type.lower()}=====!"
    )
