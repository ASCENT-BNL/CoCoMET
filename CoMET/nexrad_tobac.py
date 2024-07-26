#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:36:27 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running tobac on NEXRAD data processed using nexrad_load.py
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
    CONFIG: User configuration file
Outputs:
    nexrad_geopd: geodataframe containing all default tobac feature id outputs
"""


def nexrad_tobac_feature_id(cube, CONFIG):
    import tobac
    import geopandas as gpd
    from copy import deepcopy

    feat_cube = deepcopy(cube)
    inCONFIG = deepcopy(CONFIG)

    if "height" in inCONFIG["nexrad"]["tobac"]["feature_id"]:

        # Ensure segmentation_height is a proper number before running
        if (
            inCONFIG["nexrad"]["tobac"]["feature_id"]["height"] == None
            or type(inCONFIG["nexrad"]["tobac"]["feature_id"]["height"]) == str
            or type(CONFIG["nexrad"]["tobac"]["feature_id"]["height"]) == bool
        ):
            raise Exception(
                f"""!=====Segmentation Height Out of Bounds. You Entered: {inCONFIG["nexrad"]["tobac"]["feature_id"]["height"] .lower()}=====!"""
            )
            return
        if (
            inCONFIG["nexrad"]["tobac"]["feature_id"]["height"]
            > cube.coord("altitude").points.max()
            or inCONFIG["nexrad"]["tobac"]["feature_id"]["height"]
            < cube.coord("altitude").points.min()
        ):
            raise Exception(
                f"""!=====Segmentation Height Out of Bounds. You Entered: {inCONFIG["nexrad"]["tobac"]["feature_id"]["height"] .lower()}=====!"""
            )
            return

        # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
        height_index = find_nearest(
            cube.coord("altitude").points,
            inCONFIG["nexrad"]["tobac"]["feature_id"]["height"],
        )

        feat_cube = feat_cube[:, height_index]
        feat_cube.remove_coord("altitude")
        feat_cube.remove_coord("model_level_number")

        del inCONFIG["nexrad"]["tobac"]["feature_id"]["height"]

    # Get horozontal spacings
    dxy = tobac.get_spacings(cube)[0]

    if cube.coord("altitude").shape[0] == 1:

        # Perform tobac feature identification and then convert to a geodataframe before returning
        nexrad_radar_features = (
            tobac.feature_detection.feature_detection_multithreshold(
                feat_cube[:, 0], dxy=dxy, **inCONFIG["nexrad"]["tobac"]["feature_id"]
            )
        )

    else:

        # Perform tobac feature identification and then convert to a geodataframe before returning
        nexrad_radar_features = (
            tobac.feature_detection.feature_detection_multithreshold(
                feat_cube, dxy=dxy, **inCONFIG["nexrad"]["tobac"]["feature_id"]
            )
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


"""
Inputs:
    cube: iris cube containing the variable to be tracked
    radar_features: tobac radar features from nexrad_tobac_feature_id output
    CONFIG: User configuration file
Outputs:
    nexrad_geopd_tracks: geodataframe containing all default tobac feature id outputs
"""


def nexrad_tobac_linking(cube, radar_features, CONFIG):
    import tobac
    import logging
    import numpy as np
    import geopandas as gpd

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

    if cube.coord("altitude").shape[0] == 1:

        # Do tracking then convert output dataframe to a geodataframe
        nexrad_tracks = tobac.linking_trackpy(
            radar_features,
            cube[:, 0],
            dt=dt,
            dxy=dxy,
            vertical_coord="altitude",
            **CONFIG["nexrad"]["tobac"]["linking"],
        )

    else:

        # Do tracking then convert output dataframe to a geodataframe
        nexrad_tracks = tobac.linking_trackpy(
            radar_features,
            cube,
            dt=dt,
            dxy=dxy,
            vertical_coord="altitude",
            **CONFIG["nexrad"]["tobac"]["linking"],
        )

    if nexrad_tracks is None:
        return None

    nexrad_geopd_tracks = gpd.GeoDataFrame(
        nexrad_tracks,
        geometry=gpd.points_from_xy(nexrad_tracks.longitude, nexrad_tracks.latitude),
        crs="EPSG:4326",
    )

    return nexrad_geopd_tracks


"""
Inputs:
    cube: iris cube containing the variable to be tracked
    radar_features: tobac radar features from nexrad_tobac_feature_id output
    segmentation_type: ["2D", "3D"], whether to perform 2d segmentation or 3d segmentation
    CONFIG: User configuration file
    segmentation_height: height, in meters, to perform the updraft or reflectivity segmentation if 2d selected and tracking_var != tb
Outputs:
    (segment_array, segment_features): xarray DataArray containing segmented data and geodataframe with ncells row
"""


def nexrad_tobac_segmentation(
    cube, radar_features, segmentation_type, CONFIG, segmentation_height=None
):
    import tobac
    import xarray as xr
    from copy import deepcopy

    if radar_features is None:
        return (None, None)

    # Check tracking var
    if cube.name().lower() != "equivalent_reflectivity_factor":
        raise Exception(
            f"!=====Invalid Tracking Variable. Your Cube Has: {cube.name().lower()}=====!"
        )
        return

    inCONFIG = deepcopy(CONFIG)

    dxy = tobac.get_spacings(cube)[0]

    # 2D and 3D segmentation have different requirements so they are split up here
    if segmentation_type.lower() == "2d":

        if "height" in inCONFIG["nexrad"]["tobac"]["segmentation_2d"]:
            del inCONFIG["nexrad"]["tobac"]["segmentation_2d"]["height"]

        # Ensure segmentation_height is a proper number before running
        if type(segmentation_height) == str or type(segmentation_height) == bool:
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height}=====!"
            )
            return

        if segmentation_height is not None and cube.coord("altitude").shape[0] > 1:

            if (
                segmentation_height > cube.coord("altitude").points.max()
                or segmentation_height < cube.coord("altitude").points.min()
            ):
                raise Exception(
                    f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height}=====!"
                )
                return

        elif segmentation_height is None and cube.coord("altitude").shape[0] == 1:

            segmentation_height = cube.coord("altitude").points[0]

        elif segmentation_height is None and cube.coord("altitude").shape[0] > 1:
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height}=====!"
            )
            return

        # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
        height_index = find_nearest(cube.coord("altitude").points, segmentation_height)

        # Remove 1 dimensional coordinates caused by taking only one altitude
        seg_cube = deepcopy(cube[:, height_index])
        seg_cube.remove_coord("altitude")
        seg_cube.remove_coord("model_level_number")

        # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
        segment_cube, segment_features = tobac.segmentation_2D(
            radar_features,
            seg_cube,
            dxy=dxy,
            **inCONFIG["nexrad"]["tobac"]["segmentation_2d"],
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

        if cube.coord("altitude").shape[0] == 1:
            raise Exception(
                "!=====Invalid Segmentation Type. Only One Altitude Present=====!"
            )
            return

        # Similarly, perform 3d segmentation then return products
        segment_cube, segment_features = tobac.segmentation_3D(
            radar_features,
            cube,
            dxy=dxy,
            **inCONFIG["nexrad"]["tobac"]["segmentation_3d"],
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
        return
