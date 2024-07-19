#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:29:55 2024

@author: thahn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:36:27 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running tobac on standard gridded radar data processed using standard_radar_load.py
# =============================================================================


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def standard_radar_tobac_feature_id(cube, CONFIG):
    """
    Inputs:
        cube: iris cube containing the variable to be tracked
        CONFIG: User configuration file
    Outputs:
        radar_geopd: geodataframe containing all default tobac feature id outputs
    """

    import tobac
    import geopandas as gpd
    from copy import deepcopy

    feat_cube = deepcopy(cube)
    inCONFIG = deepcopy(CONFIG)

    if "height" in inCONFIG["standard_radar"]["tobac"]["feature_id"]:

        # Ensure tracking height is a proper number before running
        if (
            inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"] == None
            or type(inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"]) == str
            or type(CONFIG["standard_radar"]["tobac"]["feature_id"]["height"]) == bool
        ):
            raise Exception(
                f"""!=====Segmentation Height Out of Bounds. You Entered: {inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"] .lower()}=====!"""
            )
        if (
            inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"]
            > cube.coord("altitude").points.max()
            or inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"]
            < cube.coord("altitude").points.min()
        ):
            raise Exception(
                f"""!=====Segmentation Height Out of Bounds. You Entered: {inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"] .lower()}=====!"""
            )

        # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
        height_index = find_nearest(
            cube.coord("altitude").points,
            inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"],
        )

        feat_cube = feat_cube[:, height_index]
        feat_cube.remove_coord("altitude")
        feat_cube.remove_coord("model_level_number")

        del inCONFIG["standard_radar"]["tobac"]["feature_id"]["height"]

    # Get horozontal spacings
    dxy = tobac.get_spacings(cube)[0]

    if cube.coord("altitude").shape[0] == 1:

        # Perform tobac feature identification and then convert to a geodataframe before returning
        radar_features = tobac.feature_detection.feature_detection_multithreshold(
            feat_cube[:, 0],
            dxy=dxy,
            **inCONFIG["standard_radar"]["tobac"]["feature_id"],
        )

    else:

        # Perform tobac feature identification and then convert to a geodataframe before returning
        radar_features = tobac.feature_detection.feature_detection_multithreshold(
            feat_cube, dxy=dxy, **inCONFIG["standard_radar"]["tobac"]["feature_id"]
        )

    if radar_features is None:
        return None

    radar_geopd = gpd.GeoDataFrame(
        radar_features,
        geometry=gpd.points_from_xy(radar_features.longitude, radar_features.latitude),
        crs="EPSG:4326",
    )

    return radar_geopd


def standard_radar_tobac_linking(cube, radar_features, CONFIG):
    """
    Inputs:
        cube: iris cube containing the variable to be tracked
        radar_features: tobac radar features from standard_radar_tobac_feature_id output
        CONFIG: User configuration file
    Outputs:
        radar_geopd_tracks: geodataframe containing all default tobac feature id outputs
    """

    import tobac
    import numpy as np
    import geopandas as gpd

    if radar_features is None:
        return None

    dxy = tobac.get_spacings(cube)[0]

    # Get time spacing
    diffs = []
    for ii in range(cube.coord("time").points.shape[0] - 1):

        diffs.append(cube.coord("time").points[ii + 1] - cube.coord("time").points[ii])

    dt = np.nanmedian(diffs) * 60

    if cube.coord("altitude").shape[0] == 1:

        # Do tracking then convert output dataframe to a geodataframe
        radar_tracks = tobac.linking_trackpy(
            radar_features,
            cube[:, 0],
            dt=dt,
            dxy=dxy,
            **CONFIG["standard_radar"]["tobac"]["linking"],
        )

    else:

        # Do tracking then convert output dataframe to a geodataframe
        radar_tracks = tobac.linking_trackpy(
            radar_features,
            cube,
            dt=dt,
            dxy=dxy,
            vertical_coord="altitude",
            **CONFIG["standard_radar"]["tobac"]["linking"],
        )

    if radar_tracks is None:
        return None

    radar_geopd_tracks = gpd.GeoDataFrame(
        radar_tracks,
        geometry=gpd.points_from_xy(radar_tracks.longitude, radar_tracks.latitude),
        crs="EPSG:4326",
    )

    return radar_geopd_tracks


def standard_radar_tobac_segmentation(
    cube, radar_features, segmentation_type, CONFIG, segmentation_height=None
):
    """
    Inputs:
        cube: iris cube containing the variable to be tracked
        radar_features: tobac radar features from standard_radar_tobac_feature_id output
        segmentation_type: ["2D", "3D"], whether to perform 2d segmentation or 3d segmentation
        CONFIG: User configuration file
        segmentation_height: height, in meters, to perform the updraft or reflectivity segmentation if 2d selected and tracking_var != tb
    Outputs:
        (segment_array, segment_features): xarray DataArray containing segmented data and geodataframe with ncells row
    """

    import tobac
    import xarray as xr
    from copy import deepcopy

    if radar_features is None:
        return None

    # Check tracking var
    if cube.name().lower() != "equivalent_reflectivity_factor":
        raise Exception(
            f"!=====Invalid Tracking Variable. Your Cube Has: {cube.name().lower()}=====!"
        )

    inCONFIG = deepcopy(CONFIG)

    dxy = tobac.get_spacings(cube)[0]

    # 2D and 3D segmentation have different requirements so they are split up here
    if segmentation_type.lower() == "2d":

        if "height" in inCONFIG["standard_radar"]["tobac"]["segmentation_2d"]:
            del inCONFIG["standard_radar"]["tobac"]["segmentation_2d"]["height"]

        # Ensure segmentation_height is a proper number before running
        if type(segmentation_height) == str or type(segmentation_height) == bool:
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height.lower()}=====!"
            )

        if segmentation_height is not None and cube.coord("altitude").shape[0] > 1:

            if (
                segmentation_height > cube.coord("altitude").points.max()
                or segmentation_height < cube.coord("altitude").points.min()
            ):
                raise Exception(
                    f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height.lower()}=====!"
                )

        elif segmentation_height is None and cube.coord("altitude").shape[0] == 1:

            segmentation_height = cube.coord("altitude").points[0]

        elif segmentation_height is None and cube.coord("altitude").shape[0] > 1:
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height.lower()}=====!"
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
            **inCONFIG["standard_radar"]["tobac"]["segmentation_2d"],
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

    if segmentation_type.lower() == "3d":

        if cube.coord("altitude").shape[0] == 1:
            raise Exception(
                "!=====Invalid Segmentation Type. Only One Altitude Present=====!"
            )

        # Similarly, perform 3d segmentation then return products
        segment_cube, segment_features = tobac.segmentation_3D(
            radar_features,
            cube,
            dxy=dxy,
            **inCONFIG["standard_radar"]["tobac"]["segmentation_3d"],
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
