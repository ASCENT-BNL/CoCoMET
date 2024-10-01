# =============================================================================
# This defines the methods for running tobac on RAMS data processed using rams_load.py
# =============================================================================


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def rams_tobac_feature_id(cube, CONFIG):
    """
    Inputs:
        cube: iris cube containing the variable to be tracked
        CONFIG: User configuration file
    Outputs:
        rams_geopd: geodataframe containing all default tobac feature id outputs
    """

    import tobac
    import geopandas as gpd
    from copy import deepcopy

    feat_cube = deepcopy(cube)
    inCONFIG = deepcopy(CONFIG)

    if "height" in inCONFIG["rams"]["tobac"]["feature_id"]:

        # Ensure segmentation_height is a proper number before running
        if (
            inCONFIG["rams"]["tobac"]["feature_id"]["height"] == None
            or type(inCONFIG["rams"]["tobac"]["feature_id"]["height"]) == str
            or type(CONFIG["rams"]["tobac"]["feature_id"]["height"]) == bool
        ):
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {inCONFIG['rams']['tobac']['feature_id']['height'] .lower()}=====!"
            )
        if (
            inCONFIG["rams"]["tobac"]["feature_id"]["height"]
            > cube.coord("altitude").points.max()
            or inCONFIG["rams"]["tobac"]["feature_id"]["height"]
            < cube.coord("altitude").points.min()
        ):
            raise Exception(
                f"!=====Segmentation Height Out of Bounds. You Entered: {inCONFIG['rams']['tobac']['feature_id']['height'] .lower()}=====!"
            )

        # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
        height_index = find_nearest(
            cube.coord("altitude").points,
            inCONFIG["rams"]["tobac"]["feature_id"]["height"],
        )

        feat_cube = feat_cube[:, height_index]
        feat_cube.remove_coord("altitude")
        feat_cube.remove_coord("model_level_number")

        del inCONFIG["rams"]["tobac"]["feature_id"]["height"]

    # Get horozontal spacings
    dxy = tobac.get_spacings(cube)[0]

    # Perform tobac feature identification and then convert to a geodataframe before returning
    rams_radar_features = tobac.feature_detection.feature_detection_multithreshold(
        feat_cube, dxy=dxy, **inCONFIG["rams"]["tobac"]["feature_id"]
    )

    if rams_radar_features is None:
        return None

    rams_geopd = gpd.GeoDataFrame(
        rams_radar_features,
        geometry=gpd.points_from_xy(
            rams_radar_features.longitude, rams_radar_features.latitude
        ),
        crs="EPSG:4326",
    )

    return rams_geopd


def rams_tobac_linking(cube, radar_features, CONFIG):
    """
    Inputs:
        cube: iris cube containing the variable to be tracked
        radar_features: tobac radar features from rams_tobac_feature_id output
        CONFIG: User configuration file
    Outputs:
        rams_geopd_tracks: geodataframe containing all default tobac tracking outputs
    """

    import tobac
    import logging
    import geopandas as gpd

    if radar_features is None:
        return None

    # Mute tobac logging output
    logging.getLogger("trackpy").setLevel(level=logging.ERROR)

    dxy, dt = tobac.get_spacings(cube)

    # Do tracking then convert output dataframe to a geodataframe
    rams_tracks = tobac.linking_trackpy(
        radar_features,
        cube,
        dt=dt,
        dxy=dxy,
        vertical_coord="altitude",
        **CONFIG["rams"]["tobac"]["linking"],
    )

    if rams_tracks is None:
        return None

    rams_geopd_tracks = gpd.GeoDataFrame(
        rams_tracks,
        geometry=gpd.points_from_xy(rams_tracks.longitude, rams_tracks.latitude),
        crs="EPSG:4326",
    )

    return rams_geopd_tracks


def rams_tobac_segmentation(
    cube, radar_features, segmentation_type, CONFIG, segmentation_height=None
):
    """
    Inputs:
        cube: iris cube containing the variable to be tracked
        radar_features: tobac radar features from rams_tobac_feature_id output
        segmentation_type: ['2D', '3D'], whether to perform 2d segmentation or 3d segmentation
        CONFIG: User configuration file
        segmentation_height: height, in meters, to perform the updraft or reflectivity segmentation if 2d selected and tracking_var != tb or pr
    Outputs:
        (segment_array, segment_features): xarray DataArray containing segmented data and geodataframe with ncells row
    """

    import tobac
    import xarray as xr
    from copy import deepcopy

    if radar_features is None:
        return (None, None)

    # Enforce 2D tracking only for brightness temperature and precip rate tracking
    if (
        cube.name().lower() == "tb" or cube.name().lower() == "pr"
    ) and not segmentation_type.lower() == "2d":
        raise Exception(
            f"!=====Invalid Segmentation Type. You Entered: {segmentation_type.lower()}. TB and PR Tracking Restricted to 2D Segmentation=====!"
        )

    dxy = tobac.get_spacings(cube)[0]
    inCONFIG = deepcopy(CONFIG)

    # 2D and 3D segmentation have different requirements so they are split up here
    if segmentation_type.lower() == "2d":

        if "height" in inCONFIG["rams"]["tobac"]["segmentation_2d"]:
            del inCONFIG["rams"]["tobac"]["segmentation_2d"]["height"]

        # If altitude and/or model level number is present, remove it

        # If tracking var is tb or pr, bypass height
        if cube.name().lower() == "tb" or cube.name().lower() == "pr":
            # Perform the 2d segmentation at the height_index and return the segmented cube and new geodataframe
            segment_cube, segment_features = tobac.segmentation_2D(
                radar_features,
                cube,
                dxy=dxy,
                **inCONFIG["rams"]["tobac"]["segmentation_2d"],
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
                f"!=====Segmentation Height Out of Bounds. You Entered: {segmentation_height.lower()}=====!"
            )
        if (
            segmentation_height > cube.coord("altitude").points.max()
            or segmentation_height < cube.coord("altitude").points.min()
        ):
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
            **inCONFIG["rams"]["tobac"]["segmentation_2d"],
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
            radar_features, cube, dxy=dxy, **inCONFIG["rams"]["tobac"]["segmentation_3d"]
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
