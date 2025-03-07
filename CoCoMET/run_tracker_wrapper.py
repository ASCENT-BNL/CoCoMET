from copy import deepcopy

import geopandas as gpd
import iris
import numpy as np
import xarray as xr

from CoCoMET.analysis.analysis_object import Analysis_Object
from CoCoMET.analysis.calc_var import calc_var
from CoCoMET.post_processor import filter_cells

from .goes_load import goes_load_netcdf_iris
from .goes_tobac import (
    goes_tobac_feature_id,
    goes_tobac_linking,
    goes_tobac_segmentation,
)
from .mesonh_calculate_products import mesonh_calculate_reflectivity
from .mesonh_load import mesonh_load_netcdf_iris
from .mesonh_moaap import mesonh_run_moaap
from .mesonh_tams import mesonh_run_tams
from .mesonh_tobac import (
    mesonh_tobac_feature_id,
    mesonh_tobac_linking,
    mesonh_tobac_segmentation,
)
from .multi_nexrad_load import multi_nexrad_load_netcdf_iris
from .multi_nexrad_tobac import (
    multi_nexrad_tobac_feature_id,
    multi_nexrad_tobac_linking,
    multi_nexrad_tobac_segmentation,
)
from .nexrad_load import nexrad_load_netcdf_iris
from .nexrad_tobac import (
    nexrad_tobac_feature_id,
    nexrad_tobac_linking,
    nexrad_tobac_segmentation,
)
from .rams_calculate_products import rams_calculate_reflectivity
from .rams_load import rams_load_netcdf_iris

# Loading these is not memory or time intensive and is best practice to do so in the file header.
from .rams_moaap import rams_run_moaap
from .rams_tams import rams_run_tams
from .rams_tobac import (
    rams_tobac_feature_id,
    rams_tobac_linking,
    rams_tobac_segmentation,
)
from .standard_radar_load import standard_radar_load_netcdf_iris
from .standard_radar_tobac import (
    standard_radar_tobac_feature_id,
    standard_radar_tobac_linking,
    standard_radar_tobac_segmentation,
)

# Load the US converting functions
from .tracker_output_translation_layer import (
    bulk_moaap_to_US,
    bulk_tams_to_US,
    feature_id_to_US,
    linking_to_US,
    segmentation_to_US,
)
from .wrf_calculate_products import wrf_calculate_reflectivity
from .wrf_load import wrf_load_netcdf_iris
from .wrf_moaap import wrf_run_moaap
from .wrf_tams import wrf_run_tams
from .wrf_tobac import wrf_tobac_feature_id, wrf_tobac_linking, wrf_tobac_segmentation

################################################################
#################### RUN TRACKING PROGRAMS #####################
################################################################


def _create_xarrs_and_cubes(
    dataset_name: str, CONFIG: dict
) -> tuple[xr.Dataset, iris.cube, xr.Dataset, iris.cube]:
    """


    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    tracking_xarray : xr.Dataset
        An xarray with the feature tracking information.
    tracking_cube : iris.cube
        An iris cube of the feature tracking variable
    segmentation_xarray : xr.Dataset
        An xarray with the feature segmentation information
    segmentation_cube : iris.cube
        An iris cube of the feature segmentation variables
    """

    # Load in the datasets
    # some datasets have different load_netcdf arguments, so account for the differences
    # TODO: eventually we will call RAMS-MAT in rams_load_netcdf_iris, so the only exceptions will be the nexrads
    if dataset_name == "rams":
        tracking_cube, tracking_xarray = globals()[f"{dataset_name}_load_netcdf_iris"](
            CONFIG[dataset_name]["path_to_data"],
            CONFIG[dataset_name]["feature_tracking_var"],
            CONFIG[dataset_name]["path_to_header"],
            CONFIG,
        )

    elif dataset_name == "nexrad" or dataset_name == "multi_nexrad":
        if "gridding" in CONFIG[dataset_name]:
            if CONFIG["verbose"]:
                print(f"=====Gridding {dataset_name.upper()} Data=====")
            tracking_cube, tracking_xarray = globals()[
                f"{dataset_name}_load_netcdf_iris"
            ](
                CONFIG[dataset_name]["path_to_data"],
                "ar2v",
                CONFIG[dataset_name]["feature_tracking_var"],
                CONFIG,
                CONFIG[dataset_name]["gridding"]["gridding_save_path"],
            )

        else:
            tracking_cube, tracking_xarray = globals()[
                f"{dataset_name}_load_netcdf_iris"
            ](
                CONFIG[dataset_name]["path_to_data"],
                "nc",
                CONFIG[dataset_name]["feature_tracking_var"],
                CONFIG,
            )

    else:
        tracking_cube, tracking_xarray = globals()[f"{dataset_name}_load_netcdf_iris"](
            CONFIG[dataset_name]["path_to_data"],
            CONFIG[dataset_name]["feature_tracking_var"],
            CONFIG,
        )

    # if tracking and segmentation variables are different, load seperately
    if (
        CONFIG[dataset_name]["feature_tracking_var"]
        != CONFIG[dataset_name]["segmentation_var"]
    ):
        if dataset_name == "rams":
            segmentation_cube, segmentation_xarray = globals()[
                f"{dataset_name}_load_netcdf_iris"
            ](
                CONFIG[dataset_name]["path_to_data"],
                CONFIG[dataset_name]["segmentation_var"],
                CONFIG[dataset_name]["path_to_header"],
                CONFIG,
            )
        # TODO: check if you need to copy and paste the nexrad/multi-nexrad load configurations for segmentation
        else:
            tracking_cube, tracking_xarray = globals()[
                f"{dataset_name}_load_netcdf_iris"
            ](
                CONFIG[dataset_name]["path_to_data"],
                CONFIG[dataset_name]["feature_tracking_var"],
                CONFIG,
            )

    else:
        segmentation_cube = tracking_cube
        segmentation_xarray = tracking_xarray

    # if reflectivity is required for analysis, add it now
    # currently only implemented for tobac (first clause)
    if ("tobac" in CONFIG[dataset_name]) and (
        "analysis" in CONFIG[dataset_name]["tobac"]
    ):
        if (
            (
                dataset_name == "rams"
                or dataset_name == "wrf"
                or dataset_name == "mesonh"
            )
            and ("eth" in CONFIG[dataset_name]["tobac"]["analysis"])
            and CONFIG[dataset_name]["segmentation_var"].lower() != "dbz"
        ):
            reflectivity_calc = globals()[f"{dataset_name}_calculate_reflectivity"](
                tracking_xarray
            )
            tracking_xarray["DBZ"] = reflectivity_calc
            segmentation_xarray["DBZ"] = reflectivity_calc

    return (tracking_xarray, tracking_cube, segmentation_xarray, segmentation_cube)


def _run_tracker_det_and_seg(
    dataset_name: str,
    tracker: str,
    tracking_info: tuple[xr.Dataset, iris.cube, xr.Dataset, iris.cube],
    CONFIG: dict,
) -> tuple[
    xr.Dataset,
    xr.Dataset,
    list[gpd.GeoDataFrame, gpd.GeoDataFrame, xr.Dataset, xr.Dataset],
]:
    """


    Run feature detection and segmentation on a dataset for a given tracker

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    tracker : str
        The name of the tracker.
    tracking_info : tuple[xr.Dataset, iris.cube, xr.Dataset, iris.cube]
        A tuple of the feature/segmentation xarrays/iris cubes.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    tracking_xarray : xr.Dataset
        An xarray with the feature tracking information.
    segmentation_xarray : xr.Dataset, returned only if tracker == "tobac"
        An xarray with the feature segmentation information
    US_values : list[gpd.GeoDataFrame, gpd.GeoDataFrame, xr.Dataset, xr.Dataset]
        US_features : gpd.GeoDataFrame
            A CoCoMET-US standard features GeoDataFrame.
        US_tracks : gpd.GeoDataFrame
            A CoCoMET-US standard trakcs GeoDataFrame.
        US_segmentation_2d : xr.Dataset
            A CoCoMET-US standard 2D segmentation xarray.
        US_segmentation_3d : xr.Dataset
            A CoCoMET-US standard 3D segmentation xarray.
    """
    # TODO : create error warnings in case some datasets do not have some trackers implemented yet

    # Load in tracking and segmentation info
    tracking_xarray = tracking_info[0]
    tracking_cube = tracking_info[1]
    segmentation_xarray = tracking_info[2]
    segmentation_cube = tracking_info[3]

    # make sure the tracker name is in lower case
    tracker = tracker.lower()

    # now determine which tracker(s) to use
    if tracker == "tobac":
        features = None
        tracks = None
        segmentation2d = (None, None)
        segmentation3d = (None, None)

        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if "feature_id" in CONFIG[dataset_name]["tobac"]:
            if CONFIG["verbose"]:
                print(f"=====Starting {dataset_name.upper()} tobac Feature ID=====")

            features = globals()[f"{dataset_name}_tobac_feature_id"](
                tracking_cube, CONFIG
            )
            if features is None:
                raise Exception("No features identified")

        if "linking" in CONFIG[dataset_name]["tobac"]:
            if CONFIG["verbose"]:
                print(
                    f"=====Starting {dataset_name.upper()} tobac Feature Linking====="
                )

            tracks = globals()[f"{dataset_name}_tobac_linking"](
                tracking_cube, features, CONFIG
            )

        if "segmentation_2d" in CONFIG[dataset_name]["tobac"]:
            if CONFIG["verbose"]:
                print(
                    f"=====Starting {dataset_name.upper()} tobac 2D Segmentation====="
                )

            height = (
                CONFIG[dataset_name]["tobac"]["segmentation_2d"]["height"]
                if "height" in CONFIG[dataset_name]["tobac"]["segmentation_2d"]
                else None
            )

            segmentation2d = globals()[f"{dataset_name}_tobac_segmentation"](
                segmentation_cube,
                features,
                "2d",
                CONFIG,
                height,
            )

        if "segmentation_3d" in CONFIG[dataset_name]["tobac"]:
            if CONFIG["verbose"]:
                print(
                    f"=====Starting {dataset_name.upper()} tobac 3D Segmentation====="
                )

            segmentation3d = globals()[f"{dataset_name}_tobac_segmentation"](
                segmentation_cube, features, "3d", CONFIG
            )

        # Create analysis object values
        US_features = feature_id_to_US(features, "tobac")
        US_tracks = linking_to_US(tracks, "tobac")
        US_segmentation_2d = segmentation_to_US(segmentation2d[0], US_tracks, "tobac")
        US_segmentation_3d = segmentation_to_US(segmentation3d[0], US_tracks, "tobac")

        US_values = [
            US_features,
            US_tracks,
            US_segmentation_2d,
            US_segmentation_3d,
        ]
        return (tracking_xarray, segmentation_xarray, US_values)

    if tracker == "moaap" or tracker == "tams":
        # Run MOAAP or TAMS if present
        if tracker in CONFIG[dataset_name]:
            if CONFIG["verbose"]:
                print(
                    f"=====Starting {dataset_name.upper()} {tracker.upper()} Tracking====="
                )

            # Run MOAAP or TAMS
            mask_output = globals()[f"{dataset_name}_run_{tracker}"](
                tracking_xarray, CONFIG
            )

            # Convert output to US and check if is None
            if tracker == "tams":
                tams_output, latlon_coord_system = mask_output
                if tams_output is None:
                    US_values = [None, None, None]

                else:
                    US_values = bulk_tams_to_US(
                        tams_output,
                        latlon_coord_system,
                        tracking_xarray.PROJX.values,
                        tracking_xarray.PROJY.values,
                        convert_type=CONFIG[dataset_name]["tams"]["analysis_type"],
                    )

            elif tracker == "moaap":
                US_values = bulk_moaap_to_US(
                    mask_output,
                    tracking_xarray.PROJX.values,
                    tracking_xarray.PROJY.values,
                    convert_type=CONFIG[dataset_name][tracker]["analysis_type"],
                )

                if US_values is None:
                    US_values = [None, None, None]

        return (tracking_xarray, US_values)


################################################################
#################### RUN ANALYSIS PROGRAMS #####################
################################################################


def _tobac_analysis(
    dataset_name: str,
    user_return_dict: dict,
    analysis_object: Analysis_Object,
    CONFIG: dict,
) -> dict:
    """


    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    user_return_dict : dict
        A dictionary that is appended to which contains all of the analysis and tracking information for a dataset.
    analysis_object : Analysis_Object
        A  CoCoMET-US standard analysis object containing at least US_tracks, US_segmentation_2d, and US_segmentation_3d.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    user_return_dict : dict
        A dictionary that is appended to which contains all of the analysis and tracking information for a dataset.
    """

    analysis_dictionary = analysis_object.return_analysis_dictionary()
    US_tracks = analysis_dictionary["US_tracks"]
    US_segmentation_2d = analysis_dictionary["US_segmentation_2d"]
    US_segmentation_3d = analysis_dictionary["US_segmentation_3d"]

    # Make an empty dictionary to fill with analysis data
    _tobac_analysis_data = {}

    if "analysis" in CONFIG[dataset_name]["tobac"]:
        if CONFIG[dataset_name]["tobac"]["analysis"] is None:
            CONFIG[dataset_name]["tobac"]["analysis"] = {}

        if CONFIG["verbose"]:
            print(
                f"=====Starting {dataset_name.upper()} tobac Analysis Calculations====="
            )

        # Place the analysis variables which help calculate other variables first in the list if they are to be calculated
        analysis_dict = CONFIG[dataset_name]["tobac"]["analysis"]
        analysis_keys = list(analysis_dict.keys())
        depended_variables = ["volume", "perimeter", "area"]

        for dep_var in depended_variables:
            if dep_var in analysis_keys:
                analysis_keys.remove(dep_var)
                analysis_keys.insert(0, dep_var)

        analysis_vars = {key: analysis_dict[key] for key in analysis_keys}

        # Calcaulte each variable of interest and append to analysis data array
        for var in analysis_vars:
            # Make sure that ETH is calculated with reflectivity
            if var == "eth":
                CONFIG[dataset_name]["tobac"]["analysis"][var]["variable"] = "DBZ"

            # Add default tracking featured_id variable in place of variable if not present
            elif "variable" not in CONFIG[dataset_name]["tobac"]["analysis"][var]:

                if var == "cell_growth":
                    raise Exception("!=====Must input a variable for cell growth/dissipation rate=====!")

                CONFIG[dataset_name]["tobac"]["analysis"][var]["variable"] = CONFIG[
                    dataset_name
                ]["feature_tracking_var"].upper()

            # This allows us to have multiple copies of the same variable by adjoining a dash
            proper_var_name = var.lower().split("-")[0]

            arg_dictionary = deepcopy(_tobac_analysis_data)
            arg_dictionary.update(CONFIG[dataset_name]["tobac"]["analysis"][var])
            _tobac_analysis_data[var] = calc_var(
                analysis_object,
                proper_var_name,
                **arg_dictionary,
            )

    if CONFIG["verbose"]:
        print(f"=====Converting {dataset_name.upper()} tobac Output to CoCoMET-US=====")

    # Add all products to return dict
    user_return_dict[dataset_name]["tobac"] = {
        "US_tracks": US_tracks,
        "US_segmentation_2d": US_segmentation_2d,
        "US_segmentation_3d": US_segmentation_3d,
        "analysis": _tobac_analysis_data,
        "analysis_object": analysis_object,
    }

    if CONFIG["verbose"]:
        print(f"====={dataset_name.upper()} tobac Tracking Complete=====")

    return user_return_dict


def _moaap_analysis(
    dataset_name: str,
    user_return_dict: dict,
    analysis_object: Analysis_Object,
    CONFIG: dict,
) -> dict:
    """


    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    user_return_dict : dict
        A dictionary that is appended to which contains all of the analysis and tracking information for a dataset.
    analysis_object : Analysis_Object
        A  CoCoMET-US standard analysis object containing at least US_tracks, US_segmentation_2d, and US_segmentation_3d.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    user_return_dict : dict
        A dictionary that is appended to which contains all of the analysis and tracking information for a dataset.
    """
    analysis_dictionary = analysis_object.return_analysis_dictionary()
    US_tracks = analysis_dictionary["US_tracks"]
    US_segmentation_2d = analysis_dictionary["US_segmentation_2d"]
    US_segmentation_3d = analysis_dictionary["US_segmentation_3d"]

    # Make an empty dictionary to fill with analysis data
    _moaap_analysis_data = {}

    # Run analysis on MOAAP output
    if "analysis" in CONFIG[dataset_name]["moaap"]:
        if CONFIG[dataset_name]["moaap"]["analysis"] is None:
            CONFIG[dataset_name]["moaap"]["analysis"] = {}

        if CONFIG["verbose"]:
            print(
                f"=====Starting {dataset_name.upper()} MOAAP Analysis Calculations====="
            )

        if (US_tracks is not None and US_segmentation_2d is not None) or (
            US_tracks is not None and US_segmentation_3d is not None
        ):
            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG[dataset_name]["moaap"]["analysis"].keys():
                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG[dataset_name]["moaap"]["analysis"][var]:
                    CONFIG[dataset_name]["moaap"]["analysis"][var]["variable"] = CONFIG[
                        dataset_name
                    ]["feature_tracking_var"].upper()

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                _moaap_analysis_data[var] = calc_var(
                    analysis_object,
                    proper_var_name,
                    **CONFIG[dataset_name]["moaap"]["analysis"][var],
                )
        else:
            print(
                f"=====No MOAAP Tracking Information Found for {dataset_name}, Skipping Analysis====="
            )

    user_return_dict[dataset_name]["moaap"] = {
        # "mask_xarray": mask_output,
        # "US_features": US_values[0],
        "US_tracks": US_tracks,
        "US_segmentation_2d": US_segmentation_2d,
        "analysis": _moaap_analysis_data,
        "analysis_object": analysis_object,
    }

    if CONFIG["verbose"]:
        print(f"====={dataset_name.upper()} MOAAP Tracking Complete=====")

    return user_return_dict


def _tams_analysis(
    dataset_name: str,
    user_return_dict: dict,
    analysis_object: Analysis_Object,
    CONFIG: dict,
) -> dict:
    """


    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    user_return_dict : dict
        A dictionary that is appended to which contains all of the analysis and tracking information for a dataset.
    analysis_object : Analysis_Object
        A  CoCoMET-US standard analysis object containing at least US_tracks, US_segmentation_2d, and US_segmentation_3d.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    user_return_dict : dict
        A dictionary that is appended to which contains all of the analysis and tracking information for a dataset.
    """

    analysis_dictionary = analysis_object.return_analysis_dictionary()
    US_tracks = analysis_dictionary["US_tracks"]
    US_segmentation_2d = analysis_dictionary["US_segmentation_2d"]
    US_segmentation_3d = analysis_dictionary["US_segmentation_3d"]

    # Make an empty dictionary to fill with analysis data
    _tams_analysis_data = {}

    # Run analysis on TAMS output
    if "analysis" in CONFIG[dataset_name]["tams"]:
        if CONFIG[dataset_name]["tams"]["analysis"] is None:
            CONFIG[dataset_name]["tams"]["analysis"] = {}

        if CONFIG["verbose"]:
            print(
                f"=====Starting {dataset_name.upper()} TAMS Analysis Calculations====="
            )

        if (US_tracks is not None and US_segmentation_2d is not None) or (
            US_tracks is not None and US_segmentation_3d is not None
        ):
            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG[dataset_name]["tams"]["analysis"].keys():
                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG[dataset_name]["tams"]["analysis"][var]:
                    CONFIG[dataset_name]["tams"]["analysis"][var]["variable"] = CONFIG[
                        dataset_name
                    ]["feature_tracking_var"].upper()

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                _tams_analysis_data[var] = calc_var(
                    analysis_object,
                    proper_var_name,
                    **CONFIG[dataset_name]["tams"]["analysis"][var],
                )
        else:
            print(
                f"=====No TAMS Tracking Information Found for {dataset_name}, Skipping Analysis====="
            )

    user_return_dict[dataset_name]["tams"] = {
        # "mask_xarray": mask_output,
        # "US_features": US_values[0],
        "US_tracks": US_tracks,
        "US_segmentation_2d": US_segmentation_2d,
        "analysis": _tams_analysis_data,
        "analysis_object": analysis_object,
    }

    if CONFIG["verbose"]:
        print(f"====={dataset_name.upper()} TAMS Tracking Complete=====")

    return user_return_dict


################################################################
####################### RUN ALL PROGRAMS #######################
################################################################


# run_tracker needs to be at the bottom of the script to access _moaap_analysis and _tams_analysis
def run_tracker(dataset_name, tracker, user_return_dict, tracking_info, CONFIG):
    """
    Run a tracker on a dataset. If no tracker has been previously run on the dataset,
    user_return_dict should be empty

    Inputs:
        dataset_name : a string giving the name of the data you would like to track
        tracker : a string giving the name of the tracker you would like to use
        user_return_dict : a dictionary of the previous dataset processed information
        CONFIG : an ANTE-TRACE CONFIG file

    Returns:
        user_return_dict : a dictionary with current processed information
    """

    # Create a user dictionary key for the dataset if there is not one
    if dataset_name not in user_return_dict:
        user_return_dict[dataset_name] = {}

    # If the iris cubes have not been created already, do so now
    if tracking_info is None:
        tracking_info = _create_xarrs_and_cubes(dataset_name, CONFIG)

    # Determine which trackers to track on

    if tracker == "tobac":
        (tracking_xarray, segmentation_xarray, US_values) = _run_tracker_det_and_seg(
            dataset_name, tracker, tracking_info, CONFIG
        )

        # Create analysis object - for tobac
        analysis_object = Analysis_Object(
            tracking_xarray, segmentation_xarray, *US_values
        )

        # Filter the cells
        # analysis_object = filter_cells(analysis_object) # there is an issue with the memory for high resolution data sets

        user_return_dict = _tobac_analysis(
            dataset_name, user_return_dict, analysis_object, CONFIG
        )

        return user_return_dict, tracking_info

    if tracker == "moaap" or tracker == "tams":
        tracking_xarray, US_values = _run_tracker_det_and_seg(
            dataset_name, tracker, tracking_info, CONFIG
        )

        # Create analysis object - for moaap
        analysis_object = Analysis_Object(
            tracking_xarray,
            tracking_xarray,
            *US_values,
            None,  # No 3d segmentation
        )

        # Filter the cells
        # analysis_object = filter_cells(analysis_object)

        if tracker == "moaap":
            user_return_dict = _moaap_analysis(
                dataset_name,
                user_return_dict,
                analysis_object,
                CONFIG,
            )

        elif tracker == "tams":
            user_return_dict = _tams_analysis(
                dataset_name,
                user_return_dict,
                analysis_object,
                CONFIG,
            )

        return user_return_dict, tracking_info
