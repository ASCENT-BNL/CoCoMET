#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:55:16 2024

@author: thahn
"""

# =============================================================================
# This is the interface layer between all of CoMET"s backend functionality and the user. A sort of parser for a configuration input file.
# =============================================================================
# TODO: Create CoMET-UDAF Specification for return object, Update strings to be proper doc strings (type """ then press enter at top of function)


def CoMET_start(path_to_config=None, manual_mode=False, CONFIG=None):
    """
    Inputs:
        path_to_config: Path to a config.yml file containing all details of the CoMET run. See boilerplate.yml for how the file should be setup
        manual_mode: [True, False] Whether CoMET should run all functions idenpendetly or should just return CONFIG for user to run functions independently
        CONFIG: Optional to just pass a config dict object instead of filepath
    """

    import time

    # Load CONFIG if not present
    if CONFIG is None:
        CONFIG = CoMET_load(path_to_config)

    # if manual_mode = True, just return the loaded CONFIG, otherwise, go through defined setups and run them
    if manual_mode:
        return CONFIG

    # If parallelization is True, run the multiprocessing version instead
    if CONFIG["parallel_processing"]:
        import os

        os.environ["OMP_NUM_THREADS"] = str(
            CONFIG["max_cores"] * 2
        )  # Take advantage of hyper threading

        # Return CoMET multi processes output which should be a dictionary
        multi_output = CoMET_start_multi(CONFIG)

        return multi_output

    start_time = time.perf_counter()

    # Create empty dictionaries for each data type
    wrf_data = mesonh_data = nexrad_data = multi_nexrad_data = standard_radar_data = (
        goes_data
    ) = {}

    # if wrf is present in CONFIG, run the necessary wrf functions
    if "wrf" in CONFIG:

        # Call run wrf function to handle all wrf tasks
        wrf_data = run_wrf(CONFIG)

    # Handle MesoNH data
    if "mesonh" in CONFIG:

        # Call run mesonh function to handle all mesonh tasks
        mesonh_data = run_mesonh(CONFIG)

    # Handle NEXRAD data
    if "nexrad" in CONFIG:

        # Call run nexrad function to handle all nexrad tasks
        nexrad_data = run_nexrad(CONFIG)

    # Handle Multi-NEXRAD data
    if "multi_nexrad" in CONFIG:

        # Call run multi nexrad function to handle all multi nexrad tasks
        multi_nexrad_data = run_multi_nexrad(CONFIG)

    # Handle standard radar data
    if "standard_radar" in CONFIG:

        # Call run standard radar function to handle all standard radar tasks
        standard_radar_data = run_standard_radar(CONFIG)

    # Handle GOES data
    if "goes" in CONFIG:

        # Call run goes function to handle all goes tasks
        goes_data = run_goes(CONFIG)

    end_time = time.perf_counter()

    if CONFIG["verbose"]:
        print(
            f"""=====CoMET Performance Diagonistics=====\n$ Total Process Time: {"%.2f Seconds" % (end_time-start_time)}\n$ Allocated Resources: Cores = 1"""
        )

    # Return dict at end
    return (
        wrf_data
        | mesonh_data
        | nexrad_data
        | multi_nexrad_data
        | standard_radar_data
        | goes_data
    )


def CoMET_start_multi(CONFIG):
    """
    Inputs:
        CONFIG: User defined configuration data
    Outputs:
        return_dict: Dictionary designed according to the CoMET-UDAF standard
    """

    import time
    import multiprocessing

    # This is necessary for python reasons I suppose may need to check this after some kind of release
    if __name__ == "CoMET.user_interface_layer":

        start_time = time.perf_counter()

        # Keep track of active core count. Should start at one.
        active_process_count = 1

        # Keep track of for when doing gridding
        inital_max_cores = CONFIG["max_cores"]

        # Start a queue so processes can finish at different times
        queue = multiprocessing.Queue()

        processes = []
        responses = []

        # if wrf is present in CONFIG, run the necessary wrf functions
        if "wrf" in CONFIG:

            # Check to make sure max core count has not been exceeded
            if (
                CONFIG["max_cores"] is not None
                and active_process_count + 1 > CONFIG["max_cores"]
            ):
                raise Exception(
                    "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                )

            # Call run wrf function to handle all wrf tasks
            wrf_process = multiprocessing.Process(target=run_wrf, args=(CONFIG, queue))
            processes.append(wrf_process)
            wrf_process.start()

            active_process_count += 1

        # Handle MesoNH data
        if "mesonh" in CONFIG:

            # Check to make sure max core count has not been exceeded
            if (
                CONFIG["max_cores"] is not None
                and active_process_count + 1 > CONFIG["max_cores"]
            ):
                raise Exception(
                    "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                )

            # Call run MesoNH function to handle all MesoNH tasks
            mesonh_process = multiprocessing.Process(
                target=run_mesonh, args=(CONFIG, queue)
            )
            processes.append(mesonh_process)
            mesonh_process.start()

            active_process_count += 1

        # Handle GOES data
        if "goes" in CONFIG:

            # Check to make sure max core count has not been exceeded
            if (
                CONFIG["max_cores"] is not None
                and active_process_count + 1 > CONFIG["max_cores"]
            ):
                raise Exception(
                    "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                )

            # Call run goes function to handle all goes tasks
            goes_process = multiprocessing.Process(
                target=run_goes, args=(CONFIG, queue)
            )
            processes.append(goes_process)
            goes_process.start()

            active_process_count += 1

        # Handle NEXRAD data
        if "nexrad" in CONFIG:

            # Check to make sure max core count has not been exceeded
            if (
                CONFIG["max_cores"] is not None
                and active_process_count + 1 > CONFIG["max_cores"]
            ):
                raise Exception(
                    "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                )

            # If gridding is enabled, make sure doing so will not exceed the max number of cores and adjust the CONFIG values accordingly
            if CONFIG["max_cores"] is not None and "gridding" in CONFIG["nexrad"]:
                CONFIG["max_cores"] = CONFIG["max_cores"] - active_process_count - 1

                if CONFIG["max_cores"] <= 0:
                    raise Exception(
                        "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                    )

            # Call run nexrad function to handle all nexrad tasks
            nexrad_process = multiprocessing.Process(
                target=run_nexrad, args=(CONFIG, queue)
            )
            processes.append(nexrad_process)
            nexrad_process.start()

            active_process_count += 1

        # Handle Multi NEXRAD data
        if "multi_nexrad" in CONFIG:

            # Check to make sure max core count has not been exceeded
            if (
                CONFIG["max_cores"] is not None
                and active_process_count + 1 > CONFIG["max_cores"]
            ):
                raise Exception(
                    "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                )

            # If gridding is enabled, make sure doing so will not exceed the max number of cores and adjust the CONFIG values accordingly
            if CONFIG["max_cores"] is not None and "gridding" in CONFIG["multi_nexrad"]:
                CONFIG["max_cores"] = CONFIG["max_cores"] - active_process_count - 1

                if CONFIG["max_cores"] <= 0:
                    raise Exception(
                        "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                    )

            # Call run nexrad function to handle all nexrad tasks
            multi_nexrad_process = multiprocessing.Process(
                target=run_multi_nexrad, args=(CONFIG, queue)
            )
            processes.append(multi_nexrad_process)
            multi_nexrad_process.start()

            active_process_count += 1

        # Handle standard radar data
        if "standard_radar" in CONFIG:

            # Check to make sure max core count has not been exceeded
            if (
                CONFIG["max_cores"] is not None
                and active_process_count + 1 > CONFIG["max_cores"]
            ):
                raise Exception(
                    "!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!"
                )

            # Call run goes function to handle all goes tasks
            radar_process = multiprocessing.Process(
                target=run_standard_radar, args=(CONFIG, queue)
            )
            processes.append(radar_process)
            radar_process.start()

            active_process_count += 1

        for p in processes:
            responses.append(queue.get())

        for p in processes:
            p.join()
            active_process_count -= 1

        return_dict = {}

        for ii in range(len(responses)):
            return_dict = return_dict | responses[ii]

        # Reset CONFIG max cores if necessary
        CONFIG["max_cores"] = inital_max_cores

        end_time = time.perf_counter()

        if CONFIG["verbose"]:
            print(
                f"""=====CoMET Performance Diagonistics=====\n$ Total Process Time: {"%.2f Seconds" % (end_time-start_time)}\n$ Allocated Resources: Cores = {CONFIG["max_cores"]}"""
            )

        # Return dict at end
        return return_dict


def CoMET_load(path_to_config=None, CONFIG_string=None):
    """
    Inputs:
        path_to_config: Path to a config.yml file containing all details of the CoMET run. See boilerplate.yml for how the file should be setup
        CONFIG_string: String of yaml data if not using a file
    Outputs:
        CONFIG: dictionary object containing all user-defined parameters
    """

    import ast
    import yaml
    import numpy as np

    if CONFIG_string is None:

        # Open and read config file
        with open(path_to_config, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f)

    else:

        CONFIG = yaml.safe_load(CONFIG_string)

    # Check for default setup parameters, add them if not present
    if "verbose" not in CONFIG:
        CONFIG["verbose"] = True

    if "parallel_processing" not in CONFIG:
        CONFIG["parallel_processing"] = False

    # Go through each potential option and determine which functions need to run
    if "wrf" in CONFIG and CONFIG["verbose"]:
        print("=====WRF Setup Found in CONFIG=====")

    if "mesonh" in CONFIG and CONFIG["verbose"]:
        print("=====MesoNH Setup Found in CONFIG=====")

    # if nexrad present, check for tuples
    if "nexrad" in CONFIG:

        if CONFIG["verbose"]:
            print("=====NEXRAD Setup Found in CONFIG=====")

        # If nexrad gridding is needed, change grid shapes and limits back to tuples
        if "gridding" in CONFIG["nexrad"]:

            # Convert grid_shape and grid_limits back into proper tuples and ensure correct data types--auto correct automatically
            grid_shape = ast.literal_eval(CONFIG["nexrad"]["gridding"]["grid_shape"])
            # Ensure int grid shapes
            grid_shape = tuple([int(grid_shape[ii]) for ii in range(len(grid_shape))])

            # Ensure float for grid_limits
            grid_limits = ast.literal_eval(CONFIG["nexrad"]["gridding"]["grid_limits"])
            grid_limits = np.array(grid_limits).astype(float)
            grid_limits = tuple([tuple(row) for row in grid_limits])

            # Adjust CONFIG values
            CONFIG["nexrad"]["gridding"]["grid_shape"] = grid_shape
            CONFIG["nexrad"]["gridding"]["grid_limits"] = grid_limits

    # if multi nexrad present, check for tuples
    if "multi_nexrad" in CONFIG:

        if CONFIG["verbose"]:
            print("=====Multi-NEXRAD Setup Found in CONFIG=====")

        # If nexrad gridding is needed, change grid shapes and limits back to tuples
        if "gridding" in CONFIG["multi_nexrad"]:

            # Convert grid_shape and grid_limits back into proper tuples and ensure correct data types--auto correct automatically
            grid_shape = ast.literal_eval(
                CONFIG["multi_nexrad"]["gridding"]["grid_shape"]
            )
            # Ensure int grid shapes
            grid_shape = tuple([int(grid_shape[ii]) for ii in range(len(grid_shape))])

            # Ensure float for grid_limits
            grid_limits = ast.literal_eval(
                CONFIG["multi_nexrad"]["gridding"]["grid_limits"]
            )
            grid_limits = np.array(grid_limits).astype(float)
            grid_limits = tuple([tuple(row) for row in grid_limits])

            # Adjust CONFIG values
            CONFIG["multi_nexrad"]["gridding"]["grid_shape"] = grid_shape
            CONFIG["multi_nexrad"]["gridding"]["grid_limits"] = grid_limits

    if "standard_radar" in CONFIG and CONFIG["verbose"]:
        print("=====RADAR Setup Found in CONFIG")

    if "goes" in CONFIG and CONFIG["verbose"]:
        print("=====GOES Setup Found in CONFIG=====")

    return CONFIG


# =============================================================================
# This section is for running individual data types
# =============================================================================


def run_wrf(CONFIG, queue=None):
    """
    Inputs:
        CONFIG: User configuration file
    Outputs:
        user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
    """

    from .tracker_output_translation_layer import (
        feature_id_to_UDAF,
        linking_to_UDAF,
        segmentation_to_UDAF,
        bulk_moaap_to_UDAF,
    )
    from CoMET.analysis.analysis_object import Analysis_Object
    from .wrf_load import wrf_load_netcdf_iris

    if CONFIG["verbose"]:
        print("=====Loading WRF Data=====")

    wrf_tracking_cube, wrf_tracking_xarray = wrf_load_netcdf_iris(
        CONFIG["wrf"]["path_to_data"], CONFIG["wrf"]["feature_tracking_var"], CONFIG
    )

    # if tracking and segmentation variables are different, load seperately
    if CONFIG["wrf"]["feature_tracking_var"] != CONFIG["wrf"]["segmentation_var"]:

        wrf_segmentation_cube, wrf_segmentation_xarray = wrf_load_netcdf_iris(
            CONFIG["wrf"]["path_to_data"], CONFIG["wrf"]["segmentation_var"], CONFIG
        )

    else:

        wrf_segmentation_cube = wrf_tracking_cube
        wrf_segmentation_xarray = wrf_tracking_xarray

    # Add xarrays and cubes to return dict
    user_return_dict = {}

    user_return_dict["wrf"] = {
        # "tracking_xarray": wrf_tracking_xarray,
        # "tracking_cube": wrf_tracking_cube,
        # "segmentation_xarray": wrf_segmentation_xarray,
        # "segmentation_cube": wrf_segmentation_cube,
    }

    # now determine which tracker(s) to use
    if "tobac" in CONFIG["wrf"]:
        from .wrf_tobac import (
            wrf_tobac_feature_id,
            wrf_tobac_linking,
            wrf_tobac_segmentation,
        )

        wrf_features = None
        wrf_tracks = None
        wrf_segmentation2d = (None, None)
        wrf_segmentation3d = (None, None)
        wrf_tobac_analysis_data = {}

        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if "feature_id" in CONFIG["wrf"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting WRF tobac Feature ID=====")

            wrf_features = wrf_tobac_feature_id(wrf_tracking_cube, CONFIG)

        if "linking" in CONFIG["wrf"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting WRF tobac Feature Linking=====")

            wrf_tracks = wrf_tobac_linking(wrf_tracking_cube, wrf_features, CONFIG)

        if "segmentation_2d" in CONFIG["wrf"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting WRF tobac 2D Segmentation=====")

            wrf_segmentation2d = wrf_tobac_segmentation(
                wrf_segmentation_cube,
                wrf_features,
                "2d",
                CONFIG,
                CONFIG["wrf"]["tobac"]["segmentation_2d"]["height"],
            )

        if "segmentation_3d" in CONFIG["wrf"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting WRF tobac 3D Segmentation=====")

            wrf_segmentation3d = wrf_tobac_segmentation(
                wrf_segmentation_cube, wrf_features, "3d", CONFIG
            )

        # Create analysis object values
        UDAF_features = feature_id_to_UDAF(wrf_features, "tobac")
        UDAF_tracks = linking_to_UDAF(wrf_tracks, "tobac")
        UDAF_segmentation_2d = segmentation_to_UDAF(
            wrf_segmentation2d[0], UDAF_tracks, "tobac"
        )
        UDAF_segmentation_3d = segmentation_to_UDAF(
            wrf_segmentation3d[0], UDAF_tracks, "tobac"
        )

        # Create analysis object
        analysis_object = Analysis_Object(
            wrf_tracking_xarray,
            wrf_segmentation_xarray,
            UDAF_features,
            UDAF_tracks,
            UDAF_segmentation_2d,
            UDAF_segmentation_3d,
        )

        if "analysis" in CONFIG["wrf"]["tobac"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["wrf"]["tobac"]["analysis"] is None:
                CONFIG["wrf"]["tobac"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting WRF tobac Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["wrf"]["tobac"]["analysis"].keys():

                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG["wrf"]["tobac"]["analysis"][var.lower()]:
                    CONFIG["wrf"]["tobac"]["analysis"][var.lower()]["variable"] = (
                        CONFIG["wrf"]["feature_tracking_var"].upper()
                    )

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                wrf_tobac_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["wrf"]["tobac"]["analysis"][var.lower()],
                )

        if CONFIG["verbose"]:
            print("=====Converting WRF tobac Output to CoMET-UDAF=====")

        # Add all products to return dict
        user_return_dict["wrf"]["tobac"] = {
            # "feature_id": wrf_features,
            # "UDAF_features": feature_id_to_UDAF(wrf_features, "tobac"),
            # "linking": wrf_tracks,
            "UDAF_linking": UDAF_tracks,
            # "segmentation_2d": wrf_segmentation2d,
            "UDAF_segmentation_2d": UDAF_segmentation_2d,
            # "segmentation_3d": wrf_segmentation3d,
            "UDAF_segmentation_3d": UDAF_segmentation_3d,
            "analysis": wrf_tobac_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====WRF tobac Tracking Complete=====")

    # Run MOAAP if present
    if "moaap" in CONFIG["wrf"]:
        from .wrf_moaap import wrf_moaap

        wrf_moaap_analysis_data = {}

        if CONFIG["verbose"]:
            print("=====Starting WRF MOAAP Tracking=====")

        # Run MOAAP
        mask_output = wrf_moaap(wrf_tracking_xarray, CONFIG)

        # Calculate UDAF values
        UDAF_values = bulk_moaap_to_UDAF(
            mask_output,
            wrf_tracking_xarray.PROJX.values,
            wrf_tracking_xarray.PROJY.values,
            convert_type=CONFIG["wrf"]["moaap"]["analysis_type"],
        )

        if UDAF_values is None:
            UDAF_values = [None, None, None]

        # Create analysis object
        analysis_object = Analysis_Object(
            wrf_tracking_xarray,
            wrf_tracking_xarray,
            UDAF_values[0],
            UDAF_values[1],
            UDAF_values[2],
            None,
        )

        # Run analysis on MOAAP output
        if "analysis" in CONFIG["wrf"]["moaap"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["wrf"]["moaap"]["analysis"] is None:
                CONFIG["wrf"]["moaap"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting WRF MOAAP Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["wrf"]["moaap"]["analysis"].keys():

                if UDAF_values == [None, None, None]:
                    continue

                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG["wrf"]["moaap"]["analysis"][var.lower()]:
                    CONFIG["wrf"]["moaap"]["analysis"][var.lower()]["variable"] = (
                        CONFIG["wrf"]["feature_tracking_var"].upper()
                    )

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                wrf_moaap_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["wrf"]["moaap"]["analysis"][var.lower()],
                )

        user_return_dict["wrf"]["moaap"] = {
            # "mask_xarray": mask_output,
            # "UDAF_features": UDAF_values[0],
            "UDAF_linking": UDAF_values[1],
            "UDAF_segmentation_2d": UDAF_values[2],
            "analysis": wrf_moaap_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====WRF MOAAP Tracking Complete=====")

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_mesonh(CONFIG, queue=None):
    """
    Inputs:
        CONFIG: User configuration file
    Outputs:
        user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
    """

    from .tracker_output_translation_layer import (
        feature_id_to_UDAF,
        linking_to_UDAF,
        segmentation_to_UDAF,
        bulk_moaap_to_UDAF,
    )
    from CoMET.analysis.analysis_object import Analysis_Object
    from .mesonh_load import mesonh_load_netcdf_iris

    if CONFIG["verbose"]:
        print("=====Loading MesoNH Data=====")

    mesonh_tracking_cube, mesonh_tracking_xarray = mesonh_load_netcdf_iris(
        CONFIG["mesonh"]["path_to_data"],
        CONFIG["mesonh"]["feature_tracking_var"],
        CONFIG,
    )

    # if tracking and segmentation variables are different, load seperately
    if CONFIG["mesonh"]["feature_tracking_var"] != CONFIG["mesonh"]["segmentation_var"]:

        mesonh_segmentation_cube, mesonh_segmentation_xarray = mesonh_load_netcdf_iris(
            CONFIG["mesonh"]["path_to_data"],
            CONFIG["mesonh"]["segmentation_var"],
            CONFIG,
        )

    else:

        mesonh_segmentation_cube = mesonh_tracking_cube
        mesonh_segmentation_xarray = mesonh_tracking_xarray

    # Add xarrays and cubes to return dict
    user_return_dict = {}

    user_return_dict["mesonh"] = {
        # "tracking_xarray": mesonh_tracking_xarray,
        # "tracking_cube": mesonh_tracking_cube,
        # "segmentation_xarray": mesonh_segmentation_xarray,
        # "segmentation_cube": mesonh_segmentation_cube,
    }

    # now determine which tracker to use
    if "tobac" in CONFIG["mesonh"]:
        from .mesonh_tobac import (
            mesonh_tobac_feature_id,
            mesonh_tobac_linking,
            mesonh_tobac_segmentation,
        )

        mesonh_features = None
        mesonh_tracks = None
        mesonh_segmentation2d = (None, None)
        mesonh_segmentation3d = (None, None)
        mesonh_tobac_analysis_data = {}

        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if "feature_id" in CONFIG["mesonh"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting MesoNH tobac Feature ID=====")

            mesonh_features = mesonh_tobac_feature_id(mesonh_tracking_cube, CONFIG)

        if "linking" in CONFIG["mesonh"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting MesoNH tobac Feature Linking=====")

            mesonh_tracks = mesonh_tobac_linking(
                mesonh_tracking_cube, mesonh_features, CONFIG
            )

        if "segmentation_2d" in CONFIG["mesonh"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting MesoNH tobac 2D Segmentation=====")

            mesonh_segmentation2d = mesonh_tobac_segmentation(
                mesonh_segmentation_cube,
                mesonh_features,
                "2d",
                CONFIG,
                CONFIG["mesonh"]["tobac"]["segmentation_2d"]["height"],
            )

        if "segmentation_3d" in CONFIG["mesonh"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting MesoNH tobac 3D Segmentation=====")

            mesonh_segmentation3d = mesonh_tobac_segmentation(
                mesonh_segmentation_cube, mesonh_features, "3d", CONFIG
            )

        # Create analysis object values
        UDAF_features = feature_id_to_UDAF(mesonh_features, "tobac")
        UDAF_tracks = linking_to_UDAF(mesonh_tracks, "tobac")
        UDAF_segmentation_2d = segmentation_to_UDAF(
            mesonh_segmentation2d[0], UDAF_tracks, "tobac"
        )
        UDAF_segmentation_3d = segmentation_to_UDAF(
            mesonh_segmentation3d[0], UDAF_tracks, "tobac"
        )

        # Create analysis object
        analysis_object = Analysis_Object(
            mesonh_tracking_xarray,
            mesonh_segmentation_xarray,
            UDAF_features,
            UDAF_tracks,
            UDAF_segmentation_2d,
            UDAF_segmentation_3d,
        )

        if "analysis" in CONFIG["mesonh"]["tobac"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["mesonh"]["tobac"]["analysis"] is None:
                CONFIG["mesonh"]["tobac"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting MesoNH tobac Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["mesonh"]["tobac"]["analysis"].keys():

                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG["mesonh"]["tobac"]["analysis"][var.lower()]:
                    CONFIG["mesonh"]["tobac"]["analysis"][var.lower()]["variable"] = (
                        CONFIG["mesonh"]["feature_tracking_var"].upper()
                    )

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                mesonh_tobac_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["mesonh"]["tobac"]["analysis"][var.lower()],
                )

        if CONFIG["verbose"]:
            print("=====Converting MesoNH tobac Output to CoMET-UDAF=====")

        # Add all products to return dict
        user_return_dict["mesonh"]["tobac"] = {
            # "feature_id": mesonh_features,
            # "UDAF_features": feature_id_to_UDAF(mesonh_features, "tobac"),
            # "linking": mesonh_tracks,
            "UDAF_linking": UDAF_tracks,
            # "segmentation_2d": mesonh_segmentation2d,
            "UDAF_segmentation_2d": UDAF_segmentation_2d,
            # "segmentation_3d": mesonh_segmentation3d,
            "UDAF_segmentation_3d": UDAF_segmentation_3d,
            "analysis": mesonh_tobac_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====MesoNH tobac Tracking Complete=====")

    # Run MOAAP if present
    if "moaap" in CONFIG["mesonh"]:
        from .mesonh_moaap import mesonh_moaap

        messonh_moaap_analysis_data = {}

        if CONFIG["verbose"]:
            print("=====Starting MesoNH MOAAP Tracking=====")

        # Run MOAAP
        mask_output = mesonh_moaap(mesonh_tracking_xarray, CONFIG)

        # Calculate UDAF values
        UDAF_values = bulk_moaap_to_UDAF(
            mask_output,
            mesonh_tracking_xarray.PROJX.values,
            mesonh_tracking_xarray.PROJY.values,
            convert_type=CONFIG["mesonh"]["moaap"]["analysis_type"],
        )

        if UDAF_values is None:
            UDAF_values = [None, None, None]

        # Create analysis object
        analysis_object = Analysis_Object(
            mesonh_tracking_xarray,
            mesonh_tracking_xarray,
            UDAF_values[0],
            UDAF_values[1],
            UDAF_values[2],
            None,
        )

        # Run analysis on MOAAP output
        if "analysis" in CONFIG["mesonh"]["moaap"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["mesonh"]["moaap"]["analysis"] is None:
                CONFIG["mesonh"]["moaap"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting MesoNH MOAAP Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["mesonh"]["moaap"]["analysis"].keys():

                if UDAF_values == [None, None, None]:
                    continue

                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG["mesonh"]["moaap"]["analysis"][var.lower()]:
                    CONFIG["mesonh"]["moaap"]["analysis"][var.lower()]["variable"] = (
                        CONFIG["mesonh"]["feature_tracking_var"].upper()
                    )

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                messonh_moaap_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["mesonh"]["moaap"]["analysis"][var.lower()],
                )

        user_return_dict["mesonh"]["moaap"] = {
            # "mask_xarray": mask_output,
            # "UDAF_features": UDAF_values[0],
            "UDAF_linking": UDAF_values[1],
            "UDAF_segmentation_2d": UDAF_values[2],
            "analysis": messonh_moaap_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====MesoNH MOAAP Tracking Complete=====")

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_nexrad(CONFIG, queue=None):
    """
    Inputs:
        CONFIG: User configuration file
    Outputs:
        user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
    """

    from .tracker_output_translation_layer import (
        feature_id_to_UDAF,
        linking_to_UDAF,
        segmentation_to_UDAF,
    )
    from CoMET.analysis import Analysis_Object
    from .nexrad_load import nexrad_load_netcdf_iris

    if CONFIG["verbose"]:
        print("=====Loading NEXRAD Data=====")

    # determine if gridding is necessary or not
    if "gridding" in CONFIG["nexrad"]:

        if CONFIG["verbose"]:
            print("=====Gridding NEXRAD Data=====")
        nexrad_tracking_cube, nexrad_tracking_xarray = nexrad_load_netcdf_iris(
            CONFIG["nexrad"]["path_to_data"],
            "ar2v",
            CONFIG["nexrad"]["feature_tracking_var"],
            CONFIG,
            CONFIG["nexrad"]["gridding"]["gridding_save_path"],
        )

    else:
        nexrad_tracking_cube, nexrad_tracking_xarray = nexrad_load_netcdf_iris(
            CONFIG["nexrad"]["path_to_data"],
            "nc",
            CONFIG["nexrad"]["feature_tracking_var"],
            CONFIG,
        )

    # Add xarrays and cubes to return dict
    user_return_dict = {}

    user_return_dict["nexrad"] = {
        # "tracking_xarray": nexrad_tracking_xarray,
        # "tracking_cube": nexrad_tracking_cube,
        # "segmentation_xarray": nexrad_tracking_xarray,
        # "segmentation_cube": nexrad_tracking_cube,
    }

    # determine which tracker to use
    if "tobac" in CONFIG["nexrad"]:
        from .nexrad_tobac import (
            nexrad_tobac_feature_id,
            nexrad_tobac_linking,
            nexrad_tobac_segmentation,
        )

        nexrad_features = None
        nexrad_tracks = None
        nexrad_segmentation2d = (None, None)
        nexrad_segmentation3d = (None, None)
        nexrad_tobac_analysis_data = {}

        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if "feature_id" in CONFIG["nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting NEXRAD tobac Feature ID=====")

            nexrad_features = nexrad_tobac_feature_id(nexrad_tracking_cube, CONFIG)

        if "linking" in CONFIG["nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting NEXRAD tobac Feature Linking=====")

            nexrad_tracks = nexrad_tobac_linking(
                nexrad_tracking_cube, nexrad_features, CONFIG
            )

        if "segmentation_2d" in CONFIG["nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting NEXRAD tobac 2D Segmentation=====")

            nexrad_segmentation2d = nexrad_tobac_segmentation(
                nexrad_tracking_cube,
                nexrad_features,
                "2d",
                CONFIG,
                CONFIG["nexrad"]["tobac"]["segmentation_2d"]["height"],
            )

        if "segmentation_3d" in CONFIG["nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting NEXRAD tobac 3D Segmentation=====")

            nexrad_segmentation3d = nexrad_tobac_segmentation(
                nexrad_tracking_cube, nexrad_features, "3d", CONFIG
            )

        # Create analysis object values
        UDAF_features = feature_id_to_UDAF(nexrad_features, "tobac")
        UDAF_tracks = linking_to_UDAF(nexrad_tracks, "tobac")
        UDAF_segmentation_2d = segmentation_to_UDAF(
            nexrad_segmentation2d[0], UDAF_tracks, "tobac"
        )
        UDAF_segmentation_3d = segmentation_to_UDAF(
            nexrad_segmentation3d[0], UDAF_tracks, "tobac"
        )

        # Create analysis object
        analysis_object = Analysis_Object(
            nexrad_tracking_xarray,
            nexrad_tracking_xarray,
            UDAF_features,
            UDAF_tracks,
            UDAF_segmentation_2d,
            UDAF_segmentation_3d,
        )

        if "analysis" in CONFIG["nexrad"]["tobac"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["nexrad"]["tobac"]["analysis"] is None:
                CONFIG["nexrad"]["tobac"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting NEXRAD tobac Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["nexrad"]["tobac"]["analysis"].keys():

                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG["nexrad"]["tobac"]["analysis"][var.lower()]:
                    CONFIG["nexrad"]["tobac"]["analysis"][var.lower()]["variable"] = (
                        CONFIG["nexrad"]["feature_tracking_var"].upper()
                    )

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                nexrad_tobac_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["nexrad"]["tobac"]["analysis"][var.lower()],
                )

        if CONFIG["verbose"]:
            print("=====Converting NEXRAD tobac Output to CoMET-UDAF=====")

        # Add all products to return dict
        user_return_dict["nexrad"]["tobac"] = {
            # "feature_id": nexrad_features,
            # "UDAF_features": feature_id_to_UDAF(nexrad_features, "tobac"),
            # "linking": nexrad_tracks,
            "UDAF_linking": UDAF_tracks,
            # "segmentation_2d": nexrad_segmentation2d,
            "UDAF_segmentation_2d": UDAF_segmentation_2d,
            # "segmentation_3d": nexrad_segmentation3d,
            "UDAF_segmentation_3d": UDAF_segmentation_3d,
            "analysis": nexrad_tobac_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====NEXRAD tobac Tracking Complete=====")

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_multi_nexrad(CONFIG, queue=None):
    """
    Inputs:
        CONFIG: User configuration file
    Outputs:
        user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
    """

    from .tracker_output_translation_layer import (
        feature_id_to_UDAF,
        linking_to_UDAF,
        segmentation_to_UDAF,
    )
    from CoMET.analysis.analysis_object import Analysis_Object
    from .multi_nexrad_load import multi_nexrad_load_netcdf_iris

    if CONFIG["verbose"]:
        print("=====Loading Multi-NEXRAD Data=====")

    # determine if gridding is necessary or not
    if "gridding" in CONFIG["multi_nexrad"]:

        if CONFIG["verbose"]:
            print("=====Gridding Multi-NEXRAD Data=====")
        multi_nexrad_tracking_cube, multi_nexrad_tracking_xarray = (
            multi_nexrad_load_netcdf_iris(
                CONFIG["multi_nexrad"]["path_to_data"],
                "ar2v",
                CONFIG["multi_nexrad"]["feature_tracking_var"],
                CONFIG,
                CONFIG["multi_nexrad"]["gridding"]["gridding_save_path"],
            )
        )

    else:
        multi_nexrad_tracking_cube, multi_nexrad_tracking_xarray = (
            multi_nexrad_load_netcdf_iris(
                CONFIG["multi_nexrad"]["path_to_data"],
                "nc",
                CONFIG["multi_nexrad"]["feature_tracking_var"],
                CONFIG,
            )
        )

    # Add xarrays and cubes to return dict
    user_return_dict = {}

    user_return_dict["multi_nexrad"] = {
        # "tracking_xarray": multi_nexrad_tracking_xarray,
        # "tracking_cube": multi_nexrad_tracking_cube,
        # "segmentation_xarray": multi_nexrad_tracking_xarray,
        # "segmentation_cube": multi_nexrad_tracking_cube,
    }

    # determine which tracker to use
    if "tobac" in CONFIG["multi_nexrad"]:
        from .multi_nexrad_tobac import (
            multi_nexrad_tobac_feature_id,
            multi_nexrad_tobac_linking,
            multi_nexrad_tobac_segmentation,
        )

        multi_nexrad_features = None
        multi_nexrad_tracks = None
        multi_nexrad_segmentation2d = None
        multi_nexrad_segmentation3d = None
        multi_nexrad_tobac_analysis_data = {}

        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if "feature_id" in CONFIG["multi_nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting Multi-NEXRAD tobac Feature ID=====")

            multi_nexrad_features = multi_nexrad_tobac_feature_id(
                multi_nexrad_tracking_cube, CONFIG
            )

        if "linking" in CONFIG["multi_nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting Multi-NEXRAD tobac Feature Linking=====")

            multi_nexrad_tracks = multi_nexrad_tobac_linking(
                multi_nexrad_tracking_cube, multi_nexrad_features, CONFIG
            )

        if "segmentation_2d" in CONFIG["multi_nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting Multi-NEXRAD tobac 2D Segmentation=====")

            multi_nexrad_segmentation2d = multi_nexrad_tobac_segmentation(
                multi_nexrad_tracking_cube,
                multi_nexrad_features,
                "2d",
                CONFIG,
                CONFIG["multi_nexrad"]["tobac"]["segmentation_2d"]["height"],
            )

        if "segmentation_3d" in CONFIG["multi_nexrad"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting Multi-NEXRAD tobac 3D Segmentation=====")

            multi_nexrad_segmentation3d = multi_nexrad_tobac_segmentation(
                multi_nexrad_tracking_cube, multi_nexrad_features, "3d", CONFIG
            )

        # Create analysis object values
        UDAF_features = feature_id_to_UDAF(multi_nexrad_features, "tobac")
        UDAF_tracks = linking_to_UDAF(multi_nexrad_tracks, "tobac")
        UDAF_segmentation_2d = segmentation_to_UDAF(
            multi_nexrad_segmentation2d[0], UDAF_tracks, "tobac"
        )
        UDAF_segmentation_3d = segmentation_to_UDAF(
            multi_nexrad_segmentation3d[0], UDAF_tracks, "tobac"
        )

        # Create analysis object
        analysis_object = Analysis_Object(
            multi_nexrad_tracking_xarray,
            multi_nexrad_tracking_xarray,
            UDAF_features,
            UDAF_tracks,
            UDAF_segmentation_2d,
            UDAF_segmentation_3d,
        )

        if "analysis" in CONFIG["multi_nexrad"]["tobac"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["multi_nexrad"]["tobac"]["analysis"] is None:
                CONFIG["multi_nexrad"]["tobac"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting Multi-NEXRAD tobac Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["multi_nexrad"]["tobac"]["analysis"].keys():

                # Add default tracking featured_id variable in place of variable if not present
                if (
                    "variable"
                    not in CONFIG["multi_nexrad"]["tobac"]["analysis"][var.lower()]
                ):
                    CONFIG["multi_nexrad"]["tobac"]["analysis"][var.lower()][
                        "variable"
                    ] = CONFIG["multi_nexrad"]["feature_tracking_var"].upper()

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                multi_nexrad_tobac_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["multi_nexrad"]["tobac"]["analysis"][var.lower()],
                )

        if CONFIG["verbose"]:
            print("=====Converting Multi-NEXRAD tobac Output to CoMET-UDAF=====")

        # Add all products to return dict
        user_return_dict["multi_nexrad"]["tobac"] = {
            # "feature_id": multi_nexrad_features,
            # "UDAF_features": feature_id_to_UDAF(multi_nexrad_features, "tobac"),
            # "linking": multi_nexrad_tracks,
            "UDAF_linking": UDAF_tracks,
            # "segmentation_2d": multi_nexrad_segmentation2d,
            "UDAF_segmentation_2d": UDAF_segmentation_2d,
            # "segmentation_3d": multi_nexrad_segmentation3d,
            "UDAF_segmentation_3d": UDAF_segmentation_3d,
            "analysis": multi_nexrad_tobac_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====Multi-NEXRAD tobac Tracking Complete=====")

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_standard_radar(CONFIG, queue=None):
    """
    Inputs:
        CONFIG: User configuration file
    Outputs:
        user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
    """

    from .tracker_output_translation_layer import (
        feature_id_to_UDAF,
        linking_to_UDAF,
        segmentation_to_UDAF,
    )
    from CoMET.analysis.analysis_object import Analysis_Object
    from .standard_radar_load import standard_radar_load_netcdf_iris

    if CONFIG["verbose"]:
        print("=====Loading RADAR Data=====")

    radar_tracking_cube, radar_tracking_xarray = standard_radar_load_netcdf_iris(
        CONFIG["standard_radar"]["path_to_data"],
        CONFIG["standard_radar"]["feature_tracking_var"],
        CONFIG,
    )

    # Add xarrays and cubes to return dict
    user_return_dict = {}

    user_return_dict["standard_radar"] = {
        # "tracking_xarray": radar_tracking_xarray,
        # "tracking_cube": radar_tracking_cube,
        # "segmentation_xarray": radar_tracking_xarray,
        # "segmentation_cube": radar_tracking_cube,
    }

    # determine which tracker to use
    if "tobac" in CONFIG["standard_radar"]:
        from .standard_radar_tobac import (
            standard_radar_tobac_feature_id,
            standard_radar_tobac_linking,
            standard_radar_tobac_segmentation,
        )

        radar_features = None
        radar_tracks = None
        radar_segmentation2d = (None, None)
        radar_segmentation3d = (None, None)
        radar_tobac_analysis_data = {}

        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if "feature_id" in CONFIG["standard_radar"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting RADAR tobac Feature ID=====")

            radar_features = standard_radar_tobac_feature_id(
                radar_tracking_cube, CONFIG
            )

        if "linking" in CONFIG["standard_radar"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting RADAR tobac Feature Linking=====")

            radar_tracks = standard_radar_tobac_linking(
                radar_tracking_cube, radar_features, CONFIG
            )

        if "segmentation_2d" in CONFIG["standard_radar"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting RADAR tobac 2D Segmentation=====")

            height = (
                CONFIG["standard_radar"]["tobac"]["segmentation_2d"]["height"]
                if "height" in CONFIG["standard_radar"]["tobac"]["segmentation_2d"]
                else None
            )

            radar_segmentation2d = standard_radar_tobac_segmentation(
                radar_tracking_cube, radar_features, "2d", CONFIG, height
            )

        if "segmentation_3d" in CONFIG["standard_radar"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting RADAR tobac 3D Segmentation=====")

            radar_segmentation3d = standard_radar_tobac_segmentation(
                radar_tracking_cube, radar_features, "3d", CONFIG
            )

        # Create analysis object values
        UDAF_features = feature_id_to_UDAF(radar_features, "tobac")
        UDAF_tracks = linking_to_UDAF(radar_tracks, "tobac")
        UDAF_segmentation_2d = segmentation_to_UDAF(
            radar_segmentation2d[0], UDAF_tracks, "tobac"
        )
        UDAF_segmentation_3d = segmentation_to_UDAF(
            radar_segmentation3d[0], UDAF_tracks, "tobac"
        )

        # Create analysis object
        analysis_object = Analysis_Object(
            radar_tracking_xarray,
            radar_tracking_xarray,
            UDAF_features,
            UDAF_tracks,
            UDAF_segmentation_2d,
            UDAF_segmentation_3d,
        )

        if "analysis" in CONFIG["standard_radar"]["tobac"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["standard_radar"]["tobac"]["analysis"] is None:
                CONFIG["standard_radar"]["tobac"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting RADAR tobac Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["standard_radar"]["tobac"]["analysis"].keys():

                # Add default tracking featured_id variable in place of variable if not present
                if (
                    "variable"
                    not in CONFIG["standard_radar"]["tobac"]["analysis"][var.lower()]
                ):
                    CONFIG["standard_radar"]["tobac"]["analysis"][var.lower()][
                        "variable"
                    ] = CONFIG["standard_radar"]["feature_tracking_var"].upper()

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                radar_tobac_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["standard_radar"]["tobac"]["analysis"][var.lower()],
                )

        if CONFIG["verbose"]:
            print("=====Converting RADAR tobac Output to CoMET-UDAF=====")

        # Add all products to return dict
        user_return_dict["standard_radar"]["tobac"] = {
            # "feature_id": radar_features,
            # "UDAF_features": feature_id_to_UDAF(radar_features, "tobac"),
            # "linking": radar_tracks,
            "UDAF_linking": UDAF_tracks,
            # "segmentation_2d": radar_segmentation2d,
            "UDAF_segmentation_2d": UDAF_segmentation_2d,
            # "segmentation_3d": radar_segmentation3d,
            "UDAF_segmentation_3d": UDAF_segmentation_3d,
            "analysis": radar_tobac_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====RADAR tobac Tracking Complete=====")

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_goes(CONFIG, queue=None):
    """
    Inputs:
        CONFIG: User configuration file
    Outputs:
        user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
    """

    from .tracker_output_translation_layer import (
        feature_id_to_UDAF,
        linking_to_UDAF,
        segmentation_to_UDAF,
    )
    from CoMET.analysis.analysis_object import Analysis_Object
    from .goes_load import goes_load_netcdf_iris

    if CONFIG["verbose"]:
        print("=====Loading GOES Data=====")

    goes_tracking_cube, goes_tracking_xarray = goes_load_netcdf_iris(
        CONFIG["goes"]["path_to_data"], CONFIG["goes"]["feature_tracking_var"], CONFIG
    )

    # Add xarrays and cubes to return dict
    user_return_dict = {}

    user_return_dict["goes"] = {
        # "tracking_xarray": goes_tracking_xarray,
        # "tracking_cube": goes_tracking_cube,
        # "segmentation_xarray": goes_tracking_xarray,
        # "segmentation_cube": goes_tracking_cube,
    }

    # determine which tracker to use
    if "tobac" in CONFIG["goes"]:
        from .goes_tobac import (
            goes_tobac_feature_id,
            goes_tobac_linking,
            goes_tobac_segmentation,
        )

        goes_features = None
        goes_tracks = None
        goes_segmentation2d = None
        goes_tobac_analysis_data = {}

        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if "feature_id" in CONFIG["goes"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting GOES tobac Feature ID=====")

            goes_features = goes_tobac_feature_id(goes_tracking_cube, CONFIG)

        if "linking" in CONFIG["goes"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting GOES tobac Feature Linking=====")

            goes_tracks = goes_tobac_linking(goes_tracking_cube, goes_features, CONFIG)

        if "segmentation_2d" in CONFIG["goes"]["tobac"]:

            if CONFIG["verbose"]:
                print("=====Starting GOES tobac 2D Segmentation=====")

            # TB is 2D from satellite so no height parameter necessary
            goes_segmentation2d = goes_tobac_segmentation(
                goes_tracking_cube, goes_features, CONFIG
            )

        # Create analysis object values
        UDAF_features = feature_id_to_UDAF(goes_features, "tobac")
        UDAF_tracks = linking_to_UDAF(goes_tracks, "tobac")
        UDAF_segmentation_2d = segmentation_to_UDAF(
            goes_segmentation2d[0], UDAF_tracks, "tobac"
        )
        UDAF_segmentation_3d = None

        # Create analysis object
        analysis_object = Analysis_Object(
            goes_tracking_xarray,
            goes_tracking_xarray,
            UDAF_features,
            UDAF_tracks,
            UDAF_segmentation_2d,
            UDAF_segmentation_3d,
        )

        if "analysis" in CONFIG["goes"]["tobac"]:
            from CoMET.analysis.get_vars import get_var

            if CONFIG["goes"]["tobac"]["analysis"] is None:
                CONFIG["goes"]["tobac"]["analysis"] = {}

            if CONFIG["verbose"]:
                print("=====Starting GOES tobac Analysis Calculations=====")

            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG["goes"]["tobac"]["analysis"].keys():

                # Add default tracking featured_id variable in place of variable if not present
                if "variable" not in CONFIG["goes"]["tobac"]["analysis"][var.lower()]:
                    CONFIG["goes"]["tobac"]["analysis"][var.lower()]["variable"] = (
                        CONFIG["goes"]["feature_tracking_var"].upper()
                    )

                # This allows us to have multiple copies of the same variable by adjoining a dash
                proper_var_name = var.lower().split("-")[0]

                goes_tobac_analysis_data[var.lower()] = get_var(
                    analysis_object,
                    proper_var_name,
                    CONFIG["verbose"],
                    **CONFIG["goes"]["tobac"]["analysis"][var.lower()],
                )

        if CONFIG["verbose"]:
            print("=====Converting GOES tobac Output to CoMET-UDAF=====")

        # Add all products to return dict
        user_return_dict["goes"]["tobac"] = {
            # "feature_id": goes_features,
            # "UDAF_features": feature_id_to_UDAF(goes_features, "tobac"),
            # "linking": goes_tracks,
            "UDAF_linking": UDAF_tracks,
            # "segmentation_2d": goes_segmentation2d,
            "UDAF_segmentation_2d": UDAF_segmentation_2d,
            "analysis": goes_tobac_analysis_data,
            "analysis_object": analysis_object,
        }

        if CONFIG["verbose"]:
            print("=====GOES tobac Tracking Complete=====")

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict
