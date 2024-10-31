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

import ast
import multiprocessing
import multiprocessing.queues
import os
import time

import numpy as np
import yaml

from CoMET.analysis.analysis_object import Analysis_Object
from CoMET.analysis.calc_var import calc_var

from .goes_load import goes_load_netcdf_iris
from .goes_tobac import (
    goes_tobac_feature_id,
    goes_tobac_linking,
    goes_tobac_segmentation,
)
from .mesonh_load import mesonh_load_netcdf_iris
from .mesonh_moaap import mesonh_run_moaap
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
from .standard_radar_load import standard_radar_load_netcdf_iris
from .standard_radar_tobac import (
    standard_radar_tobac_feature_id,
    standard_radar_tobac_linking,
    standard_radar_tobac_segmentation,
)
from .tracker_output_translation_layer import (
    bulk_moaap_to_UDAF,
    feature_id_to_UDAF,
    linking_to_UDAF,
    segmentation_to_UDAF,
)
from .wrf_load import wrf_load_netcdf_iris
from .wrf_moaap import wrf_run_moaap
from .wrf_tobac import wrf_tobac_feature_id, wrf_tobac_linking, wrf_tobac_segmentation

from .run_tracker_wrapper import run_tracker

__all__ = [
    "CoMET_start",
    "CoMET_load",
    "run_goes",
    "run_mesonh",
    "run_multi_nexrad",
    "run_nexrad",
    "run_standard_radar",
    "run_wrf",
]


def CoMET_start(
        path_to_config: str | None = None, 
        CONFIG: dict | None = None,
) -> dict:
    """


    Parameters
    ----------
    path_to_config : str | None, optional
        Path to a config.yml file containing all details of the CoMET run. See boilerplate.yml for how the file should be setup. The default is None.
    CONFIG : dict | None, optional
        Optional to just pass a config dict object instead of filepath. The default is None.

    Returns
    -------
    dict
        Default CoMET output following CoMET-UDAF specification.

    """

    # Load CONFIG if not present
    if CONFIG is None:
        CONFIG = CoMET_load(path_to_config)

    # If parallelization is True, run the multiprocessing version instead
    if CONFIG["parallel_processing"]:

        if CONFIG["max_cores"] is not None:
            # TODO: I have no idea if this actually works hah, need to test
            os.environ["OMP_NUM_THREADS"] = str(
                CONFIG["max_cores"] * 2
            )  # Take advantage of hyper threading

        # Return CoMET multi processes output which should be a dictionary
        multi_output = CoMET_start_multi(CONFIG)

        # Reset environmental lock when done
        os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count() * 2)

        return multi_output

    start_time = time.perf_counter()

    # Create empty dictionaries for each data type
    wrf_data = mesonh_data = nexrad_data = multi_nexrad_data = standard_radar_data = rams_data = (
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

    # Handle RAMS data
    if "rams" in CONFIG:

        # Call run goes function to handle all goes tasks
        rams_data = run_rams(CONFIG)

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
        | rams_data
        | nexrad_data
        | multi_nexrad_data
        | standard_radar_data
        | goes_data
    )


def CoMET_start_multi(CONFIG: dict) -> dict:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.

    Returns
    -------
    dict
        Dictionary designed according to the CoMET-UDAF standard.

    """

    # This is necessary for python reasons I suppose may need to check this after some kind of release
    if __name__ == "CoMET.user_interface_layer":

        start_time = time.perf_counter()

        # Start a queue so processes can finish at different times
        queue = multiprocessing.Queue()

        processes = []
        responses = []

        # if wrf is present in CONFIG, run the necessary wrf functions
        if "wrf" in CONFIG:

            # Call run wrf function to handle all wrf tasks
            wrf_process = multiprocessing.Process(target=run_wrf, args=(CONFIG, queue))
            processes.append(wrf_process)
            wrf_process.start()

        # Handle MesoNH data
        if "mesonh" in CONFIG:

            # Call run MesoNH function to handle all MesoNH tasks
            mesonh_process = multiprocessing.Process(
                target=run_mesonh, args=(CONFIG, queue)
            )
            processes.append(mesonh_process)
            mesonh_process.start()

        # Handle RAMS data
        if "rams" in CONFIG:

            # Call run MesoNH function to handle all MesoNH tasks
            rams_process = multiprocessing.Process(
                target=run_rams, args=(CONFIG, queue)
            )
            processes.append(rams_process)
            rams_process.start()

        # Handle GOES data
        if "goes" in CONFIG:

            # Call run goes function to handle all goes tasks
            goes_process = multiprocessing.Process(
                target=run_goes, args=(CONFIG, queue)
            )
            processes.append(goes_process)
            goes_process.start()

        # Handle NEXRAD data
        if "nexrad" in CONFIG:

            # Call run nexrad function to handle all nexrad tasks
            nexrad_process = multiprocessing.Process(
                target=run_nexrad, args=(CONFIG, queue)
            )
            processes.append(nexrad_process)
            nexrad_process.start()

        # Handle Multi NEXRAD data
        if "multi_nexrad" in CONFIG:

            # Call run nexrad function to handle all nexrad tasks
            multi_nexrad_process = multiprocessing.Process(
                target=run_multi_nexrad, args=(CONFIG, queue)
            )
            processes.append(multi_nexrad_process)
            multi_nexrad_process.start()

        # Handle standard radar data
        if "standard_radar" in CONFIG:

            # Call run goes function to handle all goes tasks
            radar_process = multiprocessing.Process(
                target=run_standard_radar, args=(CONFIG, queue)
            )
            processes.append(radar_process)
            radar_process.start()

        for p in processes:
            responses.append(queue.get())

        for p in processes:
            p.join()

        return_dict = {}

        for ii in range(len(responses)):
            return_dict = return_dict | responses[ii]

        end_time = time.perf_counter()

        if CONFIG["verbose"]:
            print(
                f"""=====CoMET Performance Diagonistics=====\n$ Total Process Time: {"%.2f Seconds" % (end_time-start_time)}\n$ Allocated Resources: Cores = {CONFIG["max_cores"]}, Threads = {CONFIG["max_cores"] * 2}"""
            )

        # Return dict at end
        return return_dict


def CoMET_load(
    path_to_config: str | None = None, CONFIG_string: str | None = None
) -> dict:
    """


    Parameters
    ----------
    path_to_config : str | None, optional
        Path to a config.yml file containing all details of the CoMET run. See boilerplate.yml for how the file should be setup. The default is None.
    CONFIG_string : str | None, optional
        String of yaml data if not using a file. The default is None.

    Returns
    -------
    CONFIG : dict
        Dictionary object containing all user-defined parameters.

    """

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
    if "wrf" in CONFIG:

        if CONFIG["verbose"]:
            print("=====WRF Setup Found in CONFIG=====")

        # Auto capitalize if one of CoMET's computed variables
        CONFIG["wrf"]["feature_tracking_var"] = (
            CONFIG["wrf"]["feature_tracking_var"].upper()
            if CONFIG["wrf"]["feature_tracking_var"].lower()
            in ["dbz", "wa", "tb", "wr"]
            else CONFIG["wrf"]["feature_tracking_var"]
        )
        CONFIG["wrf"]["segmentation_var"] = (
            CONFIG["wrf"]["segmentation_var"].upper()
            if CONFIG["wrf"]["segmentation_var"].lower() in ["dbz", "wa", "tb", "wr"]
            else CONFIG["wrf"]["segmentation_var"]
        )

    if "mesonh" in CONFIG:

        if CONFIG["verbose"]:
            print("=====MesoNH Setup Found in CONFIG=====")

        # Auto capitalize if one of CoMET's computed variables
        CONFIG["mesonh"]["feature_tracking_var"] = (
            CONFIG["mesonh"]["feature_tracking_var"].upper()
            if CONFIG["mesonh"]["feature_tracking_var"].lower()
            in ["dbz", "wa", "tb", "wr"]
            else CONFIG["mesonh"]["feature_tracking_var"]
        )
        CONFIG["mesonh"]["segmentation_var"] = (
            CONFIG["mesonh"]["segmentation_var"].upper()
            if CONFIG["mesonh"]["segmentation_var"].lower() in ["dbz", "wa", "tb", "wr"]
            else CONFIG["mesonh"]["segmentation_var"]
        )

    if "rams" in CONFIG:

        if CONFIG["verbose"]:
            print("=====RAMS Setup Found in CONFIG=====")

        # Auto capitalize if one of CoMET's computed variables
        CONFIG["rams"]["feature_tracking_var"] = (
            CONFIG["rams"]["feature_tracking_var"].upper()
            if CONFIG["rams"]["feature_tracking_var"].lower()
            in ["dbz", "wa", "tb", "wr"]
            else CONFIG["rams"]["feature_tracking_var"]
        )
        CONFIG["rams"]["segmentation_var"] = (
            CONFIG["rams"]["segmentation_var"].upper()
            if CONFIG["rams"]["segmentation_var"].lower() in ["dbz", "wa", "tb", "wr"]
            else CONFIG["rams"]["segmentation_var"]
        )

    # if nexrad present, check for tuples
    if "nexrad" in CONFIG:

        if CONFIG["verbose"]:
            print("=====NEXRAD Setup Found in CONFIG=====")

        # Auto capitalize valid variables
        CONFIG["nexrad"]["feature_tracking_var"] = (
            CONFIG["nexrad"]["feature_tracking_var"].upper()
            if CONFIG["nexrad"]["feature_tracking_var"].lower() in ["dbz"]
            else CONFIG["nexrad"]["feature_tracking_var"]
        )
        CONFIG["nexrad"]["segmentation_var"] = (
            CONFIG["nexrad"]["segmentation_var"].upper()
            if CONFIG["nexrad"]["segmentation_var"].lower() in ["dbz"]
            else CONFIG["nexrad"]["segmentation_var"]
        )

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

        # Auto capitalize valid variables
        CONFIG["multi_nexrad"]["feature_tracking_var"] = (
            CONFIG["multi_nexrad"]["feature_tracking_var"].upper()
            if CONFIG["multi_nexrad"]["feature_tracking_var"].lower() in ["dbz"]
            else CONFIG["multi_nexrad"]["feature_tracking_var"]
        )
        CONFIG["multi_nexrad"]["segmentation_var"] = (
            CONFIG["multi_nexrad"]["segmentation_var"].upper()
            if CONFIG["multi_nexrad"]["segmentation_var"].lower() in ["dbz"]
            else CONFIG["multi_nexrad"]["segmentation_var"]
        )

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

    if "standard_radar" in CONFIG:

        if CONFIG["verbose"]:
            print("=====RADAR Setup Found in CONFIG=====")

        # Auto capitalize valid variables
        CONFIG["standard_radar"]["feature_tracking_var"] = (
            CONFIG["standard_radar"]["feature_tracking_var"].upper()
            if CONFIG["standard_radar"]["feature_tracking_var"].lower() in ["dbz"]
            else CONFIG["standard_radar"]["feature_tracking_var"]
        )
        CONFIG["standard_radar"]["segmentation_var"] = (
            CONFIG["standard_radar"]["segmentation_var"].upper()
            if CONFIG["standard_radar"]["segmentation_var"].lower() in ["dbz"]
            else CONFIG["standard_radar"]["segmentation_var"]
        )

    if "goes" in CONFIG:

        if CONFIG["verbose"]:
            print("=====GOES Setup Found in CONFIG=====")

        # Auto capitalize valid variables
        CONFIG["goes"]["feature_tracking_var"] = (
            CONFIG["goes"]["feature_tracking_var"].upper()
            if CONFIG["goes"]["feature_tracking_var"].lower() in ["tb"]
            else CONFIG["goes"]["feature_tracking_var"]
        )
        CONFIG["goes"]["segmentation_var"] = (
            CONFIG["goes"]["segmentation_var"].upper()
            if CONFIG["goes"]["segmentation_var"].lower() in ["tb"]
            else CONFIG["goes"]["segmentation_var"]
        )

    return CONFIG


# =============================================================================
# This section is for running individual data types
# =============================================================================

# TODO: loop through all of the datasets in CONFIG file and pass through the run_tracker function
def run_rams(
        CONFIG : dict,
        queue : multiprocessing.queues.Queue | None = None
) -> dict | None:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.
    queue : multiprocessing.queues.Queue | None, optional
        Multiprocessing queue to pass the output dict to for parallelization. The default is None.

    Returns
    -------
    user_return_dict : dict
        A dictionary object which contanis all tobac and CoMET-UDAF standard outputs.

    """

    if CONFIG["verbose"]:
        print("=====Loading RAMS Data=====")

    # Create a dictionary with all of the tracked information
    user_return_dict = {}
    tracking_info = None

    # Determine which tracker(s) are in the CONFIG file
    if 'tobac' in CONFIG['rams']:
        user_return_dict, tracking_info = run_tracker('rams', 'tobac', user_return_dict, tracking_info, CONFIG)
    
    if 'moaap' in CONFIG['rams']:
        user_return_dict, tracking_info = run_tracker('rams', 'moaap', user_return_dict, tracking_info, CONFIG)

    if 'tams' in CONFIG['rams']:
        user_return_dict, tracking_info = run_tracker('rams', 'tams', user_return_dict, tracking_info, CONFIG)

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict

def run_wrf(
    CONFIG: dict, 
    queue: multiprocessing.queues.Queue | None = None,
) -> dict | None:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.
    queue : multiprocessing.queues.Queue | None, optional
        Multiprocessing queue to pass the output dict to for parallelization. The default is None.

    Returns
    -------
    user_return_dict : dict
        A dictionary object which contanis all tobac and CoMET-UDAF standard outputs.

    """

    if CONFIG["verbose"]:
        print("=====Loading WRF Data=====")

    # Create a dictionary with all of the tracked information
    user_return_dict = {}
    tracking_info = None


    # Determine which tracker(s) are in the CONFIG file
    if 'tobac' in CONFIG['wrf']:
        user_return_dict, tracking_info = run_tracker('wrf', 'tobac', user_return_dict, tracking_info, CONFIG)
    
    if 'moaap' in CONFIG['wrf']:
        user_return_dict, tracking_info = run_tracker('wrf', 'moaap', user_return_dict, tracking_info, CONFIG)
    
    if 'tams' in CONFIG['wrf']:
        user_return_dict, tracking_info = run_tracker('wrf', 'tams', user_return_dict, tracking_info, CONFIG)

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_mesonh(
    CONFIG: dict, 
    queue: multiprocessing.queues.Queue | None = None,
) -> dict | None:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.
    queue : multiprocessing.queues.Queue | None, optional
        Multiprocessing queue to pass the output dict to for parallelization. The default is None.

    Returns
    -------
    user_return_dict : dict
        A dictionary object which contanis all tobac and CoMET-UDAF standard outputs.

    """

    if CONFIG["verbose"]:
        print("=====Loading MesoNH Data=====")

    # Create a dictionary with all of the tracked information   
    user_return_dict = {}
    tracking_info = None

    # Determine which tracker(s) are in the CONFIG file
    if 'tobac' in CONFIG['mesonh']:
        user_return_dict, tracking_info = run_tracker('mesonh', 'tobac', user_return_dict, tracking_info, CONFIG)
    
    if 'moaap' in CONFIG['mesonh']:
        user_return_dict, tracking_info = run_tracker('mesonh', 'moaap', user_return_dict, tracking_info, CONFIG)

    if 'tams' in CONFIG['mesonh']:
        user_return_dict, tracking_info = run_tracker('mesonh', 'tams', user_return_dict, tracking_info, CONFIG)

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_nexrad(
    CONFIG: dict, 
    queue: multiprocessing.queues.Queue | None = None,
) -> dict | None:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.
    queue : multiprocessing.queues.Queue | None, optional
        Multiprocessing queue to pass the output dict to for parallelization. The default is None.

    Returns
    -------
    user_return_dict : dict
        A dictionary object which contanis all tobac and CoMET-UDAF standard outputs.

    """

    if CONFIG["verbose"]:
        print("=====Loading NEXRAD Data=====")

    # Create a dictionary with all of the tracked information   
    user_return_dict = {}
    tracking_info = None

    # Determine which tracker(s) are in the CONFIG file
    if 'tobac' in CONFIG['nexrad']:
        user_return_dict, tracking_info = run_tracker('nexrad', 'tobac', user_return_dict, tracking_info, CONFIG)

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_multi_nexrad(
    CONFIG: dict, 
    queue: multiprocessing.queues.Queue | None = None,
) -> dict | None:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.
    queue : multiprocessing.queues.Queue | None, optional
        Multiprocessing queue to pass the output dict to for parallelization. The default is None.

    Returns
    -------
    user_return_dict : dict
        A dictionary object which contanis all tobac and CoMET-UDAF standard outputs.

    """

    if CONFIG["verbose"]:
        print("=====Loading MULTI-NEXRAD Data=====")

    # Create a dictionary with all of the tracked information   
    user_return_dict = {}
    tracking_info = None

    # Determine which tracker(s) are in the CONFIG file
    if 'tobac' in CONFIG['multi-nexrad']:
        user_return_dict = run_tracker('multi-nexrad', 'tobac', user_return_dict, tracking_info, CONFIG)

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_standard_radar(
    CONFIG: dict, queue: multiprocessing.queues.Queue | None = None
) -> dict | None:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.
    queue : multiprocessing.queues.Queue | None, optional
        Multiprocessing queue to pass the output dict to for parallelization. The default is None.

    Returns
    -------
    user_return_dict : dict
        A dictionary object which contanis all tobac and CoMET-UDAF standard outputs.

    """

    if CONFIG["verbose"]:
        print("=====Loading STANDARD_RADAR Data=====")

    # Create a dictionary with all of the tracked information   
    user_return_dict = {}
    tracking_info = None

    # Determine which tracker(s) are in the CONFIG file
    if 'tobac' in CONFIG['standard_radar']:
        user_return_dict, tracking_info = run_tracker('standard_radar', 'tobac', user_return_dict, tracking_info, CONFIG)

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict


def run_goes(
    CONFIG: dict, 
    queue: multiprocessing.queues.Queue | None = None,
) -> dict | None:
    """


    Parameters
    ----------
    CONFIG : dict
        User configuration file.
    queue : multiprocessing.queues.Queue | None, optional
        Multiprocessing queue to pass the output dict to for parallelization. The default is None.

    Returns
    -------
    user_return_dict : dict
        A dictionary object which contanis all tobac and CoMET-UDAF standard outputs.

    """

    if CONFIG["verbose"]:
        print("=====Loading GOES Data=====")

    # Create a dictionary with all of the tracked information   
    user_return_dict = {}
    tracking_info = None

    # Determine which tracker(s) are in the CONFIG file
    if 'tobac' in CONFIG['goes']:
        user_return_dict, tracking_info = run_tracker('goes', 'tobac', user_return_dict, tracking_info, CONFIG)

    # Send return dict to queue if there is a queue object passed
    if queue is not None:
        queue.put(user_return_dict)
        return None

    # Return dictionary
    return user_return_dict
