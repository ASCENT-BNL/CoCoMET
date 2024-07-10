#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:55:16 2024

@author: thahn
"""

# =============================================================================
# This is the interface layer between all of CoMET's backend functionality and the user. A sort of parser for a configuration input file. 
# =============================================================================
# TODO: Create CoMET-UDAF Specification for return object



"""
Inputs:
    path_to_config: Path to a config.yml file containing all details of the CoMET run. See boilerplate.yml for how the file should be setup
    manual_mode: [True, False] Whether CoMET should run all functions idenpendetly or should just return CONFIG for user to run functions independently 
    CONFIG: Optional to just pass a config dict object instead of filepath
"""
def CoMET_start(path_to_config, manual_mode=False, CONFIG=None):
    
    # Load CONFIG if not present
    if (CONFIG is None):
        CONFIG = CoMET_load(path_to_config)
    
    # if manual_mode = True, just return the loaded CONFIG, otherwise, go through defined setups and run them
    if (manual_mode): return(CONFIG)
    
    
    # If parallelization is True, run the multiprocessing version instead
    if (CONFIG["parallel_processing"]):
        
        # Return CoMET multi processes output which should be a dictionary
        return (CoMET_start_multi(CONFIG))
    
    
    # Create empty dictionaries for each data type
    wrf_data = nexrad_data = goes_data = {}
    
    # if wrf is present in CONFIG, run the necessary wrf functions
    if ("wrf" in CONFIG):
        
        # Call run wrf function to handle all wrf tasks
        wrf_data = run_wrf(CONFIG)


    # Handle MesoNH data
    if ("mesonh" in CONFIG):
        
        # Call run mesonh function to handle all mesonh tasks
        mesonh_data = run_mesonh(CONFIG)

    # Handle NEXRAD data
    if ("nexrad" in CONFIG):
        
        # Call run nexrad function to handle all nexrad tasks
        nexrad_data = run_nexrad(CONFIG)
        
    
    # Handle GOES data
    if ("goes" in CONFIG):
        
        # Call run goes function to handle all goes tasks
        goes_data = run_goes(CONFIG)


    # Return dict at end
    return (wrf_data | mesonh_data | nexrad_data | goes_data)



"""
Inputs:
    CONFIG: User defined configuration data
Outputs:
    return_dict: Dictionary designed according to the CoMET-UDAF standard
"""
def CoMET_start_multi(CONFIG):
    import multiprocessing
    
    # This is necessary for python reasons I suppose may need to check this after some kind of release
    if __name__ == 'CoMET.user_interface_layer':

        # Keep track of active core count. Should start at one.
        active_process_count = 1
        
        # Keep track of for when doing gridding
        inital_max_cores = CONFIG['max_cores']
        
        # Start a queue so processes can finish at different times
        queue = multiprocessing.Queue()
        
        processes = []
        responses = []
        
        # if wrf is present in CONFIG, run the necessary wrf functions
        if ("wrf" in CONFIG):
            
            # Check to make sure max core count has not been exceeded
            if (CONFIG['max_cores'] is not None and active_process_count + 1 > CONFIG['max_cores']):
                raise Exception("!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!")
                return
            
            # Call run wrf function to handle all wrf tasks
            wrf_process = multiprocessing.Process(target=run_wrf, args=(CONFIG, queue))
            processes.append(wrf_process)
            wrf_process.start()
            
            active_process_count+=1
        
        
        # Handle MesoNH data
        if ("mesonh" in CONFIG):
            
            # Check to make sure max core count has not been exceeded
            if (CONFIG['max_cores'] is not None and active_process_count + 1 > CONFIG['max_cores']):
                raise Exception("!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!")
                return
            
            # Call run MesoNH function to handle all MesoNH tasks
            mesonh_process = multiprocessing.Process(target=run_mesonh, args=(CONFIG, queue))
            processes.append(mesonh_process)
            mesonh_process.start()
            
            active_process_count+=1
            
            
        # Handle GOES data
        if ("goes" in CONFIG):
            
            # Check to make sure max core count has not been exceeded
            if (CONFIG['max_cores'] is not None and active_process_count + 1 > CONFIG['max_cores']):
                raise Exception("!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!")
                return
            
            # Call run goes function to handle all goes tasks
            goes_process = multiprocessing.Process(target=run_goes, args=(CONFIG, queue))
            processes.append(goes_process)
            goes_process.start()
            
            active_process_count+=1


        # Handle NEXRAD data
        if ('nexrad' in CONFIG):
            
            # Check to make sure max core count has not been exceeded
            if (CONFIG['max_cores'] is not None and active_process_count + 1 > CONFIG['max_cores']):
                raise Exception("!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!")
                return
            
            # If gridding is enabled, make sure doing so will not exceed the max number of cores and adjust the CONFIG values accordingly
            if (CONFIG['max_cores'] is not None and "gridding" in CONFIG['nexrad']):
                CONFIG['max_cores'] = CONFIG['max_cores'] - active_process_count - 1
            
                if (CONFIG['max_cores'] <= 0):
                    raise Exception("!=====Insufficent Number of Cores (max_cores should equal (# of input data types) + >=1 if gridding NEXRAD)=====!")
                    return
            
            # Call run nexrad function to handle all nexrad tasks
            nexrad_process = multiprocessing.Process(target=run_nexrad, args=(CONFIG, queue))
            processes.append(nexrad_process)
            nexrad_process.start()
            
            active_process_count+=1
            

        for p in processes:
            responses.append(queue.get())
        for p in processes:
            p.join()
            active_process_count-=1

        return_dict = {}
        
        for ii in range(len(responses)):
            return_dict = return_dict | responses[ii] 
    
        # Reset CONFIG max cores if necessary
        CONFIG['max_cores'] = inital_max_cores
    
        # Return dict at end
        return (return_dict)



"""
Inputs:
    path_to_config: Path to a config.yml file containing all details of the CoMET run. See boilerplate.yml for how the file should be setup
Outputs:
    CONFIG: dictionary object containing all user-defined parameters
"""
def CoMET_load(path_to_config):
    import ast
    import yaml
    import numpy as np
    
    # Open and read config file
    with open(path_to_config, 'r') as f:
        CONFIG = yaml.safe_load(f)
        
        
    # Go through each potential option and determine which functions need to run
    if ("wrf" in CONFIG and CONFIG['verbose']):
        print("=====WRF Setup Found in CONFIG=====")
    
    if ("mesonh" in CONFIG and CONFIG["verbose"]):
        print("=====MesoNH Setup Found in CONFIG=====")
    
    # if nexrad present, run nexrad 
    if ("nexrad" in CONFIG):
        if(CONFIG['verbose']): print("=====NEXRAD Setup Found in CONFIG=====")
        
        # If nexrad gridding is needed, change grid shapes and limits back to tuples
        if ("gridding" in CONFIG["nexrad"]):
        
            # Convert grid_shape and grid_limits back into proper tuples and ensure correct data types--auto correct automatically
            grid_shape = ast.literal_eval(CONFIG['nexrad']['gridding']['grid_shape'])
            # Ensure int grid shapes
            grid_shape = tuple([int(grid_shape[ii]) for ii in range(len(grid_shape))])
            
            # Ensure float for grid_limits
            grid_limits = ast.literal_eval(CONFIG['nexrad']['gridding']['grid_limits'])
            grid_limits = np.array(grid_limits).astype(float)
            grid_limits = tuple([tuple(row) for row in grid_limits])
    
            # Adjust CONFIG values
            CONFIG['nexrad']['gridding']['grid_shape'] = grid_shape
            CONFIG['nexrad']['gridding']['grid_limits'] = grid_limits
    
    
    if ("goes" in CONFIG and CONFIG['verbose']):
        print("=====GOES Setup Found in CONFIG=====")
    
    return(CONFIG)
        


# =============================================================================
# This section is for running individual data types
# =============================================================================
"""
Inputs:
    CONFIG: User configuration file
Outputs:
    user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
"""
def run_wrf(CONFIG, queue = None):
    from .tracker_output_translation_layer import feature_id_to_UDAF, linking_to_UDAF, segmentation_to_UDAF
    from .wrf_load import wrf_load_netcdf_iris
    
    if (CONFIG['verbose']): print("=====Loading WRF Data=====")
    
    wrf_tracking_cube, wrf_tracking_xarray = wrf_load_netcdf_iris(CONFIG['wrf']['path_to_data'], CONFIG['wrf']['feature_tracking_var'], CONFIG)
    
    # if tracking and segmentation variables are different, load seperately
    if (CONFIG['wrf']['feature_tracking_var'] != CONFIG['wrf']['segmentation_var']):
        
        wrf_segmentation_cube, wrf_segmentation_xarray = wrf_load_netcdf_iris(CONFIG['wrf']['path_to_data'], CONFIG['wrf']['segmentation_var'], CONFIG)
    
    else:
    
        wrf_segmentation_cube = wrf_tracking_cube
        wrf_segmentation_xarray = wrf_tracking_xarray
    
    # Add xarrays and cubes to return dict
    user_return_dict = {}
    
    user_return_dict["wrf"] = {
        "tracking_xarray": wrf_tracking_xarray,
        "tracking_cube": wrf_tracking_cube,
        "segmentation_xarray": wrf_segmentation_xarray,
        "segmentation_cube": wrf_segmentation_cube
    }
    
    # now determine which tracker to use
    if ("tobac" in CONFIG['wrf']):
        from .wrf_tobac import wrf_tobac_feature_id, wrf_tobac_linking, wrf_tobac_segmentation
        
        wrf_features = None
        wrf_tracks = None
        wrf_segmentation2d = None
        wrf_segmentation3d = None
        wrf_analysis_data = {}
        
        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if ("feature_id" in CONFIG['wrf']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting WRF tobac Feature ID=====")
            
            wrf_features = wrf_tobac_feature_id(wrf_tracking_cube, CONFIG)
        
        if ("linking" in CONFIG['wrf']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting WRF tobac Feature Linking=====")
            
            wrf_tracks = wrf_tobac_linking(wrf_tracking_cube, wrf_features, CONFIG)
        
        if ("segmentation_2d" in CONFIG['wrf']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting WRF tobac 2D Segmentation=====")
            
            wrf_segmentation2d = wrf_tobac_segmentation(wrf_segmentation_cube, wrf_features, '2d', CONFIG, CONFIG['wrf']['tobac']['segmentation_2d']['height'])
    
        if ("segmentation_3d" in CONFIG['wrf']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting WRF tobac 3D Segmentation=====")
            
            wrf_segmentation3d = wrf_tobac_segmentation(wrf_segmentation_cube, wrf_features, '3d', CONFIG)
    
        if ("analysis" in CONFIG['wrf']['tobac']):
            
            from CoMET.analysis.get_vars import get_var
            
            if (CONFIG['verbose']): print("=====Starting WRF tobac Analysis Calculations=====")
            
            # Create analysis object
            analysis_object = {
                "tracking_xarray": wrf_tracking_xarray,
                "segmentation_xarray": wrf_segmentation_xarray,
                "UDAF_features": feature_id_to_UDAF(wrf_features, "tobac"),
                "UDAF_linking": linking_to_UDAF(wrf_tracks, "tobac"),
                "UDAF_segmentation_2d": segmentation_to_UDAF(wrf_segmentation2d[0], linking_to_UDAF(wrf_tracks, "tobac"), "tobac"),
                "UDAF_segmentation_3d": segmentation_to_UDAF(wrf_segmentation3d[0], linking_to_UDAF(wrf_tracks, "tobac"), "tobac")
            }
            
            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG['wrf']['tobac']['analysis'].keys():
                
                # Add default tracking featured_id variable in place of variable if not present
                if ("variable" not in CONFIG['wrf']['tobac']['analysis'][var.lower()]): CONFIG['wrf']['tobac']['analysis'][var.lower()]["variable"] = CONFIG["wrf"]["feature_tracking_var"].upper()
                
                wrf_analysis_data[var.lower()] = (get_var(analysis_object, var, CONFIG['verbose'], **CONFIG['wrf']['tobac']['analysis'][var.lower()]))
                
    
        if (CONFIG['verbose']): print("=====Converting WRF tobac Output to CoMET-UDAF=====")
    
        # Add all products to return dict
        user_return_dict["wrf"]["tobac"] = {
            "feature_id": wrf_features,
            "UDAF_features": feature_id_to_UDAF(wrf_features, "tobac"),
            "linking": wrf_tracks,
            "UDAF_linking": linking_to_UDAF(wrf_tracks, "tobac"),
            "segmentation_2d": wrf_segmentation2d,
            "UDAF_segmentation_2d": segmentation_to_UDAF(wrf_segmentation2d[0], linking_to_UDAF(wrf_tracks, "tobac"), "tobac"),
            "segmentation_3d": wrf_segmentation3d,
            "UDAF_segmentation_3d": segmentation_to_UDAF(wrf_segmentation3d[0], linking_to_UDAF(wrf_tracks, "tobac"), "tobac"),
            "analysis": wrf_analysis_data
            }
        
        if (CONFIG['verbose']): print("=====WRF tobac Tracking Complete=====")
        
        # Send return dict to queue if there is a queue object passed
        if (queue is not None):
            queue.put(user_return_dict)
            return
        
        # Return dictionary
        return (user_return_dict)
    
    else:
        raise Exception("!=====No Tracker or Invalid Tracker Found in CONFIG=====!")
        return



"""
Inputs:
    CONFIG: User configuration file
Outputs:
    user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
"""
def run_mesonh(CONFIG, queue = None):
    from .tracker_output_translation_layer import feature_id_to_UDAF, linking_to_UDAF, segmentation_to_UDAF
    from .mesonh_load import mesonh_load_netcdf_iris
    
    if (CONFIG['verbose']): print("=====Loading MesoNH Data=====")
    
    mesonh_tracking_cube, mesonh_tracking_xarray = mesonh_load_netcdf_iris(CONFIG['mesonh']['path_to_data'], CONFIG['mesonh']['feature_tracking_var'], CONFIG)
    
    # if tracking and segmentation variables are different, load seperately
    if (CONFIG['mesonh']['feature_tracking_var'] != CONFIG['mesonh']['segmentation_var']):
        
        mesonh_segmentation_cube, mesonh_segmentation_xarray = mesonh_load_netcdf_iris(CONFIG['mesonh']['path_to_data'], CONFIG['mesonh']['segmentation_var'], CONFIG)
    
    else:
    
        mesonh_segmentation_cube = mesonh_tracking_cube
        mesonh_segmentation_xarray = mesonh_tracking_xarray
    
    # Add xarrays and cubes to return dict
    user_return_dict = {}
    
    user_return_dict["mesonh"] = {
        "tracking_xarray": mesonh_tracking_xarray,
        "tracking_cube": mesonh_tracking_cube,
        "segmentation_xarray": mesonh_segmentation_xarray,
        "segmentation_cube": mesonh_segmentation_cube
    }
    
    # now determine which tracker to use
    if ("tobac" in CONFIG['mesonh']):
        from .mesonh_tobac import mesonh_tobac_feature_id, mesonh_tobac_linking, mesonh_tobac_segmentation
        
        mesonh_features = None
        mesonh_tracks = None
        mesonh_segmentation2d = None
        mesonh_segmentation3d = None
        mesonh_analysis_data = {}
        
        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if ("feature_id" in CONFIG['mesonh']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting MesoNH tobac Feature ID=====")
            
            mesonh_features = mesonh_tobac_feature_id(mesonh_tracking_cube, CONFIG)
        
        if ("linking" in CONFIG['mesonh']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting MesoNH tobac Feature Linking=====")
            
            mesonh_tracks = mesonh_tobac_linking(mesonh_tracking_cube, mesonh_features, CONFIG)
        
        if ("segmentation_2d" in CONFIG['mesonh']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting MesoNH tobac 2D Segmentation=====")
            
            mesonh_segmentation2d = mesonh_tobac_segmentation(mesonh_segmentation_cube, mesonh_features, '2d', CONFIG, CONFIG['mesonh']['tobac']['segmentation_2d']['height'])
    
        if ("segmentation_3d" in CONFIG['mesonh']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting MesoNH tobac 3D Segmentation=====")
            
            mesonh_segmentation3d = mesonh_tobac_segmentation(mesonh_segmentation_cube, mesonh_features, '3d', CONFIG)
    
        if ("analysis" in CONFIG['mesonh']['tobac']):
            
            from CoMET.analysis.get_vars import get_var
            
            if (CONFIG['verbose']): print("=====Starting MesoNH tobac Analysis Calculations=====")
            
            # Create analysis object
            analysis_object = {
                "tracking_xarray": mesonh_tracking_xarray,
                "segmentation_xarray": mesonh_segmentation_xarray,
                "UDAF_features": feature_id_to_UDAF(mesonh_features, "tobac"),
                "UDAF_linking": linking_to_UDAF(mesonh_tracks, "tobac"),
                "UDAF_segmentation_2d": segmentation_to_UDAF(mesonh_segmentation2d[0], linking_to_UDAF(mesonh_tracks, "tobac"), "tobac"),
                "UDAF_segmentation_3d": segmentation_to_UDAF(mesonh_segmentation3d[0], linking_to_UDAF(mesonh_tracks, "tobac"), "tobac")
            }
            
            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG['mesonh']['tobac']['analysis'].keys():
                
                # Add default tracking featured_id variable in place of variable if not present
                if ("variable" not in CONFIG['mesonh']['tobac']['analysis'][var.lower()]): CONFIG['mesonh']['tobac']['analysis'][var.lower()]["variable"] = CONFIG["mesonh"]["feature_tracking_var"].upper()
                
                mesonh_analysis_data[var.lower()] = (get_var(analysis_object, var, CONFIG['verbose'], **CONFIG['mesonh']['tobac']['analysis'][var.lower()]))
                
    
        if (CONFIG['verbose']): print("=====Converting MesoNH tobac Output to CoMET-UDAF=====")
        
        # Add all products to return dict
        user_return_dict["mesonh"]["tobac"] = {
            "feature_id": mesonh_features,
            "UDAF_features": feature_id_to_UDAF(mesonh_features, "tobac"),
            "linking": mesonh_tracks,
            "UDAF_linking": linking_to_UDAF(mesonh_tracks, "tobac"),
            "segmentation_2d": mesonh_segmentation2d,
            "UDAF_segmentation_2d": segmentation_to_UDAF(mesonh_segmentation2d[0], linking_to_UDAF(mesonh_tracks, "tobac"), "tobac"),
            "segmentation_3d": mesonh_segmentation3d,
            "UDAF_segmentation_3d": segmentation_to_UDAF(mesonh_segmentation3d[0], linking_to_UDAF(mesonh_tracks, "tobac"), "tobac"),
            "analysis": mesonh_analysis_data
            }
        
        if (CONFIG['verbose']): print("=====MesoNH tobac Tracking Complete=====")
        
        # Send return dict to queue if there is a queue object passed
        if (queue is not None):
            queue.put(user_return_dict)
            return
        
        # Return dictionary
        return (user_return_dict)
    
    else:
        raise Exception("!=====No Tracker or Invalid Tracker Found in CONFIG=====!")
        return



"""
Inputs:
    CONFIG: User configuration file
Outputs:
    user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
"""
def run_nexrad(CONFIG, queue = None):
    from .tracker_output_translation_layer import feature_id_to_UDAF, linking_to_UDAF, segmentation_to_UDAF
    from .nexrad_load import nexrad_load_netcdf_iris
    
    if (CONFIG['verbose']): print("=====Loading NEXRAD Data=====")
    
    # determine if gridding is necessary or not
    if ("gridding" in CONFIG['nexrad']):
        
        if (CONFIG['verbose']): print("=====Gridding NEXRAD Data=====")
        nexrad_tracking_cube, nexrad_tracking_xarray = nexrad_load_netcdf_iris(CONFIG['nexrad']['path_to_data'], 'ar2v', CONFIG['nexrad']['feature_tracking_var'], CONFIG, CONFIG['nexrad']['gridding']['gridding_save_path'])
    
    else:
        nexrad_tracking_cube, nexrad_tracking_xarray = nexrad_load_netcdf_iris(CONFIG['nexrad']['path_to_data'], 'nc', CONFIG['nexrad']['feature_tracking_var'], CONFIG)
    
    # Add xarrays and cubes to return dict
    user_return_dict = {}
    
    user_return_dict["nexrad"] = {
        "tracking_xarray": nexrad_tracking_xarray,
        "tracking_cube": nexrad_tracking_cube,
        "segmentation_xarray": nexrad_tracking_xarray,
        "segmentation_cube": nexrad_tracking_cube
    }
    
    
    # determine which tracker to use
    if ("tobac" in CONFIG['nexrad']):
        from .nexrad_tobac import nexrad_tobac_feature_id, nexrad_tobac_linking, nexrad_tobac_segmentation
        
        nexrad_features = None
        nexrad_tracks = None
        nexrad_segmentation2d = None
        nexrad_segmentation3d = None
        nexrad_analysis_data = {}
        
        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if ("feature_id" in CONFIG['nexrad']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting NEXRAD tobac Feature ID=====")
            
            nexrad_features = nexrad_tobac_feature_id(nexrad_tracking_cube, CONFIG)
        
        if ("linking" in CONFIG['nexrad']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting NEXRAD tobac Feature Linking=====")
            
            nexrad_tracks = nexrad_tobac_linking(nexrad_tracking_cube, nexrad_features, CONFIG)
        
        if ("segmentation_2d" in CONFIG['nexrad']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting NEXRAD tobac 2D Segmentation=====")
            
            nexrad_segmentation2d = nexrad_tobac_segmentation(nexrad_tracking_cube, nexrad_features, '2d', CONFIG, CONFIG['nexrad']['tobac']['segmentation_2d']['height'])
    
        if ("segmentation_3d" in CONFIG['nexrad']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting NEXRAD tobac 3D Segmentation=====")
            
            nexrad_segmentation3d = nexrad_tobac_segmentation(nexrad_tracking_cube, nexrad_features, '3d', CONFIG)

        if ("analysis" in CONFIG['nexrad']['tobac']):
            
            from CoMET.analysis.get_vars import get_var
            
            if (CONFIG['verbose']): print("=====Starting NEXRAD tobac Analysis Calculations=====")
            
            # Create analysis object
            analysis_object = {
                "tracking_xarray": nexrad_tracking_xarray,
                "segmentation_xarray": nexrad_tracking_xarray,
                "UDAF_features": feature_id_to_UDAF(nexrad_features, "tobac"),
                "UDAF_linking": linking_to_UDAF(nexrad_tracks, "tobac"),
                "UDAF_segmentation_2d": segmentation_to_UDAF(nexrad_segmentation2d[0], linking_to_UDAF(nexrad_tracks, "tobac"), "tobac"),
                "UDAF_segmentation_3d": segmentation_to_UDAF(nexrad_segmentation3d[0], linking_to_UDAF(nexrad_tracks, "tobac"), "tobac")
            }
                
            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG['nexrad']['tobac']['analysis'].keys():
                
                # Add default tracking featured_id variable in place of variable if not present
                if ("variable" not in CONFIG['nexrad']['tobac']['analysis'][var.lower()]): CONFIG['nexrad']['tobac']['analysis'][var.lower()]["variable"] = CONFIG["nexrad"]["feature_tracking_var"].upper()
                
                nexrad_analysis_data[var.lower()] = (get_var(analysis_object, var, CONFIG['verbose'], **CONFIG['nexrad']['tobac']['analysis'][var.lower()]))

    
        if (CONFIG['verbose']): print("=====Converting NEXRAD tobac Output to CoMET-UDAF=====")

        # Add all products to return dict
        user_return_dict["nexrad"]["tobac"] = {
            "feature_id": nexrad_features,
            "UDAF_features": feature_id_to_UDAF(nexrad_features, "tobac"),
            "linking": nexrad_tracks,
            "UDAF_linking": linking_to_UDAF(nexrad_tracks, "tobac"),
            "segmentation_2d": nexrad_segmentation2d,
            "UDAF_segmentation_2d": segmentation_to_UDAF(nexrad_segmentation2d[0], linking_to_UDAF(nexrad_tracks, "tobac"), "tobac"),
            "segmentation_3d": nexrad_segmentation3d,
            "UDAF_segmentation_3d": segmentation_to_UDAF(nexrad_segmentation3d[0], linking_to_UDAF(nexrad_tracks, "tobac"), "tobac"),
            "analysis": nexrad_analysis_data
            }
        
        if (CONFIG['verbose']): print("=====NEXRAD tobac Tracking Complete=====")
    
        # Send return dict to queue if there is a queue object passed
        if (queue is not None):
            queue.put(user_return_dict)
            return
    
        # Return dictionary
        return (user_return_dict)
    
    else:
        raise Exception("!=====No Tracker or Invalid Tracker Found in CONFIG=====!")
        return

    

"""
Inputs:
    CONFIG: User configuration file
Outputs:
    user_return_dict: A dictionary object which contanis all tobac and CoMET-UDAF standard outputs
"""
def run_goes(CONFIG, queue = None):
    from .tracker_output_translation_layer import feature_id_to_UDAF, linking_to_UDAF, segmentation_to_UDAF
    from .goes_load import goes_load_netcdf_iris

    if (CONFIG['verbose']): print("=====Loading GOES Data=====")
    
    goes_tracking_cube, goes_tracking_xarray = goes_load_netcdf_iris(CONFIG['goes']['path_to_data'], CONFIG['goes']['feature_tracking_var'], CONFIG)
    
    # Add xarrays and cubes to return dict
    user_return_dict = {}
    
    user_return_dict["goes"] = {
        "tracking_xarray": goes_tracking_xarray,
        "tracking_cube": goes_tracking_cube,
        "segmentation_xarray": goes_tracking_xarray,
        "segmentation_cube": goes_tracking_cube
    }
    
    # determine which tracker to use
    if ("tobac" in CONFIG['goes']):
        from .goes_tobac import goes_tobac_feature_id, goes_tobac_linking, goes_tobac_segmentation
    
        goes_features = None
        goes_tracks = None
        goes_segmentation2d = None
        goes_analysis_data = {}
        
        
        # Perform all cell tracking, id, and segmentation steps. Then add results to return dict
        if ("feature_id" in CONFIG['goes']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting GOES tobac Feature ID=====")
            
            goes_features = goes_tobac_feature_id(goes_tracking_cube, CONFIG)
        
        if ("linking" in CONFIG['goes']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting GOES tobac Feature Linking=====")
            
            goes_tracks = goes_tobac_linking(goes_tracking_cube, goes_features, CONFIG)
        
        if ("segmentation_2d" in CONFIG['goes']['tobac']):
            
            if (CONFIG['verbose']): print("=====Starting GOES tobac 2D Segmentation=====")
            
            # TB is 2D from satellite so no height parameter necessary
            goes_segmentation2d = goes_tobac_segmentation(goes_tracking_cube, goes_features, CONFIG)
        
        if ("analysis" in CONFIG['goes']['tobac']):
            
            from CoMET.analysis.get_vars import get_var
            
            if (CONFIG['verbose']): print("=====Starting GOES tobac Analysis Calculations=====")
            
            # Create analysis object
            analysis_object = {
                "tracking_xarray": goes_tracking_xarray,
                "segmentation_xarray": goes_tracking_xarray,
                "UDAF_features": feature_id_to_UDAF(goes_features, "tobac"),
                "UDAF_linking": linking_to_UDAF(goes_tracks, "tobac"),
                "UDAF_segmentation_2d": segmentation_to_UDAF(goes_segmentation2d[0], linking_to_UDAF(goes_tracks, "tobac"), "tobac"),
                "UDAF_segmentation_3d": None
            }
            
            # Calcaulte each variable of interest and append to analysis data array
            for var in CONFIG['goes']['tobac']['analysis'].keys():
                
                # Add default tracking featured_id variable in place of variable if not present
                if ("variable" not in CONFIG['goes']['tobac']['analysis'][var.lower()]): CONFIG['goes']['tobac']['analysis'][var.lower()]["variable"] = CONFIG["goes"]["feature_tracking_var"].upper()
                
                goes_analysis_data[var.lower()] = (get_var(analysis_object, var, CONFIG['verbose'], **CONFIG['goes']['tobac']['analysis'][var.lower()]))


        if (CONFIG['verbose']): print("=====Converting GOES tobac Output to CoMET-UDAF=====")
        
        # Add all products to return dict
        user_return_dict["goes"]["tobac"] = {
            "feature_id": goes_features,
            "UDAF_features": feature_id_to_UDAF(goes_features, "tobac"),
            "linking": goes_tracks,
            "UDAF_linking": linking_to_UDAF(goes_tracks, "tobac"),
            "segmentation_2d": goes_segmentation2d,
            "UDAF_segmentation_2d": segmentation_to_UDAF(goes_segmentation2d[0], linking_to_UDAF(goes_tracks, "tobac"), "tobac"),
            "analysis": goes_analysis_data
            }
        
        if (CONFIG['verbose']): print("=====GOES tobac Tracking Complete=====")
        
        # Send return dict to queue if there is a queue object passed
        if (queue is not None):
            queue.put(user_return_dict)
            return
        
        # Return dictionary
        return (user_return_dict)
    
    else:
        raise Exception("!=====No Tracker or Invalid Tracker Found in CONFIG=====!")
        return
