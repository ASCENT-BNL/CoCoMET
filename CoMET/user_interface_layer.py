#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:55:16 2024

@author: thahn
"""

# =============================================================================
# This is the interface layer between all of CoMET's backend functionality and the user. A sort of parser for a configuration input file. 
# =============================================================================


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
    
    # if manual_mode = True, just return the loaded CONFIG
    if (manual_mode): return(CONFIG)
    
    # Create dictionary which holds all returnable info
    user_return_dict = {}
    
    # Otherwise, go through defined setups and run them
    
    # if wrf is present in CONFIG, run the necessary wrf functions
    if ("wrf" in CONFIG):
        import wrf_load as wl
        
        # Add wrf section to return dict
        if ("wrf" not in user_return_dict):
            user_return_dict["wrf"] = {}
        
        wrf_tracking_cube, wrf_tracking_xarray = wl.wrf_load_netcdf_iris(CONFIG['wrf']['path_to_data'], CONFIG['wrf']['feature_tracking_var'], CONFIG)
        
        # if tracking and segmentation variables are different, load seperately
        if (CONFIG['wrf']['feature_tracking_var'] != CONFIG['wrf']['segmentation_var']):
            wrf_segmentation_cube, wrf_segmentation_xarray = wl.wrf_load_netcdf_iris(CONFIG['wrf']['path_to_data'], CONFIG['wrf']['segmentation_var'], CONFIG)
        else:
            wrf_segmentation_cube = wrf_tracking_cube
            wrf_segmentation_xarray = wrf_tracking_xarray
        
        
        # now determine which tracker to use
        if ("tobac" in CONFIG['wrf']):
            import wrf_tobac as wt
            
            wrf_features = None
            wrf_tracks = None
            wrf_segmentation2d = None
            wrf_segmentation3d = None
            
            # Perform all cell tracking, id, and segmentation steps. Then return results as tuple
            if ("feature_id" in CONFIG['wrf']['tobac']):
                
                # If height present, do 2D tracking
                if ("height" in CONFIG['wrf']['tobac']['feature_id']):
                    
                    # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
                    height_index = wt.find_nearest(wrf_tracking_cube.coord('altitude').points, CONFIG['wrf']['tobac']['feature_id']['height'])
                    
                    # Delete height before passing to tobac
                    del CONFIG['wrf']['tobac']['feature_id']['height']
                    
                    temp_cube = wrf_tracking_cube[:,height_index]
                    wrf_features = wt.wrf_tobac_feature_id(temp_cube, CONFIG['wrf']['tracking_type'], CONFIG)
                    
                else:
                    wrf_features = wt.wrf_tobac_feature_id(wrf_tracking_cube, CONFIG['wrf']['tracking_type'], CONFIG)
            
            if ("linking" in CONFIG['wrf']['tobac']):
                wrf_tracks = wt.wrf_tobac_linking(wrf_tracking_cube, CONFIG['wrf']['tracking_type'], wrf_features, CONFIG)
            
            if ("segmentation_2d" in CONFIG['wrf']['tobac']):
                # remove height from CONFIG before passing to tobac
                ht = CONFIG['wrf']['tobac']['segmentation_2d']['height']
                del CONFIG['wrf']['tobac']['segmentation_2d']['height']
                
                wrf_segmentation2d = wt.wrf_tobac_segmentation(wrf_segmentation_cube, CONFIG['wrf']['tracking_type'], wrf_features, '2d', CONFIG, ht)
        
            if ("segmentation_3d" in CONFIG['wrf']['tobac']):
                wrf_segmentation3d = wt.wrf_tobac_segmentation(wrf_segmentation_cube, CONFIG['wrf']['tracking_type'], wrf_features, '3d', CONFIG)
        
            # TODO: Figure out new way of returning data to user. Current way will return whichever option comes first and not run the others
            # Add all products to return dict
            user_return_dict["wrf"]["tobac"] = {
                "feature_id": wrf_features,
                "linking": wrf_tracks,
                "segmentation_2d": wrf_segmentation2d,
                "segmentation_3d": wrf_segmentation3d
                }
        
        else:
            raise Exception("!=====No Tracker or Invalid Tracker Found in CONFIG=====!")
            return
    
    
    
    # Handle NEXRAD data
    if ('nexrad' in CONFIG):
        import nexrad_load as nl
        
        # Add nexrad section to return dict
        if ("nexrad" not in user_return_dict):
            user_return_dict["nexrad"] = {}
        
        # determine if gridding is necessary or not
        if ("gridding" in CONFIG['nexrad']):
            nexrad_tracking_cube, nexrad_tracking_xarray = nl.nexrad_load_netcdf_iris(CONFIG['nexrad']['path_to_data'], 'ar2v', CONFIG['nexrad']['feature_tracking_var'], CONFIG, CONFIG['nexrad']['gridding']['gridding_save_path'])
        
        else:
            nexrad_tracking_cube, nexrad_tracking_xarray = nl.nexrad_load_netcdf_iris(CONFIG['nexrad']['path_to_data'], 'nc', CONFIG['nexrad']['feature_tracking_var'], CONFIG)
        
        
        # determine which tracker to use
        if ("tobac" in CONFIG['nexrad']):
            import nexrad_tobac as nt
            
            nexrad_features = None
            nexrad_tracks = None
            nexrad_segmentation2d = None
            nexrad_segmentation3d = None
            
            # Perform all cell tracking, id, and segmentation steps. Then return results as tuple
            if ("feature_id" in CONFIG['nexrad']['tobac']):
                
                # If height present, do 2D tracking
                if ("height" in CONFIG['nexrad']['tobac']['feature_id']):
                    
                    # Find the nearest model height to the entered segmentation height--bypasses precision issues and allows for selection of rounded heights
                    height_index = nt.find_nearest(nexrad_tracking_cube.coord('altitude').points, CONFIG['nexrad']['tobac']['feature_id']['height'])
                    
                    # Delete height before passing to tobac
                    del CONFIG['nexrad']['tobac']['feature_id']['height']
                    
                    temp_cube = nexrad_tracking_cube[:,height_index]
                    nexrad_features = nt.nexrad_tobac_feature_id(temp_cube, CONFIG['nexrad']['tracking_type'], CONFIG)
                    
                else:
                    nexrad_features = nt.nexrad_tobac_feature_id(nexrad_tracking_cube, CONFIG['nexrad']['tracking_type'], CONFIG)
            
            if ("linking" in CONFIG['nexrad']['tobac']):
                nexrad_tracks = nt.nexrad_tobac_linking(nexrad_tracking_cube, CONFIG['nexrad']['tracking_type'], nexrad_features, CONFIG)
            
            if ("segmentation_2d" in CONFIG['nexrad']['tobac']):
                # remove height from CONFIG before passing to tobac
                ht = CONFIG['nexrad']['tobac']['segmentation_2d']['height']
                del CONFIG['nexrad']['tobac']['segmentation_2d']['height']
                
                nexrad_segmentation2d = nt.nexrad_tobac_segmentation(nexrad_tracking_cube, CONFIG['nexrad']['tracking_type'], nexrad_features, '2d', CONFIG, ht)
        
            if ("segmentation_3d" in CONFIG['nexrad']['tobac']):
                nexrad_segmentation3d = nt.nexrad_tobac_segmentation(nexrad_tracking_cube, CONFIG['nexrad']['tracking_type'], nexrad_features, '3d', CONFIG)
        

            # Add all products to return dict
            user_return_dict["nexrad"]["tobac"] = {
                "feature_id": nexrad_features,
                "linking": nexrad_tracks,
                "segmentation_2d": nexrad_segmentation2d,
                "segmentation_3d": nexrad_segmentation3d
                }
        
        else:
            raise Exception("!=====No Tracker or Invalid Tracker Found in CONFIG=====!")
            return
    
    
    
    # Handle GOES data
    if ("goes" in CONFIG):
        import goes_load as gl
        
        # Add goes section to return dict
        if ("goes" not in user_return_dict):
            user_return_dict["goes"] = {}
        
        goes_tracking_cube, goes_tracking_xarray = gl.goes_load_netcdf_iris(CONFIG['goes']['path_to_data'], CONFIG['goes']['feature_tracking_var'], CONFIG)
        
        # determine which tracker to use
        if ("tobac" in CONFIG['goes']):
            import goes_tobac as gt
        
            goes_features = None
            goes_tracks = None
            goes_segmentation2d = None
            
            
            # Perform all cell tracking, id, and segmentation steps. Then return results as tuple
            if ("feature_id" in CONFIG['goes']['tobac']):
                goes_features = gt.goes_tobac_feature_id(goes_tracking_cube, CONFIG['goes']['tracking_type'], CONFIG)
            
            if ("linking" in CONFIG['goes']['tobac']):
                goes_tracks = gt.goes_tobac_linking(goes_tracking_cube, CONFIG['goes']['tracking_type'], goes_features, CONFIG)
            
            if ("segmentation_2d" in CONFIG['goes']['tobac']):
                # TB is 2D from satellite so no height parameter necessary
                goes_segmentation2d = gt.goes_tobac_segmentation(goes_tracking_cube, CONFIG['goes']['tracking_type'], goes_features, CONFIG)
        

            # Add all products to return dict
            user_return_dict["goes"]["tobac"] = {
                "feature_id": goes_features,
                "linking": goes_tracks,
                "segmentation_2d": goes_segmentation2d
                }
        
        else:
            raise Exception("!=====No Tracker or Invalid Tracker Found in CONFIG=====!")
            return

    # return dict at end
    return (user_return_dict)


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
        