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
"""
def CoMET_Load(path_to_config):
    import ast
    import yaml
    import numpy as np
    
    # Open and read config file
    with open(path_to_config, 'r') as f:
        CONFIG = yaml.safe_load(f)
        
        
    # TODO: Do some checks here    
    
    # If nexrad gridding is needed, change grid shapes and limits back to tuples
    if ("nexrad_gridding" in CONFIG):
        
        # Convert grid_shape and grid_limits back into proper tuples and ensure correct data types--auto correct automatically
        grid_shape = ast.literal_eval(CONFIG['nexrad_gridding']['grid_shape'])
        # Ensure int grid shapes
        grid_shape = tuple([int(grid_shape[ii]) for ii in range(len(grid_shape))])
        
        # Ensure float for grid_limits
        grid_limits = ast.literal_eval(CONFIG['nexrad_gridding']['grid_limits'])
        grid_limits = np.array(grid_limits).astype(float)
        grid_limits = tuple([tuple(row) for row in grid_limits])

        # Adjust CONFIG values
        CONFIG['nexrad_gridding']['grid_shape'] = grid_shape
        CONFIG['nexrad_gridding']['grid_limits'] = grid_limits
    
    
    
    return(CONFIG)
        