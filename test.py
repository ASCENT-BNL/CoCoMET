#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:44:52 2024

@author: thahn
"""


"""
TODO: Write unit tests using small test dataset(s)? 
"""

import CoMET

CONFIG = CoMET.CoMET_Load('./Example_Configs/boilerplate.yml')

print(CONFIG)


# Example params for testing
feature_detection_params_dbz = dict()
feature_detection_params_dbz['threshold'] = [30, 40, 50, 60]
feature_detection_params_dbz['target'] = 'maximum'
feature_detection_params_dbz['position_threshold'] = 'weighted_diff'
feature_detection_params_dbz['sigma_threshold'] = 0.5
feature_detection_params_dbz['n_min_threshold'] = 4

# Example params for testing
feature_detection_params_w = dict()
feature_detection_params_w['threshold'] = [2,3,4,5]
feature_detection_params_w['target'] = 'maximum'
feature_detection_params_w['position_threshold'] = 'weighted_diff'
feature_detection_params_w['sigma_threshold'] = 0.5
feature_detection_params_w['n_min_threshold'] = 4

# Example params for testing
feature_detection_params_tb = dict()
feature_detection_params_tb['threshold'] = [250,260,270]
feature_detection_params_tb['target'] = 'minimum'
feature_detection_params_tb['position_threshold'] = 'weighted_diff'
feature_detection_params_tb['sigma_threshold'] = 0.5
feature_detection_params_tb['n_min_threshold'] = 4

# Keyword arguments for linking step:
parameters_linking_dbz = dict()
parameters_linking_dbz['method_linking']='predict'
parameters_linking_dbz['adaptive_stop']=0.2
parameters_linking_dbz['adaptive_step']=0.95
parameters_linking_dbz['order']=1
parameters_linking_dbz['subnetwork_size']=10
parameters_linking_dbz['memory']=1
parameters_linking_dbz['v_max']=20

#Segmentation arguments
parameters_segmentation_dbz=dict()
parameters_segmentation_dbz['method']='watershed'
parameters_segmentation_dbz['threshold']=15  # dbZ threshold

#Segmentation arguments
parameters_segmentation_w=dict()
parameters_segmentation_w['method']='watershed'
parameters_segmentation_w['threshold']=1  # dbZ threshold

#Segmentation arguments
parameters_segmentation_tb=dict()
parameters_segmentation_tb['method']='watershed'
parameters_segmentation_tb['threshold']=280  # dbZ threshold


wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'dbz')
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', feature_detection_params_dbz)
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', 'dbz', wrf_features, parameters_linking_dbz)
wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', 'dbz', wrf_features, parameters_segmentation_dbz, '3D')
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', 'dbz', wrf_features, parameters_segmentation_dbz, '2D', 2000)

wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'w')
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', feature_detection_params_w)
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', 'w', wrf_features, parameters_linking_dbz)
wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', 'w', wrf_features, parameters_segmentation_w, '3D')
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', 'w', wrf_features, parameters_segmentation_w, '2D', 2000)

wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'tb')
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', feature_detection_params_tb)
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', 'tb', wrf_features, parameters_linking_dbz)
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', 'tb', wrf_features, parameters_segmentation_tb, '2D', 2000)