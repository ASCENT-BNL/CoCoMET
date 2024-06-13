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


# Example params for testing
feature_detection_params_dbz = dict()
feature_detection_params_dbz['threshold'] = [20, 30, 40, 50, 60]
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


print('=====Starting WRF dbz Tracking=====')
wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'dbz')
print("*////")
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', feature_detection_params_dbz)
print("**///")
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, parameters_linking_dbz)
print("***//")
wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '3D', parameters_segmentation_dbz)
print("****/")
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', parameters_segmentation_dbz, 2000)
print('=====Finished WRF dbz Tracking=====')


print('=====Starting WRF w Tracking=====')
wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'w')
print("*////")
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', feature_detection_params_w)
print("**///")
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, parameters_linking_dbz)
print("***//")
wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '3D', parameters_segmentation_w)
print("****/")
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', parameters_segmentation_w, 2000)
print('=====Finished WRF w Tracking=====')


print('=====Starting WRF tb Tracking=====')
wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'tb')
print("*///")
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', feature_detection_params_tb)
print("**//")
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, parameters_linking_dbz)
print("***/")
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', parameters_segmentation_tb, None)
print('=====Finished WRF tb Tracking=====')


print('=====Starting NEXRAD dbz Tracking=====')
radar_cube,radar_xarray=CoMET.nexrad_load_netcdf_iris('/D3/data/thahn/NEXRAD/HAS012527906/0003/*_V06.ar2v', 'ar2v', 'dbz', CONFIG, '/D3/data/thahn/NEXRAD/HAS012527906/grids/')
print("*////")
radar_features = CoMET.nexrad_tobac_feature_id(radar_cube, 'IC', feature_detection_params_dbz)
print("**///")
radar_tracks = CoMET.nexrad_tobac_linking(radar_cube, 'IC', radar_features, parameters_linking_dbz)
print("***//")
radar_segment_array_3d, radar_segment_pd_3d = CoMET.nexrad_tobac_segmentation(radar_cube, 'IC', radar_features, '3D', parameters_segmentation_dbz)
print("****/")
radar_segment_array_2d, radar_segment_pd_2d = CoMET.nexrad_tobac_segmentation(radar_cube, 'IC', radar_features, '2D', parameters_segmentation_dbz, 2000)
print('=====Finished WRF dbz Tracking=====')

print(radar_tracks)