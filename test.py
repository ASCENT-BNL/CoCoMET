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


CONFIG = CoMET.CoMET_Load('./Example_Configs/wrf_dbz_config.yml')
print('=====Starting WRF dbz Tracking=====')
wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'dbz')
print("*////")
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', CONFIG)
print("**///")
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, CONFIG)
print("***//")
wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '3D', CONFIG)
print("****/")
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', CONFIG, 2000)
print('=====Finished WRF dbz Tracking=====')


print('=====Starting NEXRAD dbz Tracking=====')
# radar_cube,radar_xarray=CoMET.nexrad_load_netcdf_iris('/D3/data/thahn/NEXRAD/HAS012527906/0003/*_V06.ar2v', 'ar2v', 'dbz', CONFIG, '/D3/data/thahn/NEXRAD/HAS012527906/grids/')
radar_cube,radar_xarray=CoMET.nexrad_load_netcdf_iris('/D3/data/thahn/NEXRAD/HAS012527906/grids/*', 'nc', 'dbz', CONFIG)
print("*////")
radar_features = CoMET.nexrad_tobac_feature_id(radar_cube, 'IC', CONFIG)
print("**///")
radar_tracks = CoMET.nexrad_tobac_linking(radar_cube, 'IC', radar_features, CONFIG)
print("***//")
radar_segment_array_3d, radar_segment_pd_3d = CoMET.nexrad_tobac_segmentation(radar_cube, 'IC', radar_features, '3D', CONFIG)
print("****/")
radar_segment_array_2d, radar_segment_pd_2d = CoMET.nexrad_tobac_segmentation(radar_cube, 'IC', radar_features, '2D', CONFIG, 2000)
print('=====Finished NEXRAD dbz Tracking=====')


CONFIG = CoMET.CoMET_Load('./Example_Configs/wrf_w_config.yml')
print('=====Starting WRF w Tracking=====')
wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'w')
print("*////")
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', CONFIG)
print("**///")
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, CONFIG)
print("***//")
wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '3D', CONFIG)
print("****/")
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', CONFIG, 2000)
print('=====Finished WRF w Tracking=====')


CONFIG = CoMET.CoMET_Load('./Example_Configs/wrf_tb_config.yml')
print('=====Starting WRF tb Tracking=====')
wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'tb')
print("*///")
wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', CONFIG)
print("**//")
wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, CONFIG)
print("***/")
wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', CONFIG, None)
print('=====Finished WRF tb Tracking=====')


print('=====Starting GOES tb Tracking=====')
goes_cube, goes_xarray = CoMET.goes_load_netcdf_iris("/D3/data/thahn/GOES/noaa-goes16/ABI-L2-CMIPC/2023/2023_07_11/*", 'tb', CONFIG)
print("*///")
goes_features = CoMET.goes_tobac_feature_id(goes_cube, 'IC', CONFIG)
print("**//")
goes_tracks = CoMET.goes_tobac_linking(goes_cube, 'IC', goes_features, CONFIG)
print("***/")
goes_segment_array_2d, segment_pd_2d = CoMET.goes_tobac_segmentation(goes_cube, 'IC', goes_features, CONFIG)
print('=====Finished GOES tb Tracking=====')