#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:44:52 2024

@author: thahn
"""


"""
TODO: Add full testing coverage
"""

# Test CONFIGS
wrf_CONFIG_dbz = {
    'verbose': True, # Whether to use verbose output
    'parallel_processing': True, # [bool] Whether or not to use parallel processing for certain tasks
    'max_cores': 32, # Number of cores to use if parallel_processing==True; Enter None for unlimited

    'wrf': {
        'path_to_data': '/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*',

        'tracking_type': 'IC',

        'feature_tracking_var': 'dbz',
        'segmentation_var': 'dbz',

        'tobac': {
            'feature_id': {
                'threshold': [30, 40, 50, 60],
                'target': 'maximum',
                'position_threshold': 'weighted_diff',
                'sigma_threshold': 0.5,
                'n_min_threshold': 4
            },
            
            'linking': { 
                'method_linking': 'predict',
                'adaptive_stop': 0.2,
                'adaptive_step': 0.95,
                'order': 1,
                'subnetwork_size': 10,
                'memory': 1,
                'v_max': 20
            },
            
            "segmentation_3d": {
                "method": 'watershed',
                "threshold": 15
            }
            
        }
        
    }
    
}


import CoMET
import unittest

class Test_User_Interface_Layer(unittest.TestCase):
    
    # Test to make sure CoMET can handle CONFIG inputs well
    def test_CoMET_start(self):
        self.assertEqual(CoMET.CoMET_start(None, True, CONFIG=wrf_CONFIG_dbz), wrf_CONFIG_dbz)
        self.assertEqual(CoMET.CoMET_start('./examples/example_configs/wrf_test_config_dbz.yml', True), wrf_CONFIG_dbz)
    
    # Make sure the CONFIG loading is working
    def test_config_load(self):
        CONFIG = CoMET.CoMET_load('./examples/example_configs/boilerplate.yml')
        self.assertEqual(type(CONFIG), dict, "Should return dictionary object")
        
    
# Test for WRF input combined with tobac tracking
class Test_WRF_tobac(unittest.TestCase):
    
    CONFIG_dbz = None
    CONFIG_w = None
    CONFIG_tb = None
    
    wrf_cube_dbz = None
    wrf_cube_w = None
    wrf_cube_tb = None
    
    wrf_features_dbz = None
    wrf_features_w = None
    wrf_features_tb = None
    
    # Test the loading of the netcdf and iris for reflectivity
    def test_wrf_load_netcdf_iris_dbz(self):
        CONFIG = CoMET.CoMET_start('./examples/example_configs//wrf_test_config_dbz.yml', True)
        self.__class__.CONFIG_dbz = CONFIG
        
        wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris('/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*', 'dbz', CONFIG)
        self.__class__.wrf_cube_dbz = wrf_cube
        
        self.assertEqual(wrf_cube.name(), 'DBZ')
        self.assertEqual(len(wrf_cube.coords()), 11)
        self.assertAlmostEqual(wrf_cube.data.max(),60.6, places=2)
        self.assertAlmostEqual(wrf_cube.data.min(),-30, places=2)
        self.assertTupleEqual(wrf_cube.data.shape, (289, 44, 213, 219))
        self.assertEqual(wrf_cube.dim_coords[0].name(), 'time')
        self.assertEqual(wrf_cube.dim_coords[1].name(), 'altitude')
        self.assertEqual(wrf_cube.dim_coords[2].name(), 'south_north')
        self.assertEqual(wrf_cube.dim_coords[3].name(), 'west_east')
        self.assertAlmostEqual(wrf_cube.coord('latitude').points.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_cube.coord('latitude').points.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_cube.coord('longitude').points.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_cube.coord('longitude').points.max(), -95.72, places=2)
        
        
        self.assertTrue('DBZ' in wrf_xarray)
        self.assertTupleEqual(wrf_xarray.DBZ.shape, (289, 44, 213, 219))
        self.assertAlmostEqual(wrf_xarray.DBZ.values.max(), 60.6, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.values.min(), -30, places=2)
        self.assertTupleEqual(wrf_xarray.DBZ.dims, ('Time', 'bottom_top', 'south_north', 'west_east'))
        self.assertAlmostEqual(wrf_xarray.DBZ.XLAT.values.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.XLAT.values.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.XLONG.values.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.XLONG.values.max(), -95.72, places=2)
    
        
    # Test the loading of the netcdf and iris for updraft
    def test_wrf_load_netcdf_iris_w(self):
        CONFIG = CoMET.CoMET_start('./examples/example_configs/wrf_test_config_w.yml', True)
        self.__class__.CONFIG_w = CONFIG
        
        wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris('/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*', 'W', CONFIG)
        self.__class__.wrf_cube_w = wrf_cube
        
        self.assertEqual(wrf_cube.name(), 'WA')
        self.assertEqual(len(wrf_cube.coords()), 11)
        self.assertAlmostEqual(wrf_cube.data.max(),19.08, places=2)
        self.assertAlmostEqual(wrf_cube.data.min(),-8.76, places=2)
        self.assertTupleEqual(wrf_cube.data.shape, (289, 44, 213, 219))
        self.assertEqual(wrf_cube.dim_coords[0].name(), 'time')
        self.assertEqual(wrf_cube.dim_coords[1].name(), 'altitude')
        self.assertEqual(wrf_cube.dim_coords[2].name(), 'south_north')
        self.assertEqual(wrf_cube.dim_coords[3].name(), 'west_east')
        self.assertAlmostEqual(wrf_cube.coord('latitude').points.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_cube.coord('latitude').points.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_cube.coord('longitude').points.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_cube.coord('longitude').points.max(), -95.72, places=2)
        
        
        self.assertTrue('WA' in wrf_xarray)
        self.assertTupleEqual(wrf_xarray.WA.shape, (289, 44, 213, 219))
        self.assertAlmostEqual(wrf_xarray.WA.values.max(), 19.08, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.values.min(), -8.76, places=2)
        self.assertTupleEqual(wrf_xarray.WA.dims, ('Time', 'bottom_top', 'south_north', 'west_east'))
        self.assertAlmostEqual(wrf_xarray.WA.XLAT.values.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.XLAT.values.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.XLONG.values.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.XLONG.values.max(), -95.72, places=2)
        
    
    # Test the loading of the netcdf and iris for brightness temperature
    def test_wrf_load_netcdf_iris_tb(self):
        CONFIG = CoMET.CoMET_start('./examples/example_configs/wrf_test_config_tb.yml', True)
        self.__class__.CONFIG_tb = CONFIG
        
        wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris('/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*', 'tb', CONFIG)
        self.__class__.wrf_cube_tb = wrf_cube
        
        self.assertEqual(wrf_cube.name(), 'TB')
        self.assertEqual(len(wrf_cube.coords()), 9)
        self.assertAlmostEqual(wrf_cube.data[1:].max(),305.86, places=2)
        self.assertAlmostEqual(wrf_cube.data[1:].min(),213.65, places=2)
        self.assertTupleEqual(wrf_cube.data.shape, (289, 213, 219))
        self.assertEqual(wrf_cube.dim_coords[0].name(), 'time')
        self.assertEqual(wrf_cube.dim_coords[1].name(), 'south_north')
        self.assertEqual(wrf_cube.dim_coords[2].name(), 'west_east')
        self.assertAlmostEqual(wrf_cube.coord('latitude').points.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_cube.coord('latitude').points.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_cube.coord('longitude').points.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_cube.coord('longitude').points.max(), -95.72, places=2)
        
        
        self.assertTrue('TB' in wrf_xarray)
        self.assertTupleEqual(wrf_xarray.TB.shape, (289, 213, 219))
        self.assertAlmostEqual(wrf_xarray.TB.values[1:].max(), 305.86, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.values[1:].min(), 213.65, places=2)
        self.assertTupleEqual(wrf_xarray.TB.dims, ('Time', 'south_north', 'west_east'))
        self.assertAlmostEqual(wrf_xarray.TB.XLAT.values.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.XLAT.values.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.XLONG.values.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.XLONG.values.max(), -95.72, places=2)
    
    
    # Test the loading of just the netcdf for reflectivity to ensure contintuity
    def test_wrf_load_netcdf_dbz(self):  
        wrf_xarray = CoMET.wrf_load_netcdf('/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*', 'dbz', self.__class__.CONFIG_dbz)
        
        self.assertTrue('DBZ' in wrf_xarray)
        self.assertTupleEqual(wrf_xarray.DBZ.shape, (289, 44, 213, 219))
        self.assertAlmostEqual(wrf_xarray.DBZ.values.max(), 60.6, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.values.min(), -30, places=2)
        self.assertTupleEqual(wrf_xarray.DBZ.dims, ('Time', 'bottom_top', 'south_north', 'west_east'))
        self.assertAlmostEqual(wrf_xarray.DBZ.XLAT.values.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.XLAT.values.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.XLONG.values.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_xarray.DBZ.XLONG.values.max(), -95.72, places=2)
        
    
    # Test the loading of just the netcdf for updrafts to ensure contintuity
    def test_wrf_load_netcdf_w(self):
        wrf_xarray = CoMET.wrf_load_netcdf('/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*', 'w', self.__class__.CONFIG_w)
        
        self.assertTrue('WA' in wrf_xarray)
        self.assertTupleEqual(wrf_xarray.WA.shape, (289, 44, 213, 219))
        self.assertAlmostEqual(wrf_xarray.WA.values.max(), 19.08, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.values.min(), -8.76, places=2)
        self.assertTupleEqual(wrf_xarray.WA.dims, ('Time', 'bottom_top', 'south_north', 'west_east'))
        self.assertAlmostEqual(wrf_xarray.WA.XLAT.values.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.XLAT.values.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.XLONG.values.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_xarray.WA.XLONG.values.max(), -95.72, places=2)
        
        
    # Test the loading of just the netcdf for brightness temperature to ensure contintuity
    def test_wrf_load_netcdf_tb(self):
        wrf_xarray = CoMET.wrf_load_netcdf('/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*', 'tb', self.__class__.CONFIG_tb)
        
        self.assertTrue('TB' in wrf_xarray)
        self.assertTupleEqual(wrf_xarray.TB.shape, (289, 213, 219))
        self.assertAlmostEqual(wrf_xarray.TB.values[1:].max(), 305.86, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.values[1:].min(), 213.65, places=2)
        self.assertTupleEqual(wrf_xarray.TB.dims, ('Time', 'south_north', 'west_east'))
        self.assertAlmostEqual(wrf_xarray.TB.XLAT.values.min(), 35.86, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.XLAT.values.max(), 38.22, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.XLONG.values.min(), -98.73, places=2)
        self.assertAlmostEqual(wrf_xarray.TB.XLONG.values.max(), -95.72, places=2)
    
    
    # Test tobac feature id for reflectivity
    def test_wrf_tobac_feature_id_dbz(self):
        import geopandas
        import numpy as np
        
        wrf_features = CoMET.wrf_tobac_feature_id(self.__class__.wrf_cube_dbz, 'IC', self.__class__.CONFIG_dbz)
        self.__class__.wrf_features_dbz = wrf_features
        
        self.assertTupleEqual(wrf_features.shape, (1516, 21))
        self.assertEqual(type(wrf_features), geopandas.geodataframe.GeoDataFrame)
        self.assertTrue(np.all(wrf_features.longitude!=0))
        self.assertTrue(np.all(wrf_features.latitude!=0))
        
    
    # Test tobac feature id for updrafts
    def test_wrf_tobac_feature_id_w(self):
        import geopandas
        import numpy as np
        
        wrf_features = CoMET.wrf_tobac_feature_id(self.__class__.wrf_cube_w, 'IC', self.__class__.CONFIG_w)
        self.__class__.wrf_features_w = wrf_features
        
        self.assertTupleEqual(wrf_features.shape, (437, 21))
        self.assertEqual(type(wrf_features), geopandas.geodataframe.GeoDataFrame)
        self.assertTrue(np.all(wrf_features.longitude!=0))
        self.assertTrue(np.all(wrf_features.latitude!=0))
        
    
    # Test tobac feature id for brightness temperature
    def test_wrf_tobac_feature_id_tb(self):
        import geopandas
        import numpy as np
        
        wrf_features = CoMET.wrf_tobac_feature_id(self.__class__.wrf_cube_tb, 'IC', self.__class__.CONFIG_tb)
        self.__class__.wrf_features_tb = wrf_features
        
        self.assertTupleEqual(wrf_features.shape, (475, 18))
        self.assertEqual(type(wrf_features), geopandas.geodataframe.GeoDataFrame)
        self.assertTrue(np.all(wrf_features.longitude!=0))
        self.assertTrue(np.all(wrf_features.latitude!=0))


    # Test tobac linking for reflectivity
    def test_wrf_tobac_linking_dbz(self):
        import geopandas
        import numpy as np
        
        wrf_tracks = CoMET.wrf_tobac_linking(self.__class__.wrf_cube_dbz, 'IC', self.__class__.wrf_features_dbz, self.__class__.CONFIG_dbz)
        
        self.assertTupleEqual(wrf_tracks.shape, (1516, 23))
        self.assertEqual(type(wrf_tracks), geopandas.geodataframe.GeoDataFrame)
        self.assertTrue(np.all(wrf_tracks.longitude!=0))
        self.assertTrue(np.all(wrf_tracks.latitude!=0))
        
    
    # Test tobac linking for updrafts
    def test_wrf_tobac_linking_w(self):
        import geopandas
        import numpy as np
        
        wrf_tracks = CoMET.wrf_tobac_linking(self.__class__.wrf_cube_w, 'IC', self.__class__.wrf_features_w, self.__class__.CONFIG_w)
        
        self.assertTupleEqual(wrf_tracks.shape, (437, 23))
        self.assertEqual(type(wrf_tracks), geopandas.geodataframe.GeoDataFrame)
        self.assertTrue(np.all(wrf_tracks.longitude!=0))
        self.assertTrue(np.all(wrf_tracks.latitude!=0))
        
    
    # Test tobac linking for brightness temperature
    def test_wrf_tobac_linking_tb(self):
        import geopandas
        import numpy as np
        
        wrf_tracks = CoMET.wrf_tobac_linking(self.__class__.wrf_cube_tb, 'IC', self.__class__.wrf_features_tb, self.__class__.CONFIG_tb)
        
        self.assertTupleEqual(wrf_tracks.shape, (475, 20))
        self.assertEqual(type(wrf_tracks), geopandas.geodataframe.GeoDataFrame)
        self.assertTrue(np.all(wrf_tracks.longitude!=0))
        self.assertTrue(np.all(wrf_tracks.latitude!=0))
        
        
if __name__ == '__main__':
    unittest.main()
    
    
    
    

# CONFIG = CoMET.CoMET_load('./Example_Configs/boilerplate.yml')
# print('=====Starting WRF dbz Tracking=====')
# wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout_2023_07_09/wrfout_d02*", 'dbz', CONFIG)
# print("*////")
# wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', CONFIG)
# print("**///")
# wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, CONFIG)
# print("***//")
# wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '3D', CONFIG)
# print("****/")
# wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', CONFIG, 2000)
# print('=====Finished WRF dbz Tracking=====')


# print('=====Starting NEXRAD dbz Tracking=====')
# # radar_cube,radar_xarray=CoMET.nexrad_load_netcdf_iris('/D3/data/thahn/NEXRAD/HAS012527906/0003/*_V06.ar2v', 'ar2v', 'dbz', CONFIG, '/D3/data/thahn/NEXRAD/HAS012527906/grids/')
# radar_cube,radar_xarray=CoMET.nexrad_load_netcdf_iris('/D3/data/thahn/NEXRAD/HAS012527906/grids/*', 'nc', 'dbz', CONFIG)
# print("*////")
# radar_features = CoMET.nexrad_tobac_feature_id(radar_cube, 'IC', CONFIG)
# print("**///")
# radar_tracks = CoMET.nexrad_tobac_linking(radar_cube, 'IC', radar_features, CONFIG)
# print("***//")
# radar_segment_array_3d, radar_segment_pd_3d = CoMET.nexrad_tobac_segmentation(radar_cube, 'IC', radar_features, '3D', CONFIG)
# print("****/")
# radar_segment_array_2d, radar_segment_pd_2d = CoMET.nexrad_tobac_segmentation(radar_cube, 'IC', radar_features, '2D', CONFIG, 2000)
# print('=====Finished NEXRAD dbz Tracking=====')


# CONFIG = CoMET.CoMET_load('./Example_Configs/wrf_w_config.yml')
# print('=====Starting WRF w Tracking=====')
# wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'wa')
# print("*////")
# wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', CONFIG)
# print("**///")
# wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, CONFIG)
# print("***//")
# wrf_segment_array_3d, wrf_segment_pd_3d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '3D', CONFIG)
# print("****/")
# wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', CONFIG, 2000)
# print('=====Finished WRF w Tracking=====')


# CONFIG = CoMET.CoMET_load('./Example_Configs/wrf_tb_config.yml')
# print('=====Starting WRF tb Tracking=====')
# wrf_cube, wrf_xarray = CoMET.wrf_load_netcdf_iris("/D3/data/thahn/wrf/wrfout/wrfout_d02*", 'tb')
# print("*///")
# wrf_features = CoMET.wrf_tobac_feature_id(wrf_cube, 'IC', CONFIG)
# print("**//")
# wrf_tracks = CoMET.wrf_tobac_linking(wrf_cube, 'IC', wrf_features, CONFIG)
# print("***/")
# wrf_segment_array_2d, wrf_segment_pd_2d = CoMET.wrf_tobac_segmentation(wrf_cube, 'IC', wrf_features, '2D', CONFIG, None)
# print('=====Finished WRF tb Tracking=====')


# print('=====Starting GOES tb Tracking=====')
# goes_cube, goes_xarray = CoMET.goes_load_netcdf_iris("/D3/data/thahn/GOES/noaa-goes16/ABI-L2-CMIPC/2023/2023_07_11/*", 'tb', CONFIG)
# print("*///")
# goes_features = CoMET.goes_tobac_feature_id(goes_cube, 'IC', CONFIG)
# print("**//")
# goes_tracks = CoMET.goes_tobac_linking(goes_cube, 'IC', goes_features, CONFIG)
# print("***/")
# goes_segment_array_2d, segment_pd_2d = CoMET.goes_tobac_segmentation(goes_cube, 'IC', goes_features, CONFIG)
# print('=====Finished GOES tb Tracking=====')