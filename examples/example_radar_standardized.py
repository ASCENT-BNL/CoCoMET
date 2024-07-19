#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:21:37 2024

@author: thahn
"""

import xarray as xr
import numpy as np

test_radar = xr.open_mfdataset("/share/disk25/data/sgupta/data/RADAR/thiago-grid/dbz_2km_20150528.nc" ,coords="all", concat_dim="time", combine="nested").dbz_2km_20150528.expand_dims(dim={"z":1},axis=1)
test_radar = test_radar.assign_coords()

first_time = test_radar.time.values[0]
test_radar = test_radar.assign_coords(time=(test_radar.time.values),
                                            south_north = ("y0", np.arange(test_radar.shape[3])), west_east = ("x0", np.arange(test_radar.shape[2])),
                                            projection_x_coordinate = ("x0", test_radar.projection_x_coordinate[0].values * 2000), projection_y_coordinate = ("y0", test_radar.projection_y_coordinate[0].values * 2000),
                                            x = ("x0",np.arange(test_radar.shape[2])), y = ("y0", np.arange(test_radar.shape[3])),
                                            model_level_number = ("z", np.arange(test_radar.shape[1])), altitude = ("z", [2000]),z=("z",[2000]),
                                            lat = (["x0", "y0"], test_radar.latitude[0].values), lon = (["x0", "y0"], test_radar.longitude[0].values))


# Adjust dimension names to be standards accepted by iris
test_radar["z"] = test_radar.z.assign_attrs({"standard_name": "altitude", "units": "m"})
test_radar["lat"] = test_radar.lat.assign_attrs({"standard_name": "latitude", "units": "degree_N"})
test_radar["lon"] = test_radar.lon.assign_attrs({"standard_name": "longitude", "units": "degree_E"})
test_radar["projection_x_coordinate"] = test_radar.projection_x_coordinate.assign_attrs({"units": "m"})
test_radar["projection_y_coordinate"] = test_radar.projection_y_coordinate.assign_attrs({"units": "m"})
test_radar = test_radar.swap_dims({"x0":"x","y0":"y"}).drop_vars(["longitude","latitude"])

test_radar = test_radar.rename("reflectivity")
test_radar = test_radar.assign_attrs({ "long_name": "Reflectivity", "units": "dBZ", "standard_name": "equivalent_reflectivity_factor" })
test_radar = test_radar.transpose("time","z","y","x")


test_radar.to_netcdf("/D3/data/thahn/RADAR/proper_dbz_2km_20150528.nc")