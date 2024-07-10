#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:01:16 2024

@author: thahn
"""

# =============================================================================
# Takes in a filepath containing WRF netCDF data and converts it to a netcdf dataset and/or an iris cube for use in trackers
# =============================================================================



"""
Inputs: 
    filepath: glob style path to MesoNH files (i.e. ./data/MesoNH/500m*)
    trackingVar: ["dbz","tb","wa"], variable which is going to be used for tracking--either reflectivity, brightness temperature, or updraft velocity

Outputs:
    cube: iris cube containing either reflectivity, updraft velocity, or brightness temperature values
    mesonh_netcdf: xarray dataset containing merged MesoNH data
"""
def mesonh_load_netcdf_iris(filepath, tracking_var, CONFIG):
    import os
    import glob
    import numpy as np
    import xarray as xr
    from .mesonh_calculate_products import mesonh_calculate_reflectivity, mesonh_calculate_msl_z, mesonh_calculate_brightness_temp
    from .mesonhcube import load

    # Get one filename for guessing spacing
    filename = [os.path.basename(x) for x in glob.glob("/D3/data/thahn/MesoNH/CLN/0_5km/*")][0]
    mesonh_xarray = xr.open_mfdataset(filepath, coords="all", concat_dim="time", combine="nested")
    
    # Correct for 360 degree lat/lon system by subtracting 360 from values exceeding 180 degrees...correction for latitude may not be necessary
    mesonh_xarray = mesonh_xarray.assign_coords(lat=mesonh_xarray.lat.where(mesonh_xarray.lat <= 180, lambda lat: lat-360),
                                                lon=mesonh_xarray.lon.where(mesonh_xarray.lon <= 180, lambda lon: lon-360))
    mesonh_xarray = mesonh_xarray.unify_chunks()
    
    
    if (tracking_var.lower() == "dbz"):
        
        mesonh_reflectivity = mesonh_calculate_reflectivity(mesonh_xarray)

        mesonh_xarray["DBZ"] = mesonh_reflectivity
        cube = load(mesonh_xarray,"DBZ",filename)
        
        # add correct altitude based off of average height at each height index
        ht = mesonh_calculate_msl_z(mesonh_xarray)
        
        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord("altitude").points = correct_alts
        
        # Add altitude field for easier processing later
        mesonh_xarray["DBZ"] = mesonh_xarray["DBZ"].assign_coords(altitude = ("z", correct_alts))
        
    elif (tracking_var.lower() == "tb"):
        
        # Brightness temperature is only 2d so no heights needed
        mesonh_xarray["TB"] = (["time","y","x"],mesonh_calculate_brightness_temp(mesonh_xarray))
        mesonh_xarray["TB"].attrs["units"] = "K"
        
        # Adjust dask chunks
        mesonh_xarray["TB"] = mesonh_xarray["TB"].chunk(mesonh_xarray["LWup_TOA"].chunksizes)
        cube = load(mesonh_xarray,"TB",filename)
        
    elif (tracking_var.lower() == "wa"):
        
        # Get updraft velocity at mass points (maybe?)
        mesonh_wa = mesonh_xarray.w
        
        mesonh_xarray["WA"] = mesonh_wa
        cube = load(mesonh_xarray,"WA",filename)
        
        # Add correct altitude based off of average height at each height index
        ht = mesonh_calculate_msl_z(mesonh_xarray)
        
        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord("altitude").points = correct_alts
        
        # Add altitude field for easier processing later
        mesonh_xarray["WA"] = mesonh_xarray["WA"].assign_coords(altitude = ("z", correct_alts))
    
    else:
        raise Exception(f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!")
        return
      
    return ((cube,mesonh_xarray.unify_chunks()))



"""
Inputs: 
    filepath: glob style path to MesoNH files (i.e. ./data/MesoNH/500m*)
    trackingVar: ["dbz","tb","wa"], variable which is going to be used for tracking--either reflectivity, brightness temperature, or updraft velocity

Outputs:
    mesonh_netcdf: xarray dataset containing merged MesoNH data
"""
def mesonh_load_netcdf(filepath, tracking_var, CONFIG):
    import numpy as np
    import xarray as xr
    from .mesonh_calculate_products import mesonh_calculate_reflectivity, mesonh_calculate_msl_z, mesonh_calculate_brightness_temp

    # Get one filename for guessing spacing
    mesonh_xarray = xr.open_mfdataset(filepath, coords="all", concat_dim="time", combine="nested")
    mesonh_xarray = mesonh_xarray.assign_coords(lat=mesonh_xarray.lat,lon=mesonh_xarray.lon)
    mesonh_xarray = mesonh_xarray.unify_chunks()
    
    
    if (tracking_var.lower() == "dbz"):
        
        mesonh_reflectivity = mesonh_calculate_reflectivity(mesonh_xarray)

        mesonh_xarray["DBZ"] = mesonh_reflectivity
        
        # add correct altitude based off of average height at each height index
        ht = mesonh_calculate_msl_z(mesonh_xarray)
        
        correct_alts = [np.mean(h.values) for h in ht]
        
        # Add altitude field for easier processing later
        mesonh_xarray["DBZ"] = mesonh_xarray["DBZ"].assign_coords(altitude = ("z", correct_alts))
        
    elif (tracking_var.lower() == "tb"):
        
        # Brightness temperature is only 2d so no heights needed
        mesonh_xarray["TB"] = (["time","y","x"],mesonh_calculate_brightness_temp(mesonh_xarray))
        mesonh_xarray["TB"].attrs["units"] = "K"
        
        # Adjust dask chunks
        mesonh_xarray["TB"] = mesonh_xarray["TB"].chunk(mesonh_xarray["LWup_TOA"].chunksizes)
        
    elif (tracking_var.lower() == "wa"):
        
        # Get updraft velocity at mass points (maybe?)
        mesonh_wa = mesonh_xarray.w
        
        mesonh_xarray["WA"] = mesonh_wa
        
        # Add correct altitude based off of average height at each height index
        ht = mesonh_calculate_msl_z(mesonh_xarray)
        
        correct_alts = [np.mean(h.values) for h in ht]
        
        # Add altitude field for easier processing later
        mesonh_xarray["WA"] = mesonh_xarray["WA"].assign_coords(altitude = ("z", correct_alts))
    
    else:
        raise Exception(f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!")
        return
      
    return (mesonh_xarray.unify_chunks())
