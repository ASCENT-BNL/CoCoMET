#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:19:26 2024

@author: thahn
"""

# =============================================================================
# Takes in a filepath containing WRF netCDF data and converts it to a netcdf dataset and/or an iris cube for use in trackers
# =============================================================================


"""
Inputs: 
    fid: xarray dataset of WRF data

Outputs:
    TB: numpy array containing brightness temperature at each point and time--same dimension as input
"""
def cal_TB(fid):
    import numpy as np
    
    OLR = fid['OLR'].values
    
    TB = np.empty(OLR.shape)
    
    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8 # W m^-2 K^-4
    
    for tt,ix,iy in np.ndindex(OLR.shape):
        tf = (OLR[tt,ix,iy]/sigma)**.25
        TB[tt,ix,iy] = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
        
    return(TB)



"""
Inputs: 
    filepath: path to wrfout files (i.e. ./data/wrfout/wrfout_d03_*), works with * delimintator
    trackingVar: ['dbz','tb','w'], variable which is going to be used for tracking--either reflectivity, brightness temperature, or updraft velocity

Outputs:
    cube: iris cube containing either reflectivity or brightness temperature values
    wrf_netcdf: xarray dataset containing merged WRF data
"""
def wrf_load_netcdf_iris(filepath, tracking_var, CONFIG):
    import glob
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset
    from wrf import getvar, ALL_TIMES
    # DELETE THIS BEFORE TESTING
    # from wrfcube import load
    from .wrfcube import load
    
    file_names = np.sort(glob.glob(filepath))
    wrf_netcdf = [Dataset(f) for f in file_names]

    wrf_xarray = xr.open_mfdataset(filepath, coords='all', concat_dim='Time', combine='nested')
    
    if (tracking_var.lower() == 'dbz'):
        wrf_reflectivity = getvar(wrf_netcdf, 'dbz', timeidx=ALL_TIMES, method='cat', squeeze=False)

        wrf_xarray['DBZ'] = wrf_reflectivity
        # Change projection to a string as some trackers do not work well with non-string objects
        wrf_xarray['DBZ'].attrs['projection']=str(wrf_xarray['DBZ'].attrs['projection'])
        cube = load(wrf_xarray,'DBZ')
        
        # Remove superflous dimensions
        cube.remove_coord('Time')
        cube.remove_coord('datetime')
        
        # add correct altitude based off of average height at each height index
        ht = getvar(wrf_netcdf, "z", units="m")
        
        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord('altitude').points = correct_alts
        
    elif (tracking_var.lower() == 'tb'):
        # Brightness temperature is only 2d so no heights needed
        wrf_xarray['TB'] = (['Time','south_north','west_east'],cal_TB(wrf_xarray))
        wrf_xarray['TB'].attrs['units'] = 'K'
        cube = load(wrf_xarray,'TB')
        
    elif (tracking_var.lower() == 'w'):
        # Get updraft velocity at mass points
        wrf_wa = getvar(wrf_netcdf, 'wa', timeidx=ALL_TIMES, method='cat', squeeze=False)
        
        wrf_xarray['WA'] = wrf_wa
        # Change projection to a string as some trackers do not work well with non-string objects
        wrf_xarray['WA'].attrs['projection']=str(wrf_xarray['WA'].attrs['projection'])
        cube = load(wrf_xarray,'WA')
        
        # Remove superflous dimensions
        cube.remove_coord('Time')
        cube.remove_coord('datetime')
        
        # Add correct altitude based off of average height at each height index staggered for winds***
        ht = getvar(wrf_netcdf, "z", units="m")
        
        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord('altitude').points = correct_alts
    
    else:
        raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
        return
    
    # Free memory
    del wrf_netcdf
      
    return ((cube,wrf_xarray))



"""
Inputs:
    filepath: path to wrfout files (i.e. ./data/wrfout/wrfout_d03_*), works with * delimintator
    trackingVar: ['dbz','tb','w'], variable which is going to be used for tracking--either reflectivity, brightness temperature, or updraft velocity

Outputs:sudo snap install outlook-for-linux --edge
    wrf_netcdf: xarray dataset containing merged WRF data
"""
def wrf_load_netcdf(filepath, tracking_var, CONFIG):
    import glob
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset
    from wrf import getvar, ALL_TIMES

    
    file_names = np.sort(glob.glob(filepath))

    wrf_xarray = xr.open_mfdataset(filepath, coords='all', concat_dim='Time', combine='nested')
    
    
    # Does the same thing as the above function without forming the data into iris cubes. For use in future trackers and when tobac depreciates iris cubes.
    if (tracking_var.lower() == 'dbz'):
        wrf_netcdf = [Dataset(f) for f in file_names]
        
        wrf_reflectivity = getvar(wrf_netcdf, 'dbz', timeidx=ALL_TIMES, method='cat', squeeze=False)
        
        wrf_xarray['DBZ'] = wrf_reflectivity
        wrf_xarray['DBZ'].attrs['projection']=str(wrf_xarray['DBZ'].attrs['projection'])
        
        # Free memory
        del wrf_netcdf
        
    elif (tracking_var.lower() == 'tb'):
        wrf_xarray['TB'] = (['Time','south_north','west_east'],cal_TB(wrf_xarray))
        wrf_xarray['TB'].attrs['units'] = 'K'
        
    elif (tracking_var.lower() == 'w'):
        wrf_netcdf = [Dataset(f) for f in file_names]
        
        # Get updraft velocity at mass points
        wrf_wa = getvar(wrf_netcdf, 'wa', timeidx=ALL_TIMES, method='cat', squeeze=False)


        wrf_xarray['WA'] = wrf_wa
        wrf_xarray['WA'].attrs['projection']=str(wrf_xarray['WA'].attrs['projection'])
        
        # Free memory
        del wrf_netcdf
        
    else:
        raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
        return
    
    return (wrf_xarray)


