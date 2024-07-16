#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:29:56 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running MOAAP on WRF data processed using wrf_load.py
# =============================================================================



# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np
    
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx



"""
Inputs:
    wrf_xarray: xarray Dataset containing WRF data calculated from wrf_load.py
    CONFIG: User configuration file
Outputs:
    TBD
"""
def wrf_moaap(wrf_xarray, CONFIG):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from .wrf_calculate_products import wrf_calculate_brightness_temp
    from CoMET.MOAAP import moaap
    
    # Get basic setup variables including lat/lon, delta time, a pandas time range vector (TODO: adjust output to )
    latitudes = wrf_xarray.XLAT[0].values
    longitudes = wrf_xarray.XLONG[0].values
    dt = np.median(np.diff(wrf_xarray.XTIME).astype("timedelta64[m]")).astype(float)
    times = pd.date_range(start=wrf_xarray.XTIME[0].values,end=wrf_xarray.XTIME[-1].values,freq=str(dt)+"min")
    mask = np.ones(latitudes.shape)
    
    # Get all necessary variables from WRF output to input into MOAAP
    print(1)
    # Get pressure heights
    p = wrf_xarray["P"]
    pb = wrf_xarray["PB"]
    
    total_p = (p+pb)/100
    avg_geo_pres = [np.mean(h.values) for h in total_p[0]]
    print(2)
    # Get height_idx of 850hPA, 500hPA, and 200hPA
    height_idx_850 = find_nearest(avg_geo_pres, 850)
    height_idx_500 = find_nearest(avg_geo_pres, 500)
    height_idx_200 = find_nearest(avg_geo_pres, 200)
    print(3)
    # get destaggered 850hPA wind speeds and 200hPA wind speeds
    v_winds_850 = (0.5*wrf_xarray["V"][:,height_idx_850,1:,:] + 0.5*wrf_xarray["V"][:,height_idx_850,:-1,:]).values
    u_winds_850 = (0.5*wrf_xarray["U"][:,height_idx_850,:,1:] + 0.5*wrf_xarray["U"][:,height_idx_850,:,:-1]).values
    v_winds_200 = (0.5*wrf_xarray["V"][:,height_idx_200,1:,:] + 0.5*wrf_xarray["V"][:,height_idx_200,:-1,:]).values
    u_winds_200 = (0.5*wrf_xarray["U"][:,height_idx_200,:,1:] + 0.5*wrf_xarray["U"][:,height_idx_200,:,:-1]).values
    print(4)
    # Get 850hPA air temperature
    t = wrf_xarray["T"]
    
    # Calculate proper pressures and actual temperature
    full_t = t + 300
    full_p = p + pb
    print(5)
    air_temp = (full_t * (full_p / 1e5)**(287.0/1004.5))[:,height_idx_850].values
    print(6)
    
    # Get geopotential heights
    ph = wrf_xarray["PH"]
    phb = wrf_xarray["PHB"]
    geopt = (ph + phb)
    
    # DESTAGGER geopt
    geopt = 0.5 * geopt[:,1:] + 0.5 * geopt[:,:-1]
    print(7)
    # Get brightness temp
    tb = wrf_calculate_brightness_temp(wrf_xarray)
    print(8)
    
    moaap(longitudes,
          latitudes,
          times,
          dt/60,
          mask,
          DataName="CoMET_WRF_MOAAP_TRACKING",
          OutputFolder=CONFIG["wrf"]["moaap"]["tracking_save_path"],
          # Data Variables
          v850=v_winds_850,
          u850=u_winds_850,
          t850=air_temp,
          q850=wrf_xarray["QVAPOR"][:,height_idx_850],
          slp=full_p[:,0],
          ivte=None,
          ivtn=None,
          z500=geopt.values[:,height_idx_500],
          v200=v_winds_200,
          u200=u_winds_200,
          pr=(wrf_xarray.RAINC+wrf_xarray.RAINNC).values,
          tb=tb,
          
          # Any user defined params
          **CONFIG["wrf"]["moaap"]
          )


    output_filepath = CONFIG["wrf"]["moaap"]["tracking_save_path"] + str(times[0].year)+str(times[0].month).zfill(2)+'_CoMET_WRF_MOAAP_TRACKING_ObjectMasks_dt-%.2f' % dt + 'min_MOAAP-masks'+'.nc'
    mask_file = xr.open_mfdataset(output_filepath, coords="all", concat_dim="time", combine="nested")
    
    return(mask_file)