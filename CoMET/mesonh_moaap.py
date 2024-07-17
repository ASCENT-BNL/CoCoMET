#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:14:14 2024

@author: thahn
"""

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
def mesonh_moaap(mesonh_xarray, CONFIG):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from .mesonh_calculate_products import mesonh_calculate_brightness_temp
    from CoMET.MOAAP import moaap
    
    # Get basic setup variables including lat/lon, delta time, a pandas time range vector (TODO: adjust output to )
    latitudes = mesonh_xarray.lat[0].values
    longitudes = mesonh_xarray.lon[0].values
    dt = np.median(np.diff(mesonh_xarray.time).astype("timedelta64[m]")).astype(float)
    times = pd.date_range(start=mesonh_xarray.time[0].values,end=mesonh_xarray.time[-1].values,freq=str(dt)+"min")
    mask = np.ones(latitudes.shape)
    
    
    # TODO: Find out how to calculate geopotential heights and accumulated precipitation
    # Get all necessary variables from WRF output to input into MOAAP
    
    # Get pressure heights
    total_p = mesonh_xarray.P
    avg_geo_pres = [np.mean(h.values) for h in total_p[0]]
    
    # Get height_idx of 850hPA, 500hPA, and 200hPA
    height_idx_850 = find_nearest(avg_geo_pres, 850)
    # height_idx_500 = find_nearest(avg_geo_pres, 500)
    height_idx_200 = find_nearest(avg_geo_pres, 200)
    
    # get 850hPA wind speeds and 200hPA wind speeds
    v_winds_850 = mesonh_xarray.v[:, height_idx_850].values
    u_winds_850 = mesonh_xarray.u[:, height_idx_850].values
    v_winds_200 = mesonh_xarray.v[:, height_idx_200].values
    u_winds_200 = mesonh_xarray.u[:, height_idx_200].values
    
    # Get 850hPA air temperature   
    air_temp = mesonh_xarray.T[:,height_idx_850].values
    
    # Get brightness temp
    tb = mesonh_xarray.TB.values if "TB" in mesonh_xarray else mesonh_calculate_brightness_temp(mesonh_xarray)
    
    # Get total mixing ratio at 850hPA
    mr = (mesonh_xarray.qv + mesonh_xarray.qc + mesonh_xarray.qr + mesonh_xarray.qi + mesonh_xarray.qs + mesonh_xarray.qg)[:,height_idx_850]
    
    # Get geopotential heights
    # geopt = 
    
    # Get accumulated precipitation
    # pr = (wrf_xarray.RAINC+wrf_xarray.RAINNC)
    
    moaap(longitudes,
          latitudes,
          times,
          dt/60,
          mask,
          DataName="CoMET_MesoNH_MOAAP_TRACKING",
          OutputFolder=CONFIG["mesonh"]["moaap"]["tracking_save_path"],
          # Data Variables
          v850=v_winds_850,
          u850=u_winds_850,
          t850=air_temp,
          q850=mr,
          slp=mesonh_xarray.SLP*100,
          ivte=None,
          ivtn=None,
          z500=None,
          v200=v_winds_200,
          u200=u_winds_200,
          pr=None,
          tb=tb,
          
          # Any user defined params
          **CONFIG["mesonh"]["moaap"]
          )


    output_filepath = CONFIG["mesonh"]["moaap"]["tracking_save_path"] + str(times[0].year)+str(times[0].month).zfill(2)+'_CoMET_MesoNH_MOAAP_TRACKING_ObjectMasks_dt-%.2f' % dt + 'min_MOAAP-masks'+'.nc'
    mask_file = xr.open_mfdataset(output_filepath, coords="all", concat_dim="time", combine="nested")
    
    return(mask_file)