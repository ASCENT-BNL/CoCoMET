def mesonh_run_tams(mesonh_xarray, CONFIG):
    """
    Inputs:
        mesonh_xarray: xarray Dataset containing MesoNH data calculated from mesonh_load.py
        CONFIG: User configuration file
    Outputs:
        ce: a geopandas dataframe with the identified cloud elements
        latlon_coord_system: a tuple of the latitude and longitude coordinate arrays
    """
    import xarray as xr
    import datetime
    import numpy as np
    
    from .mesonh_calculate_products import (
        mesonh_calculate_brightness_temp
        )
    from .TAMS import run

    # make a copy of MesoNH dataset to configure it to an acceptable format for TAMS
    mesonh_for_tams_copy = xr.Dataset({})

    # if brightness temperature is already in mesonh_xarray use it
    if "TB" not in mesonh_xarray:
        tb = xr.DataArray(mesonh_calculate_brightness_temp(mesonh_xarray), dims=['time', 'y', 'x'])
        mesonh_for_tams_copy['ctt'] = tb.assign_attrs(
            { "long_name":  "Brightness temperature",
            "units":      "K"}
        )
        mesonh_for_tams_copy['ctt'].chunk(mesonh_xarray["top"].chunksizes)
    else:
        tb = mesonh_xarray['TB']

    # # if precipitation rate is already in mesonh_xarray use it
    # if "PR" not in mesonh_xarray:
    #     pr = mesonh_calculate_precip_rate(mesonh_xarray)
    #     mesonh_for_tams_copy['pr'] = pr.assign_attrs(
    #         {"long_name":  "Precipitation rate",
    #         "units":      "mm h-1"}
    #     )
    #     mesonh_for_tams_copy['pr'].chunk(mesonh_xarray["TOPT"].chunksizes)
    # else:
    #     pr = mesonh_xarray['PR']

    # format the times into a list of datetime objects
    datetime_times = [datetime.datetime.strptime(str(time), '%Y-%m-%dT%H:%M:%S.%f000') for time in mesonh_xarray['time'].values]

    # make the coordinates and rename the dimensions
    mesonh_for_tams_copy = mesonh_for_tams_copy.assign_coords(time=(["time"], datetime_times))
    mesonh_for_tams_copy = mesonh_for_tams_copy.assign_coords(lat=(["time", "y", "x"], mesonh_xarray["lat"].data))
    mesonh_for_tams_copy = mesonh_for_tams_copy.assign_coords(lon=(["time", "y", "x"], mesonh_xarray["lon"].data))

    mesonh_for_tams_copy = mesonh_for_tams_copy.rename({"y" : "lat", "x" : "lon"}) # rename the dimensions

    ce, latlon_coord_system = run(
        mesonh_for_tams_copy,
        **CONFIG["mesonh"]["tams"]
    )

    return ce, latlon_coord_system