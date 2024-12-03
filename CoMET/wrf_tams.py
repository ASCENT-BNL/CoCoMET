def wrf_run_tams(wrf_xarray, CONFIG):
    """
    Inputs:
        wrf_xarray: xarray Dataset containing RAMS data calculated from wrf_load.py
        CONFIG: User configuration file
    Outputs:
        ce: a geopandas dataframe with the identified cloud elements
        latlon_coord_system: a tuple of the latitude and longitude coordinate arrays
    """
    import xarray as xr
    import numpy as np
    import datetime
        
    from .wrf_calculate_products import (
        wrf_calculate_brightness_temp,
        wrf_calculate_precip_rate
    )
    from .TAMS import run

    # make a copy of RAMS dataset to configure it to an acceptable format for TAMS
    wrf_for_tams_copy = xr.Dataset({})

    # if brightness temperature is already in wrf_xarray use it
    if "TB" not in wrf_xarray:
        tb = xr.DataArray(wrf_calculate_brightness_temp(wrf_xarray), dims=["Time", "south_north", "west_east"])
        wrf_for_tams_copy['ctt'] = tb.assign_attrs(
            { "long_name":  "Brightness temperature",
            "units":      "K"}
        )
        wrf_for_tams_copy['ctt'].chunk(wrf_xarray["XLAT"].chunksizes)
    else:
        wrf_for_tams_copy['ctt'] = wrf_xarray['TB']

    # if precipitation rate is already in wrf_xarray use it
    if "PR" not in wrf_xarray:
        pr = xr.DataArray(wrf_calculate_precip_rate(wrf_xarray),
                        dims=["Time", "south_north", "west_east"])
        wrf_for_tams_copy['pr'] = pr.assign_attrs(
            {"long_name":  "Precipitation rate",
            "units":      "mm h-1"}
        )
        wrf_for_tams_copy['pr'].chunk(wrf_xarray["XLAT"].chunksizes)

        wrf_for_tams_copy
    else:
        wrf_for_tams_copy['pr'] = wrf_xarray['PR']

    # format the times into a list of datetime objects
    # dt = wrf_xarray.DT
    # start_date = wrf_xarray.SIMULATION_START_DATE
    # datetime_start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d_%H:%M:%S')

    time = []
    for t in wrf_xarray.XTIME.values:
        time_str = t.astype("datetime64[s]").astype(str)
        datetime_time = datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        time.append(datetime_time)

    # make the coordinates and rename the dimensions
    wrf_for_tams_copy = wrf_for_tams_copy.rename({"Time" : "time", "south_north" : "lat", "west_east" : "lon"}) # rename the dimensions

    wrf_for_tams_copy = wrf_for_tams_copy.assign_coords(time=(["time"], time))
    wrf_for_tams_copy = wrf_for_tams_copy.assign_coords(lat=(["time", "lat", "lon"], wrf_xarray["XLAT"].data))
    wrf_for_tams_copy = wrf_for_tams_copy.assign_coords(lon=(["time", "lat", "lon"], wrf_xarray["XLONG"].data))

    # Had to manually return the lat/lon coordinate system to perform projections for analysis object
    ce, latlon_coord_system = run(
        wrf_for_tams_copy,
        **CONFIG["wrf"]["tams"]
    )

    return ce, latlon_coord_system