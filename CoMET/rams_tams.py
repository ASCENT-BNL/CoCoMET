def rams_run_tams(rams_xarray, CONFIG):
    """
    Inputs:
        rams_xarray: xarray Dataset containing RAMS data calculated from rams_load.py
        CONFIG: User configuration file
    Outputs:
        ce: a geopandas dataframe with the identified cloud elements
        latlon_coord_system: a tuple of the latitude and longitude coordinate arrays
    """
    import xarray as xr
    import numpy as np
    import datetime
    
    from .rams_calculate_products import (
        rams_calculate_brightness_temp,
        rams_calculate_precip_rate
    )
    from .TAMS import run

    # make a copy of RAMS dataset to configure it to an acceptable format for TAMS
    rams_for_tams_copy = xr.Dataset({})

    # if brightness temperature is already in rams_xarray use it
    if "TB" not in rams_xarray:
        tb = rams_calculate_brightness_temp(rams_xarray)
        rams_for_tams_copy['ctt'] = tb.assign_attrs(
            { "long_name":  "Brightness temperature",
            "units":      "K"}
        )
        rams_for_tams_copy['ctt'].chunk(rams_xarray["TOPT"].chunksizes)
    else:
        tb = rams_xarray['TB']

    # if precipitation rate is already in rams_xarray use it
    if "PR" not in rams_xarray:
        pr = rams_calculate_precip_rate(rams_xarray)
        rams_for_tams_copy['pr'] = pr.assign_attrs(
            {"long_name":  "Precipitation rate",
            "units":      "mm h-1"}
        )
        rams_for_tams_copy['pr'].chunk(rams_xarray["TOPT"].chunksizes)
    else:
        pr = rams_xarray['PR']

    # format the times into a list of datetime objects
    dt = rams_xarray.DT
    init_date_str_unformatted = rams_xarray.date[-19:]
    datetime_start_date = datetime.datetime.strptime(init_date_str_unformatted, '%Y-%m-%d %H:%M:%S')
    time = []
    for t in rams_xarray.Time.values:
        ds = t * dt
        change = datetime.timedelta(seconds=ds)

        time.append(datetime_start_date + change)

    # make the coordinates and rename the dimensions
    rams_for_tams_copy = rams_for_tams_copy.assign_coords(time=(["Time"], time))
    rams_for_tams_copy = rams_for_tams_copy.assign_coords(lat=(["Time", "south_north", "west_east"], rams_xarray["GLAT"].data))
    rams_for_tams_copy = rams_for_tams_copy.assign_coords(lon=(["Time", "south_north", "west_east"], rams_xarray["GLON"].data))

    rams_for_tams_copy = rams_for_tams_copy.rename({"Time" : "time", "south_north" : "lat", "west_east" : "lon"}) # rename the dimensions

    ce, latlon_coord_system = run(
        rams_for_tams_copy,
        **CONFIG["rams"]["tams"]
    )

    return ce, latlon_coord_system