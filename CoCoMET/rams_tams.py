import datetime

import geopandas as gpd
import xarray as xr

from .rams_calculate_products import (
    rams_calculate_brightness_temp,
    rams_calculate_precip_rate,
)
from .rams_load import rams_load_netcdf_iris
from .TAMS import run


def rams_run_tams(
    rams_xarray: xr.Dataset, CONFIG: dict
) -> tuple[gpd.GeoDataFrame, tuple]:
    """


    Parameters
    ----------
    rams_xarray : xarray.core.dataset.Dataset
        Xarray Dataset containing RAMS data calculated from rams_load_netcdf(_iris).
    CONFIG : dict
        Default CoCoMET CONFIG file.

    Returns
    -------
    ce : geopandas.geodataframe.GeoDataFrame
        A geopandas dataframe with the identified cloud elements.
    latlon_coord_system : tuple
        A tuple of the latitude and longitude coordinate arrays.

    """

    # make a copy of RAMS dataset to configure it to an acceptable format for TAMS
    rams_for_tams_copy = xr.Dataset({})

    # if brightness temperature is already in rams_xarray use it
    if "TB" not in rams_xarray:
        tb = rams_calculate_brightness_temp(rams_xarray)

    else:
        tb = rams_xarray["TB"]

    rams_for_tams_copy["ctt"] = tb.assign_attrs(
        {"long_name": "Brightness temperature", "units": "K"}
    )
    rams_for_tams_copy["ctt"].chunk(rams_xarray["TOPT"].chunksizes)
    rams_xarray = rams_xarray.assign(TB=rams_for_tams_copy["ctt"])

    # # if precipitation rate is already in rams_xarray use it
    # if "PR" not in rams_xarray:
    #     pr = rams_load_netcdf_iris(CONFIG["rams"]["path_to_data"], "PR", CONFIG["rams"]["path_to_header"], CONFIG)[1]["PR"] # TODO: it is cumbersome to reload the rams_xarray just to calculate precipitation. This needs major reworking

    # else:
    #     pr = rams_xarray["PR"]

    # rams_for_tams_copy["pr"] = pr.assign_attrs(
    #     {"long_name": "Precipitation rate", "units": "mm h-1"}
    # )
    # rams_for_tams_copy["pr"].chunk(rams_xarray["TOPT"].chunksizes)

    # format the times into a list of datetime objects
    dt = rams_xarray.DT
    init_date_str_unformatted = rams_xarray.date[-19:]
    datetime_start_date = datetime.datetime.strptime(
        init_date_str_unformatted, "%Y-%m-%d %H:%M:%S"
    )
    time = []
    for t in rams_xarray.Time.values:
        ds = t * dt
        change = datetime.timedelta(seconds=ds)

        time.append(datetime_start_date + change)

    # make the coordinates and rename the dimensions
    rams_for_tams_copy = rams_for_tams_copy.assign_coords(time=(["Time"], time))
    rams_for_tams_copy = rams_for_tams_copy.assign_coords(
        lat=(["Time", "south_north", "west_east"], rams_xarray["GLAT"].data)
    )
    rams_for_tams_copy = rams_for_tams_copy.assign_coords(
        lon=(["Time", "south_north", "west_east"], rams_xarray["GLON"].data)
    )

    rams_for_tams_copy = rams_for_tams_copy.rename(
        {"Time": "time", "south_north": "lat", "west_east": "lon"}
    )  # rename the dimensions

    ce, latlon_coord_system = run(rams_for_tams_copy, **CONFIG["rams"]["tams"])

    return (ce, latlon_coord_system)
