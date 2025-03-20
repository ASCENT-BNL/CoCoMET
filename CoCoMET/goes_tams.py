import re
import datetime
from dateutil.parser import isoparse

import geopandas as gpd
import xarray as xr

from .TAMS import run

def goes_run_tams(
    goes_xarray: xr.DataArray, 
    CONFIG: dict,
) -> tuple[gpd.GeoDataFrame, tuple]:
    """


    Parameters
    ----------
    goes_xarrayay : xarray.core.dataset.Dataset
        Xarray Dataset containing GOES data calculated from goes_load_netcdf(_iris).
    CONFIG : dict
        Default CoCoMET User CONFIG File.

    Returns
    -------
    ce : geopandas.geodataframe.GeoDataFrame
        A geopandas dataframe with the identified cloud elements.
    latlon_coord_system : tuple
        A tuple of the latitude and longitude coordinate arrays.

    """

    # make a copy of RAMS dataset to configure it to an acceptable format for TAMS
    goes_for_tams_copy = xr.Dataset({})

    tb = xr.DataArray(
        goes_xarray.values,
        dims=["Time", "south_north", "west_east"],
    )

    goes_for_tams_copy["ctt"] = tb.assign_attrs(
        {"long_name": "Brightness temperature", "units": "K"}
    )

    start_date = re.sub( "^\\D*(\\d)", "\\1", goes_xarray["t"].attrs['units'] )

    start_datetime = isoparse(start_date)
    start_year = start_datetime.year
    start_month = start_datetime.month
    start_day = start_datetime.day

    start_hour = start_datetime.hour
    start_minute = start_datetime.minute
    start_second = start_datetime.second

    time = []
    for minutes_since in goes_xarray["t"].values:

        new_temp_minute = int(start_minute + minutes_since)
        new_hour = int(start_hour + new_temp_minute // 60)
        new_second = int(start_second + new_temp_minute % 1 )
        new_minute = new_temp_minute % 60 // 1

        new_time = datetime.datetime(start_year, start_month, start_day, new_hour, new_minute, new_second)
        time.append(new_time)
    

    # make the coordinates and rename the dimensions
    goes_for_tams_copy = goes_for_tams_copy.rename(
        {"Time": "time", "south_north": "lat", "west_east": "lon"}
    )  # rename the dimensions

    goes_for_tams_copy = goes_for_tams_copy.assign_coords(time=(["time"], time))
    goes_for_tams_copy = goes_for_tams_copy.assign_coords(
        lat=(["lat", "lon"], goes_xarray["lat"].data)
    )

    goes_for_tams_copy = goes_for_tams_copy.assign_coords(
        lon=(["lat", "lon"], goes_xarray["lon"].data)
    )

    # Had to manually return the lat/lon coordinate system to perform projections for analysis object
    ce, latlon_coord_system_2d = run(goes_for_tams_copy, **CONFIG["goes"]["tams"])

    # the latitude longitude coordinate systems are only in dimesions lat, lon. Add time dimension to them
    total_time = len(goes_for_tams_copy["time"])
    lat_coord_sys = latlon_coord_system_2d[0].expand_dims(dim={"time" : range(total_time)})
    lon_coord_sys = latlon_coord_system_2d[1].expand_dims(dim={"time" : range(total_time)})
    latlon_coord_system = (lat_coord_sys, lon_coord_sys)
    
    return ce, latlon_coord_system
