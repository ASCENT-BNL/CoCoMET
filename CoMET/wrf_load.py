#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:19:26 2024

@author: thahn
"""

# =============================================================================
# Takes in a filepath containing WRF netCDF data and converts it to a netcdf dataset and/or an iris cube for use in trackers
# =============================================================================

import cftime
import iris.cube
import numpy as np
import xarray as xr

from .wrf_calculate_products import (
    wrf_calculate_agl_z,
    wrf_calculate_brightness_temp,
    wrf_calculate_precip_rate,
    wrf_calculate_reflectivity,
    wrf_calculate_wa,
)
from .wrfcube import load


def wrf_load_netcdf_iris(
    filepath: str, tracking_var: str, CONFIG: dict
) -> tuple[iris.cube.Cube, xr.Dataset]:
    """


    Parameters
    ----------
    filepath : str
        Glob style path to wrfout files (i.e. ./data/wrfout/wrfout_d03_*).
    tracking_var : str
        ["dbz","tb","wa","pr", "..."], variable which is going to be used for tracking.
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if missing wrf field or invalid tracking variable entered.

    Returns
    -------
    cube : iris.cube.Cube
        Iris cube containing tracking variable data.
    wrf_xarray : xarray.core.dataset.Dataset
        Xarray dataset containing merged WRF data.

    """

    wrf_xarray = xr.open_mfdataset(
        filepath, coords="all", concat_dim="Time", combine="nested"
    )

    if "wrf" in CONFIG:

        # Check for idealized data then correct times
        if "is_idealized" in CONFIG["wrf"]:
            if CONFIG["wrf"]["is_idealized"]:
                # Update Times
                # Get time differences in minutes
                time_diffs = (
                    np.diff(wrf_xarray.XTIME.values)
                    .astype("timedelta64[m]")
                    .astype("int")
                )

                time_list = [0]

                for diff in time_diffs:
                    time_list.append(time_list[-1] + diff)

                # Initialize simulation at January 1, 2000 for convinence
                time_list = cftime.num2date(
                    time_list, units="minutes since 2000-01-01 00:00:00"
                )
                wrf_xarray.assign_coords(XTIME=("Time", time_list))
                wrf_xarray["XTIME"] = wrf_xarray["XTIME"].assign_attrs(
                    {"description": "minutes since 2000-01-01 00:00:00"}
                )

        # Subset time based on user inputs
        if "min_frame_index" in CONFIG["wrf"] or "max_frame_index" in CONFIG["wrf"]:
            min_frame = (
                CONFIG["wrf"]["min_frame_index"]
                if "min_frame_index" in CONFIG["wrf"]
                else 0
            )
            max_frame = (
                CONFIG["wrf"]["max_frame_index"] + 1
                if "max_frame_index" in CONFIG["wrf"]
                else wrf_xarray.dims["Time"]
            )

            wrf_xarray = wrf_xarray.isel(
                Time=np.arange(
                    min_frame,
                    max_frame,
                ),
                drop=True,
            )

    else:
        raise Exception('!=====CONFIG Missing "wrf" Field=====!')

    # Add projection x and y coordinates to WRF
    proj_y_values = wrf_xarray.DY * (
        np.arange(0, wrf_xarray.south_north.shape[0]) + 0.5
    )
    proj_x_values = wrf_xarray.DX * (np.arange(0, wrf_xarray.west_east.shape[0]) + 0.5)

    wrf_xarray["PROJY"] = ("south_north", proj_y_values)
    wrf_xarray["PROJX"] = ("west_east", proj_x_values)

    if tracking_var.lower() == "dbz":

        wrf_reflectivity = wrf_calculate_reflectivity(wrf_xarray)

        wrf_xarray["DBZ"] = wrf_reflectivity
        cube = load(wrf_xarray, "DBZ")

        # add correct altitude based off of average height at each height index
        ht = wrf_calculate_agl_z(wrf_xarray)

        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord("altitude").points = correct_alts

        # Add altitude field for easier processing later
        wrf_xarray["DBZ"] = wrf_xarray["DBZ"].assign_coords(
            altitude=("bottom_top", correct_alts)
        )

    elif tracking_var.lower() == "tb":

        # Brightness temperature is only 2d so no heights needed
        wrf_xarray["TB"] = (
            ["Time", "south_north", "west_east"],
            wrf_calculate_brightness_temp(wrf_xarray),
        )
        wrf_xarray["TB"].attrs["units"] = "K"

        # Adjust dask chunks
        wrf_xarray["TB"] = wrf_xarray["TB"].chunk(wrf_xarray["OLR"].chunksizes)

        cube = load(wrf_xarray, "TB")

    elif tracking_var.lower() == "wa":

        # Get updraft velocity at mass points
        wrf_wa = wrf_calculate_wa(wrf_xarray)

        wrf_xarray["WA"] = wrf_wa
        cube = load(wrf_xarray, "WA")

        # Add correct altitude based off of average height at each height index
        ht = wrf_calculate_agl_z(wrf_xarray)

        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord("altitude").points = correct_alts

        # Add altitude field for easier processing later
        wrf_xarray["WA"] = wrf_xarray["WA"].assign_coords(
            altitude=("bottom_top", correct_alts)
        )

    elif tracking_var.lower() == "pr":

        # Precipitation rate is only 2d so no heights needed
        wrf_xarray["PR"] = (
            ["Time", "south_north", "west_east"],
            wrf_calculate_precip_rate(wrf_xarray),
        )
        wrf_xarray["PR"].attrs["units"] = "mm/hr"

        # Adjust dask chunks
        wrf_xarray["PR"] = wrf_xarray["PR"].chunk(wrf_xarray["RAINC"].chunksizes)

        cube = load(wrf_xarray, "PR")

    else:
        # If not any of the above, try using user inputed value
        try:

            var_values = wrf_xarray[tracking_var.upper()]
            cube = load(wrf_xarray, tracking_var.upper())

            if len(var_values.shape) == 4:

                # Add correct altitude based off of average height at each height index
                ht = wrf_calculate_agl_z(wrf_xarray)

                correct_alts = [np.mean(h.values) for h in ht]
                cube.coord("altitude").points = correct_alts

                # Add altitude field for easier processing later
                wrf_xarray[tracking_var.upper()] = wrf_xarray[
                    tracking_var.upper()
                ].assign_coords(altitude=("bottom_top", correct_alts))

        except:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )

    return (cube, wrf_xarray.unify_chunks())


def wrf_load_netcdf(filepath: str, tracking_var: str, CONFIG: dict) -> xr.Dataset:
    """


    Parameters
    ----------
    filepath : str
        Glob style path to wrfout files (i.e. ./data/wrfout/wrfout_d03_*).
    tracking_var : str
        ["dbz","tb","wa","pr", "..."], variable which is going to be used for tracking.
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if missing wrf field or invalid tracking variable entered.

    Returns
    -------
    wrf_xarray : xarray.core.dataset.Dataset
        Xarray dataset containing merged WRF data.

    """

    wrf_xarray = xr.open_mfdataset(
        filepath, coords="all", concat_dim="Time", combine="nested"
    )

    if "wrf" in CONFIG:

        if "is_idealized" in CONFIG["wrf"]:
            if CONFIG["wrf"]["is_idealized"]:
                # Update Times
                # Get time differences in minutes
                time_diffs = (
                    np.diff(wrf_xarray.XTIME.values)
                    .astype("timedelta64[m]")
                    .astype("int")
                )

                time_list = [0]

                for diff in time_diffs:
                    time_list.append(time_list[-1] + diff)

                # Initialize simulation at January 1, 2000 for convinence
                time_list = cftime.num2date(
                    time_list, units="minutes since 2000-01-01 00:00:00"
                )
                wrf_xarray.assign_coords(XTIME=("Time", time_list))
                wrf_xarray["XTIME"] = wrf_xarray["XTIME"].assign_attrs(
                    {"description": "minutes since 2000-01-01 00:00:00"}
                )

        # Subset time based on user inputs
        if "min_frame_index" in CONFIG["wrf"] or "max_frame_index" in CONFIG["wrf"]:
            min_frame = (
                CONFIG["wrf"]["min_frame_index"]
                if "min_frame_index" in CONFIG["wrf"]
                else 0
            )
            max_frame = (
                CONFIG["wrf"]["max_frame_index"] + 1
                if "max_frame_index" in CONFIG["wrf"]
                else wrf_xarray.dims["Time"]
            )

            wrf_xarray = wrf_xarray.isel(
                Time=np.arange(
                    min_frame,
                    max_frame,
                ),
                drop=True,
            )

    else:
        raise Exception('!=====CONFIG Missing "wrf" Field=====!')

    # Add projection x and y coordinates to WRF
    proj_y_values = wrf_xarray.DY * (
        np.arange(0, wrf_xarray.south_north.shape[0]) + 0.5
    )
    proj_x_values = wrf_xarray.DX * (np.arange(0, wrf_xarray.west_east.shape[0]) + 0.5)

    wrf_xarray["PROJY"] = ("south_north", proj_y_values)
    wrf_xarray["PROJX"] = ("west_east", proj_x_values)

    # Does the same thing as the above function without forming the data into iris cubes. For use in future trackers and when tobac depreciates iris cubes.
    if tracking_var.lower() == "dbz":

        wrf_reflectivity = wrf_calculate_reflectivity(wrf_xarray)

        wrf_xarray["DBZ"] = wrf_reflectivity

        # Add correct altitude based off of average height at each height index
        ht = wrf_calculate_agl_z(wrf_xarray)

        correct_alts = [np.mean(h.values) for h in ht]

        # Add altitude field for easier processing later
        wrf_xarray["DBZ"] = wrf_xarray["DBZ"].assign_coords(
            altitude=("bottom_top", correct_alts)
        )

    elif tracking_var.lower() == "tb":

        wrf_xarray["TB"] = (
            ["Time", "south_north", "west_east"],
            wrf_calculate_brightness_temp(wrf_xarray),
        )
        wrf_xarray["TB"].attrs["units"] = "K"

        # Adjust dask chunks
        wrf_xarray["TB"] = wrf_xarray["TB"].chunk(wrf_xarray["OLR"].chunksizes)

    elif tracking_var.lower() == "wa":

        # Get updraft velocity at mass points
        wrf_wa = wrf_calculate_wa(wrf_xarray)

        wrf_xarray["WA"] = wrf_wa

        # Add correct altitude based off of average height at each height index
        ht = wrf_calculate_agl_z(wrf_xarray)

        correct_alts = [np.mean(h.values) for h in ht]

        # Add altitude field for easier processing later
        wrf_xarray["WA"] = wrf_xarray["WA"].assign_coords(
            altitude=("bottom_top", correct_alts)
        )

    else:
        # If not any of the above, try using user inputed value
        try:

            var_values = wrf_xarray[tracking_var.upper()]

            if len(var_values.shape) == 4:

                # Add correct altitude based off of average height at each height index
                ht = wrf_calculate_agl_z(wrf_xarray)

                correct_alts = [np.mean(h.values) for h in ht]

                # Add altitude field for easier processing later
                wrf_xarray[tracking_var.upper()] = wrf_xarray[
                    tracking_var.upper()
                ].assign_coords(altitude=("bottom_top", correct_alts))

        except:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )

    return wrf_xarray.unify_chunks()
