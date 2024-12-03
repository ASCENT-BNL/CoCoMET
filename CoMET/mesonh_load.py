#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:01:16 2024

@author: thahn
"""

# =============================================================================
# Takes in a filepath containing WRF netCDF data and converts it to a netcdf dataset and/or an iris cube for use in trackers
# =============================================================================

import glob
import os

import iris
import iris.cube
import numpy as np
import vincenty
import xarray as xr
import datetime

from .mesonh_calculate_products import (
    mesonh_calculate_agl_z,
    mesonh_calculate_brightness_temp,
    mesonh_calculate_reflectivity,
)
from .mesonhcube import load


def guess_horizontal_spacing(
    mesonh_xarray: xr.Dataset, filename: str
) -> tuple[float, float]:
    """
    This functions attempts to find the horizontal spacing of MesoNH data

    Parameters
    ----------
    mesonh_xarray : xarray.core.dataset.Dataset
        Standard mesonh xarray dataset.
    filename : str
        The name of a mesonh input file.

    Returns
    -------
    dy : float
        Estimated y spacing (m).
    dx : float
        Estimated x spacing (m).
    """

    try:

        # Try to guess dimension from file name
        dis_value = filename.split("m")[0]

        # If is in kilometers, convert to meters
        if "k" in dis_value:

            dis_value = float(dis_value.replace("k", "")) * 1000

        else:

            dis_value = float(dis_value)

        return (dis_value, dis_value)

    except ValueError:
        print(
            "!=====Non-Default MesoNH Filename Found, Estimating Distance Instead=====!"
        )

        # Guess x dimension by finding distance between two points offset by one x value
        x_dis = (
            vincenty.vincenty(
                (mesonh_xarray.lat[0, 0, 0].values, mesonh_xarray.lon[0, 0, 0].values),
                (mesonh_xarray.lat[0, 0, 1].values, mesonh_xarray.lon[0, 0, 1].values),
            )
            * 1000
        )
        # Guess y dimension by finding distance between two points offset by one y value
        y_dis = (
            vincenty.vincenty(
                (mesonh_xarray.lat[0, 0, 0].values, mesonh_xarray.lon[0, 0, 0].values),
                (mesonh_xarray.lat[0, 1, 0].values, mesonh_xarray.lon[0, 1, 0].values),
            )
            * 1000
        )

        # Round to nearest tens place
        x_dis = 10 * round(x_dis / 10)
        y_dis = 10 * round(y_dis / 10)

        return (y_dis, x_dis)


def mesonh_load_netcdf_iris(
    filepath: str, tracking_var: str, CONFIG: dict
) -> tuple[iris.cube.Cube, xr.Dataset]:
    """


    Parameters
    ----------
    filepath : str
        Glob style path to MesoNH files (i.e. ./data/MesoNH/500m*).
    tracking_var : str
        ["dbz","tb","wa", "pr", ...], variable which is going to be used for tracking
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if entered invalid tracking variable.
        ValueError if the simulation times are not continuous.

    Returns
    -------
    cube : iris.cube.Cube
        Iris cube containing tracking variable.
    mesonh_xarray : xarray.core.dataset.Dataset
        Xarray dataset containing merged MesoNH data.

    """

    # Get one filename for guessing spacing
    filename = [os.path.basename(x) for x in glob.glob(filepath)][0]
    mesonh_xarray = xr.open_mfdataset(
        filepath, coords="all", concat_dim="time", combine="nested"
    )

    if "mesonh" in CONFIG:

        # Subset time based on user inputs
        if (
            "min_frame_index" in CONFIG["mesonh"]
            or "max_frame_index" in CONFIG["mesonh"]
        ):
            min_frame = (
                CONFIG["mesonh"]["min_frame_index"]
                if "min_frame_index" in CONFIG["mesonh"]
                else 0
            )
            max_frame = (
                CONFIG["mesonh"]["max_frame_index"] + 1
                if "max_frame_index" in CONFIG["mesonh"]
                else mesonh_xarray.dims["time"]
            )

            mesonh_xarray = mesonh_xarray.isel(
                time=np.arange(min_frame, max_frame), drop=True
            )

    else:
        raise Exception('!=====CONFIG Missing "mesonh" field=====!')

    # Get the time differences 
    datetime_times = [datetime.datetime.strptime(str(time), '%Y-%m-%dT%H:%M:%S.%f000') for time in mesonh_xarray['time'].values]
    dt_datetimes = np.diff(datetime_times)
    dt_seconds = [s.total_seconds() for s in dt_datetimes]

    if len(np.unique(dt_seconds)) > 1:
        raise ValueError("Simulation times are inconsistent")
    else:
        mesonh_xarray.attrs["DT"] = np.unique(dt_seconds)[0]

    # Correct for 360 degree lat/lon system by subtracting 360 from values exceeding 180 degrees...correction for latitude may not be necessary
    mesonh_xarray = mesonh_xarray.assign_coords(
        lat=mesonh_xarray.lat.where(mesonh_xarray.lat <= 180, lambda lat: lat - 360),
        lon=mesonh_xarray.lon.where(mesonh_xarray.lon <= 180, lambda lon: lon - 360),
    )
    mesonh_xarray = mesonh_xarray.unify_chunks()

    dx, dy = guess_horizontal_spacing(mesonh_xarray, filename)

    # Add projection x and y coordinates to MesoNH
    proj_y_values = dy * (np.arange(0, mesonh_xarray.y.size) + 0.5)
    proj_x_values = dx * (np.arange(0, mesonh_xarray.x.size) + 0.5)

    mesonh_xarray["PROJY"] = ("y", proj_y_values)
    mesonh_xarray["PROJX"] = ("x", proj_x_values)

    if tracking_var.lower() == "dbz":

        mesonh_reflectivity = mesonh_calculate_reflectivity(mesonh_xarray)

        mesonh_xarray["DBZ"] = mesonh_reflectivity
        cube = load(mesonh_xarray, "DBZ", filename)

        # add correct altitude based off of average height at each height index
        ht = mesonh_calculate_agl_z(mesonh_xarray)

        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord("altitude").points = correct_alts

        # Add altitude field for easier processing later
        mesonh_xarray["DBZ"] = mesonh_xarray["DBZ"].assign_coords(
            altitude=("z", correct_alts)
        )

    elif tracking_var.lower() == "tb":

        # Brightness temperature is only 2d so no heights needed
        mesonh_xarray["TB"] = (
            ["time", "y", "x"],
            mesonh_calculate_brightness_temp(mesonh_xarray),
        )
        mesonh_xarray["TB"].attrs["units"] = "K"

        # Adjust dask chunks
        mesonh_xarray["TB"] = mesonh_xarray["TB"].chunk(
            mesonh_xarray["LWup_TOA"].chunksizes
        )
        cube = load(mesonh_xarray, "TB", filename)

    elif tracking_var.lower() == "wa":

        # Get updraft velocity at mass points (maybe?)
        mesonh_wa = mesonh_xarray.w

        mesonh_xarray["WA"] = mesonh_wa
        cube = load(mesonh_xarray, "WA", filename)

        # Add correct altitude based off of average height at each height index
        ht = mesonh_calculate_agl_z(mesonh_xarray)

        correct_alts = [np.mean(h.values) for h in ht]
        cube.coord("altitude").points = correct_alts

        # Add altitude field for easier processing later
        mesonh_xarray["WA"] = mesonh_xarray["WA"].assign_coords(
            altitude=("z", correct_alts)
        )

    elif tracking_var.lower() == "pr":

        # precipitation rate is only 2d so no heights needed
        mesonh_xarray["PR"] = mesonh_xarray["pcp_rate"]

        mesonh_xarray["PR"].attrs["units"] = "mm/hour"

        # Adjust dask chunks
        mesonh_xarray["PR"] = mesonh_xarray["PR"].chunk(
            mesonh_xarray["LWup_TOA"].chunksizes
        )
        cube = load(mesonh_xarray, "PR", filename)

    else:
        # If not any of the above, try using user inputed value
        try:

            var_values = mesonh_xarray[tracking_var]
            cube = load(mesonh_xarray, tracking_var, filename)

            if len(var_values.shape) == 4:

                # add correct altitude based off of average height at each height index
                ht = mesonh_calculate_agl_z(mesonh_xarray)

                correct_alts = [np.mean(h.values) for h in ht]
                cube.coord("altitude").points = correct_alts

                # Add altitude field for easier processing later
                mesonh_xarray[tracking_var] = mesonh_xarray[tracking_var].assign_coords(
                    altitude=("z", correct_alts)
                )

        except:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var}=====!"
            )

    return (cube, mesonh_xarray.unify_chunks())


def mesonh_load_netcdf(filepath: str, tracking_var: str, CONFIG: dict) -> xr.Dataset:
    """


    Parameters
    ----------
    filepath : str
        Glob style path to MesoNH files (i.e. ./data/MesoNH/500m*).
    tracking_var : str
        ["dbz","tb","wa", "pr", ...], variable which is going to be used for tracking
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if entered invalid tracking variable.

    Returns
    -------
    mesonh_xarray : xarray.core.dataset.Dataset
        Xarray dataset containing merged MesoNH data.

    """

    # Get one filename for guessing spacing
    filename = [os.path.basename(x) for x in glob.glob(filepath)][0]

    mesonh_xarray = xr.open_mfdataset(
        filepath, coords="all", concat_dim="time", combine="nested"
    )

    if "mesonh" in CONFIG:

        # Subset time based on user inputs
        if (
            "min_frame_index" in CONFIG["mesonh"]
            or "max_frame_index" in CONFIG["mesonh"]
        ):
            min_frame = (
                CONFIG["mesonh"]["min_frame_index"]
                if "min_frame_index" in CONFIG["mesonh"]
                else 0
            )
            max_frame = (
                CONFIG["mesonh"]["max_frame_index"] + 1
                if "max_frame_index" in CONFIG["mesonh"]
                else mesonh_xarray.dims["time"]
            )

            mesonh_xarray = mesonh_xarray.isel(
                time=np.arange(min_frame, max_frame), drop=True
            )

    else:
        raise Exception('!=====CONFIG Missing "mesonh" field=====!')

    # Correct for 360 degree lat/lon system by subtracting 360 from values exceeding 180 degrees...correction for latitude may not be necessary
    mesonh_xarray = mesonh_xarray.assign_coords(
        lat=mesonh_xarray.lat.where(mesonh_xarray.lat <= 180, lambda lat: lat - 360),
        lon=mesonh_xarray.lon.where(mesonh_xarray.lon <= 180, lambda lon: lon - 360),
    )
    mesonh_xarray = mesonh_xarray.unify_chunks()

    dx, dy = guess_horizontal_spacing(mesonh_xarray, filename)

    # Add projection x and y coordinates to MesoNH
    proj_y_values = dy * (np.arange(0, mesonh_xarray.y.size) + 0.5)
    proj_x_values = dx * (np.arange(0, mesonh_xarray.x.size) + 0.5)

    mesonh_xarray["PROJY"] = ("y", proj_y_values)
    mesonh_xarray["PROJX"] = ("x", proj_x_values)

    if tracking_var.lower() == "dbz":

        mesonh_reflectivity = mesonh_calculate_reflectivity(mesonh_xarray)

        mesonh_xarray["DBZ"] = mesonh_reflectivity

        # add correct altitude based off of average height at each height index
        ht = mesonh_calculate_agl_z(mesonh_xarray)

        correct_alts = [np.mean(h.values) for h in ht]

        # Add altitude field for easier processing later
        mesonh_xarray["DBZ"] = mesonh_xarray["DBZ"].assign_coords(
            altitude=("z", correct_alts)
        )

    elif tracking_var.lower() == "tb":

        # Brightness temperature is only 2d so no heights needed
        mesonh_xarray["TB"] = (
            ["time", "y", "x"],
            mesonh_calculate_brightness_temp(mesonh_xarray),
        )
        mesonh_xarray["TB"].attrs["units"] = "K"

        # Adjust dask chunks
        mesonh_xarray["TB"] = mesonh_xarray["TB"].chunk(
            mesonh_xarray["LWup_TOA"].chunksizes
        )

    elif tracking_var.lower() == "wa":

        # Get updraft velocity at mass points (maybe?)
        mesonh_wa = mesonh_xarray.w

        mesonh_xarray["WA"] = mesonh_wa

        # Add correct altitude based off of average height at each height index
        ht = mesonh_calculate_agl_z(mesonh_xarray)

        correct_alts = [np.mean(h.values) for h in ht]

        # Add altitude field for easier processing later
        mesonh_xarray["WA"] = mesonh_xarray["WA"].assign_coords(
            altitude=("z", correct_alts)
        )

    elif tracking_var.lower() == "pr":

        # Brightness temperature is only 2d so no heights needed
        mesonh_xarray["PR"] = mesonh_xarray["pcp_rate"]

        mesonh_xarray["PR"].attrs["units"] = "mm/unit"

        # Adjust dask chunks
        mesonh_xarray["PR"] = mesonh_xarray["PR"].chunk(
            mesonh_xarray["LWup_TOA"].chunksizes
        )

    else:
        # If not any of the above, try using user inputed value
        try:

            var_values = mesonh_xarray[tracking_var]

            if len(var_values.shape) == 4:

                # add correct altitude based off of average height at each height index
                ht = mesonh_calculate_agl_z(mesonh_xarray)

                correct_alts = [np.mean(h.values) for h in ht]

                # Add altitude field for easier processing later
                mesonh_xarray[tracking_var] = mesonh_xarray[tracking_var].assign_coords(
                    altitude=("bottom_top", correct_alts)
                )

        except:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var}=====!"
            )

    return mesonh_xarray.unify_chunks()
