#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:19:26 2024

@author: thahn
"""

# =============================================================================
# Takes in a filepath containing WRF netCDF data and converts it to a netcdf dataset and/or an iris cube for use in trackers
# =============================================================================


def wrf_load_netcdf_iris(filepath, tracking_var, CONFIG):
    """
    Inputs:
        filepath: glob style path to wrfout files (i.e. ./data/wrfout/wrfout_d03_*)
        trackingVar: ["dbz","tb","wa"], variable which is going to be used for tracking--either reflectivity, brightness temperature, or updraft velocity

    Outputs:
        cube: iris cube containing either reflectivity, updraft velocity, or brightness temperature values
        wrf_netcdf: xarray dataset containing merged WRF data
    """

    import numpy as np
    import xarray as xr
    from .wrf_calculate_products import (
        wrf_calculate_reflectivity,
        wrf_calculate_agl_z,
        wrf_calculate_wa,
        wrf_calculate_brightness_temp,
        wrf_calculate_precip_rate,
    )
    from .wrfcube import load

    wrf_xarray = xr.open_mfdataset(
        filepath, coords="all", concat_dim="Time", combine="nested"
    )

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
        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )

    return (cube, wrf_xarray.unify_chunks())


def wrf_load_netcdf(filepath, tracking_var, CONFIG):
    """
    Inputs:
        filepath: path to wrfout files (i.e. ./data/wrfout/wrfout_d03_*), works with * delimintator
        trackingVar: ["dbz","tb","wa"], variable which is going to be used for tracking--either reflectivity, brightness temperature, or updraft velocity

    Outputs:sudo snap install outlook-for-linux --edge
        wrf_netcdf: xarray dataset containing merged WRF data
    """

    import numpy as np
    import xarray as xr
    from .wrf_calculate_products import (
        wrf_calculate_reflectivity,
        wrf_calculate_agl_z,
        wrf_calculate_wa,
        wrf_calculate_brightness_temp,
    )

    wrf_xarray = xr.open_mfdataset(
        filepath, coords="all", concat_dim="Time", combine="nested"
    )

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
        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )

    return wrf_xarray.unify_chunks()
