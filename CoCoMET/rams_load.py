#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:40:33 2024

@author: hweiner
"""

# =============================================================================
# Takes in a filepath containing RAMS netCDF data and converts it to a netcdf dataset and/or an iris cube for use in trackers
# =============================================================================

import logging

import cftime
import iris.cube
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .rams_calculate_products import (
    rams_calculate_brightness_temp,
    rams_calculate_precip_rate,
    rams_calculate_reflectivity,
    rams_calculate_wa,
)
from .rams_configure import configure_rams
from .ramscube import load


# TODO: You should be able to track on any variable in the RAMS dataset, also need to add PR tracking
def rams_load_netcdf_iris(
    filepath: str,
    tracking_var: str,
    path_to_header: str,
    CONFIG: dict = None,
) -> tuple[iris.cube.Cube, xr.Dataset]:
    """


    Parameters
    ----------
    filepath : str
        glob style path to rams files (i.e. ./data/ramsout/ramsout_d03_*.h5).
    tracking_var : str
        ["dbz","tb","wa"], variable which is going to be used for tracking--either reflectivity, brightness temperature, or updraft velocity.
    path_to_header : str
        glob style path to rams header files (i.e. ./data/ramsout/ramsheader_*.txt).
    CONFIG : dict, optional
        Standard CoCoMET CONFIG file. The default is None.
    debug : int, optional
        Debug verbosity level, from 0-2. The default is 0.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    cube : iris.cube.Cube
        iris cube containing either reflectivity, updraft velocity, or brightness temperature values.
    rams_xarrat : xarray.core.dataset.Dataset
        Xarray dataset containing merged rams data.

    """

    rams_xarray = xr.open_mfdataset(
        filepath,
        coords="all",
        concat_dim="Time",
        combine="nested",
        decode_times=False,
        engine="h5netcdf",
        phony_dims="sort",
    )

    # This is already in configure_rams, but eventually we will call RAMS-MAT. Then either put the time formatting here or keep it in RAMS-MAT
    if "rams" in CONFIG:
        # Check for idealized data then correct times
        if "is_idealized" in CONFIG["rams"]:
            if CONFIG["rams"]["is_idealized"]:
                # Update Times
                # Get time differences in minutes
                time_diffs = (
                    np.diff(rams_xarray.Times.values)
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
                rams_xarray.assign_coords(XTIME=("Time", time_list))
                rams_xarray["XTIME"] = rams_xarray["XTIME"].assign_attrs(
                    {"description": "minutes since 2000-01-01 00:00:00"}
                )

        # If there are specified bounds, bound the data
        if "bounds" in CONFIG["rams"]:
            # If it is idealized data, print a warning
            if "is_idealized" in CONFIG["rams"]:
                if CONFIG["rams"]["is_idealized"]:
                    logging.warning("!=====Setting bounds for idealized data=====!")

            mask_lon = (rams_xarray.GLON >= CONFIG["rams"]["bounds"][0]) & (
                rams_xarray.GLON <= CONFIG["rams"]["bounds"][1]
            )
            mask_lat = (rams_xarray.GLAT >= CONFIG["rams"]["bounds"][2]) & (
                rams_xarray.GLAT <= CONFIG["rams"]["bounds"][3]
            )

            rams_xarray = rams_xarray.where(mask_lon & mask_lat, drop=True)

        # Subset time based on user inputs
        if "min_frame_index" in CONFIG["rams"] or "max_frame_index" in CONFIG["rams"]:
            min_frame = (
                CONFIG["rams"]["min_frame_index"]
                if "min_frame_index" in CONFIG["rams"]
                else 0
            )
            max_frame = (
                CONFIG["rams"]["max_frame_index"] + 1
                if "max_frame_index" in CONFIG["rams"]
                else rams_xarray.dims["Time"]
            )

            rams_xarray = rams_xarray.isel(
                Time=np.arange(
                    min_frame,
                    max_frame,
                ),
                drop=True,
            )

    else:
        raise Exception("""!=====CONFIG Missing "rams" Field=====!""")

    if tracking_var.lower() == "tb":

        # Configure rams xarray for brightness temperature
        rams_xarray = configure_rams(
            rams_xarray,
            path_to_header,
            CONFIG=CONFIG,
            configure_variables=["TOA_OLR", "LWUP"],
        )

        # Brightness temperature is only 2d so no heights needed
        rams_xarray["TB"] = rams_calculate_brightness_temp(rams_xarray)
        rams_xarray["TB"].attrs["units"] = "K"

        cube = load(rams_xarray, "TB")

    elif tracking_var.lower() == "pr":

        use_available_variables = False
        # Configure rams xarray for brightness temperature

        # If there is a specified calculation type for precipitation, use that
        if "calculation_type" in CONFIG["rams"]:
            if (
                CONFIG["rams"]["calculation_type"]
                == "surface time averaged precipitation rate"
                or CONFIG["rams"]["calculation_type"] is None
            ):
                rams_xarray = configure_rams(
                    rams_xarray,
                    path_to_header,
                    CONFIG=CONFIG,
                    configure_variables=[
                        "ACCPR",
                        "ACCPP",
                        "ACCPS",
                        "ACCPA",
                        "ACCPG",
                        "ACCPH",
                        "ACCPD",
                        "ACONPR",
                        "ACCPIP",
                        "ACCPIC",
                        "ACCPID",
                    ],
                )
                rams_xarray["PR"] = rams_calculate_precip_rate(
                    rams_xarray, calculation_type=CONFIG["rams"]["calculation_type"]
                )

            elif (
                CONFIG["rams"]["calculation_type"]
                == "surface instantaneous precipitation rate"
            ):
                rams_xarray = configure_rams(
                    rams_xarray,
                    path_to_header,
                    CONFIG=CONFIG,
                    configure_variables=[
                        "PCPRR",
                        "PCPRP",
                        "PCPRS",
                        "PCPRA",
                        "PCPRG",
                        "PCPRH",
                        "PCPRD",
                        "CONPRR",
                        "PCPRIP",
                        "PCPRIC",
                        "PCPRID",
                    ],
                )
                rams_xarray["PR"] = rams_calculate_precip_rate(
                    rams_xarray, calculation_type=CONFIG["rams"]["calculation_type"]
                )

            elif (
                CONFIG["rams"]["calculation_type"]
                == "volumetric instantaneous precipitation rate"
            ):
                rams_xarray = configure_rams(
                    rams_xarray,
                    path_to_header,
                    CONFIG=CONFIG,
                    configure_variables=[
                        "PCPVR",
                        "PCPVP",
                        "PCPVS",
                        "PCPVA",
                        "PCPVG",
                        "PCPVH",
                        "PCPVD",
                        "CONPRR",
                        "PCPVIP",
                        "PCPVIC",
                        "PCPVID",
                    ],
                )
                rams_xarray["PR"] = rams_calculate_precip_rate(
                    rams_xarray, calculation_type=CONFIG["rams"]["calculation_type"]
                )

            else:
                print(
                    "No calculation type found in CONFIG file, using available variables"
                )
                use_available_variables = True

        # If there is no calculation type, see which variables are available and use those
        if "calculation_type" not in CONFIG["rams"] or use_available_variables:
            if "ACCPR" in rams_xarray:
                rams_xarray = configure_rams(
                    rams_xarray,
                    path_to_header,
                    CONFIG=CONFIG,
                    configure_variables=[
                        "ACCPR",
                        "ACCPP",
                        "ACCPS",
                        "ACCPA",
                        "ACCPG",
                        "ACCPH",
                        "ACCPD",
                        "ACONPR",
                        "ACCPIP",
                        "ACCPIC",
                        "ACCPID",
                    ],
                )
                rams_xarray["PR"] = rams_calculate_precip_rate(
                    rams_xarray,
                    calculation_type="surface time averaged precipitation rate",
                )

            elif "PCPRR" in rams_xarray:
                rams_xarray = configure_rams(
                    rams_xarray,
                    path_to_header,
                    CONFIG=CONFIG,
                    configure_variables=[
                        "PCPRR",
                        "PCPRP",
                        "PCPRS",
                        "PCPRA",
                        "PCPRG",
                        "PCPRH",
                        "PCPRD",
                        "CONPRR",
                        "PCPRIP",
                        "PCPRIC",
                        "PCPRID",
                    ],
                )
                rams_xarray["PR"] = rams_calculate_precip_rate(
                    rams_xarray,
                    calculation_type="surface instantaneous precipitation rate",
                )

            elif "PCPVR" in rams_xarray:
                rams_xarray = configure_rams(
                    rams_xarray,
                    path_to_header,
                    CONFIG=CONFIG,
                    configure_variables=[
                        "PCPVR",
                        "PCPVP",
                        "PCPVS",
                        "PCPVA",
                        "PCPVG",
                        "PCPVH",
                        "PCPVD",
                        "CONPRR",
                        "PCPVIP",
                        "PCPVIC",
                        "PCPVID",
                    ],
                )
                rams_xarray["PR"] = rams_calculate_precip_rate(
                    rams_xarray,
                    calculation_type="volumetric surface instantaneous precipitation rate",
                )

        rams_xarray["PR"].attrs["units"] = "mm/hr"

        cube = load(rams_xarray, "PR")

        # If tracking 3D precipitation rate, add altitude to the cube
        if len(rams_xarray["PR"].values.shape) == 4:
            cube.coord("altitude").points = rams_xarray["altitudes"].values

    elif tracking_var.lower() == "dbz":

        # Configure rams xarray for reflectivity
        rams_xarray = configure_rams(
            rams_xarray,
            path_to_header,
            CONFIG=CONFIG,
            configure_variables=[
                "RRP",
                "RPP",
                "RSP",
                "RAP",
                "RGP",
                "RHP",
                "CRP",
                "CPP",
                "CSP",
                "CAP",
                "CGP",
                "CHP",
            ],
        )

        rams_xarray["DBZ"] = rams_calculate_reflectivity(rams_xarray)

        cube = load(rams_xarray, "DBZ")
        cube.coord("altitude").points = rams_xarray[
            "altitudes"
        ].values  # there is already an altitude coordinate in the xarray

        # Add altitude field for easier processing later
        rams_xarray["DBZ"] = rams_xarray["DBZ"].assign_coords(
            altitude=(["bottom_top"], rams_xarray["altitudes"].values)
        )

    elif tracking_var.lower() == "wa":

        # Configure rams xarray for brightness temperature
        rams_xarray = configure_rams(
            rams_xarray, path_to_header, CONFIG=CONFIG, configure_variables=[]
        )

        # Get updraft velocity at mass points
        rams_wa = rams_calculate_wa(rams_xarray)

        rams_xarray["WA"] = rams_wa

        cube = load(rams_xarray, "WA")
        cube.coord("altitude").points = rams_xarray[
            "altitudes"
        ].values  # there is already an altitude coordinate in the xarray

        # Add altitude field for easier processing later
        rams_xarray["WA"] = rams_xarray["WA"].assign_coords(
            altitude=("bottom_top", rams_xarray["altitudes"].values)
        )

    else:
        # If not any of the above, try using user inputed value
        try:
            # Configure rams xarray for brightness temperature
            rams_xarray = configure_rams(
                rams_xarray,
                path_to_header,
                CONFIG=CONFIG,
                configure_variables=[tracking_var.upper()],
            )

            var_values = rams_xarray[tracking_var.upper()]
            cube = load(rams_xarray, tracking_var.upper())

            if len(var_values.shape) == 4:
                # Add correct altitude based off of average height at each height index

                cube.coord("altitude").points = rams_xarray[
                    "altitudes"
                ].values  # there is already an altitude coordinate in the xarray

                # Add altitude field for easier processing later
                rams_xarray[tracking_var.upper()] = rams_xarray[
                    tracking_var.upper()
                ].assign_coords(
                    altitude=("bottom_top", rams_xarray["altitudes"].values)
                )

        except:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )

    return (cube, rams_xarray.unify_chunks())
