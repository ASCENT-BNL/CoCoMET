#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:40:33 2024

@author: hweiner
"""

# =============================================================================
# Takes in a filepath containing RAMS netCDF data and converts it to a netcdf dataset and/or an iris cube for use in trackers
# =============================================================================

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
    debug: int = 0,
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
    rams_xarray = configure_rams(rams_xarray, path_to_header, CONFIG=CONFIG)

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
        # Brightness temperature is only 2d so no heights needed
        rams_xarray["TB"] = rams_calculate_brightness_temp(rams_xarray)
        rams_xarray["TB"].attrs["units"] = "K"

        rams_xarray["TB"] = rams_xarray["TB"].chunk(rams_xarray["TOA_OLR"].chunksizes)

        cube = load(rams_xarray, "TB")

        if debug == 1.0:
            print(f"""this is the tb output: {rams_xarray["TB"].values}""")

        if debug == 2.0:
            plt.imshow(rams_xarray["TB"].values[0, :, :], origin="lower")
            plt.title("Brightness Temperature in K at t=0")
            plt.colorbar()
            plt.show()

    elif tracking_var.lower() == "pr":
        # Precipitation rate is only 2d so no heights needed
        rams_xarray["PR"] = rams_calculate_precip_rate(rams_xarray)

        rams_xarray["PR"].attrs["units"] = "mm/hr"

        # Adjust dask chunks
        rams_xarray["PR"] = rams_xarray["PR"].chunk(rams_xarray["TOPT"].chunksizes)

        cube = load(rams_xarray, "PR")

        if debug == 1.0:
            print(f"""this is the pr output: {rams_xarray["PR"].values}""")

        if debug == 2.0:
            plt.imshow(rams_xarray["PR"].values[0, :, :], origin="lower")
            plt.title("Precipitation Rate in mm/hr at t = 0")
            plt.colorbar()
            plt.show()

    elif tracking_var.lower() == "dbz":
        rams_reflectivity = rams_calculate_reflectivity(rams_xarray)

        rams_xarray["DBZ"] = rams_reflectivity

        cube = load(rams_xarray, "DBZ")
        cube.coord("altitude").points = rams_xarray["altitudes"].values

        # Add altitude field for easier processing later
        rams_xarray["DBZ"] = rams_xarray["DBZ"].assign_coords(
            altitude=("bottom_top", rams_xarray["altitudes"].values)
        )
        if debug == 1.0:
            print(
                f"""this is the dbz output: {rams_xarray["DBZ"].values} with shape {rams_xarray["DBZ"].shape}"""
            )
        if debug == 2.0:
            from matplotlib import cm
            from matplotlib.colors import ListedColormap

            bottom = cm.get_cmap("Reds", 128)
            top = cm.get_cmap("Blues_r", 128)

            newcolors = np.vstack(
                (top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128)))
            )
            cmap = ListedColormap(newcolors, name="RedBlue")
            z = 44
            y = 250
            plt.figure(figsize=[9, 6])
            varToPlot = rams_xarray["DBZ"].values
            varToPlot[rams_xarray["DBZ"].values <= -10] = np.nan
            plt.imshow(
                varToPlot[0, z, :, :],
                origin="lower",
                cmap=cmap,
                aspect="auto",
                vmin=-10,
                vmax=45,
            )
            plt.title(
                """{rams_xarray["altitudes"].values[z]:.3f}m Slice of Reflectivity in dBz at t=0"""
            )
            plt.colorbar()
            plt.ylabel("Grid points (south_north)")
            plt.xlabel("Grid points (west_east)")
            plt.show()

            fig, ax = plt.subplots(1, figsize=[9, 6])
            im = plt.imshow(
                varToPlot[0, :, y, :],
                origin="lower",
                cmap=cmap,
                aspect="auto",
                vmin=-10,
                vmax=45,
            )
            ax.set_title(
                f"Cross section (x-z plane) of Reflectivity in dBz \n at y (south_north)={y}, t=0"
            )
            ax.set_ylabel("Altitude [m]")
            ax.set_xlabel("Grid points (west_east)")
            plt.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
            ax.set_yticks(
                np.arange(len(rams_xarray["DBZ"].values[0, :, 0, 0])),
                np.round(rams_xarray["altitudes"].values, 2),
            )
            array = np.asarray(rams_xarray["altitudes"].values)
            idx = (np.abs(array - 15000)).argmin()
            ax.set_ylim(0, idx)
            plt.locator_params(axis="y", nbins=8)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=[9, 6])
            plt.imshow(
                varToPlot[0, 0, :, :], origin="lower", cmap=cmap, vmin=-10, vmax=45
            )
            plt.title(
                "Cross section (x-y plane) of Reflectivity in dBz \n at z (bottom_top)=0, t=0"
            )
            plt.ylabel("Grid points (south_north)")
            plt.xlabel("Grid points (west_east)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

    elif tracking_var.lower() == "wa":
        # Get updraft velocity at mass points
        rams_wa = rams_calculate_wa(rams_xarray)

        rams_xarray["WA"] = rams_wa

        cube = load(rams_xarray, "WA")
        cube.coord("altitude").points = rams_xarray["altitudes"].values

        # Add altitude field for easier processing later
        rams_xarray["WA"] = rams_xarray["WA"].assign_coords(
            altitude=("bottom_top", rams_xarray["altitudes"].values)
        )

        if debug == 1:
            print(f"""the wa output is {rams_xarray["WA"].values}""")
        if debug == 2.0:
            z = 44
            t = 0
            plt.imshow(
                rams_xarray["WA"].values[t, z, :, :], origin="lower", cmap="Blues"
            )
            plt.title(
                f"""{rams_xarray["altitudes"].values[z]:.3f}m Slice of Vertical Velocity of Winds in m/s at t={t}"""
            )
            plt.colorbar()
            plt.ylabel("Grid points (south_north)")
            plt.xlabel("Grid points (west_east)")
            plt.show()

            plt.figure(figsize=[10, 10])
            plt.imshow(
                rams_xarray["WA"].values[t, :, 0, :],
                origin="lower",
                cmap="plasma",
                aspect="auto",
            )
            plt.title(
                f"Cross section of Vertical Wind Velocity in m/s \n at y (south_north)=0, t={t} time units"
            )
            plt.ylabel("Altitude [m]")
            plt.xlabel("Grid points (west_east)")
            plt.yticks(
                np.arange(len(rams_xarray["WA"].values[0, :, 0, 0])),
                np.round(rams_xarray["altitudes"].values, 2),
            )
            plt.locator_params(axis="y", nbins=6)
            plt.colorbar()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=[10, 10])
            plt.imshow(
                rams_xarray["WA"].values[t, z, :, :],
                origin="lower",
                cmap="plasma",
                aspect="auto",
            )
            plt.title(
                f"""Cross section of Vertical Wind Velocity in m/s \n at z (bottom_zop)={rams_xarray["altitudes"].values[z]:.3f}m, t={t} time units"""
            )
            plt.ylabel("Grid points (south_north)")
            plt.xlabel("Grid points (west_east)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
    else:
        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )

    return (cube, rams_xarray.unify_chunks())
