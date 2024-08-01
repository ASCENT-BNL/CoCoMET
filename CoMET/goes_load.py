#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 13 11:33:16 2024

@author: thahn
"""

# =============================================================================
# Loads in and converts GOES satellite data to an iris cube and/or an xarray DataArray
# =============================================================================

import warnings

import iris
import iris.cube
import numpy as np
import xarray as xr


# Used to calculate lat and lon values for GOES data which only has proj_x and proj_y values
def calc_latlon(ds: xr.Dataset) -> xr.Dataset:
    """


    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset of the GOES data.

    Returns
    -------
    ds : TYPE
        xarray Dataset of the GOES data with added lat/lon data.

    """

    # The math for this function was taken from
    # https://makersportal.com/blog/2018/11/25/goes-r-satellite-latitude-and-longitude-grid-projection-algorithm
    x = ds.x
    y = ds.y
    goes_imager_projection = ds.goes_imager_projection

    x, y = np.meshgrid(x, y)

    r_eq = goes_imager_projection.attrs["semi_major_axis"]
    r_pol = goes_imager_projection.attrs["semi_minor_axis"]
    l_0 = goes_imager_projection.attrs["longitude_of_projection_origin"] * (np.pi / 180)
    h_sat = goes_imager_projection.attrs["perspective_point_height"]
    H = r_eq + h_sat

    a = np.sin(x) ** 2 + (
        np.cos(x) ** 2 * (np.cos(y) ** 2 + (r_eq**2 / r_pol**2) * np.sin(y) ** 2)
    )
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H**2 - r_eq**2

    r_s = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    s_x = r_s * np.cos(x) * np.cos(y)
    s_y = -r_s * np.sin(x)
    s_z = r_s * np.cos(x) * np.sin(y)

    lat = np.arctan(
        (r_eq**2 / r_pol**2) * (s_z / np.sqrt((H - s_x) ** 2 + s_y**2))
    ) * (180 / np.pi)
    lon = (l_0 - np.arctan(s_y / (H - s_x))) * (180 / np.pi)

    ds = ds.assign_coords({"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)})
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"

    return ds


def goes_load_netcdf_iris(
    path_to_files: str, tracking_var: str, CONFIG: dict
) -> tuple[iris.cube.Cube, xr.DataArray]:
    """


    Parameters
    ----------
    path_to_files : str
        Glob path to input files i.e. "/data/usr/GOES/*.nc".
    tracking_var : str
        ["tb"], variable which is going to be used for tracking--brightness temperature..
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if missing GOES field on CONFIG or invalid tracking variable entered.

    Returns
    -------
    cube : iris.cube.Cube
        Iris cube continaing birhtness temperature ready for tobac tracking.
    goes_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing GOES brightness temperature data.

    """

    # Convert to iris cube and return
    if tracking_var.lower() == "tb":

        # Import GOES satetellite data as xarray dataset
        goes_xarray = xr.open_mfdataset(
            path_to_files, coords="all", concat_dim="t", combine="nested"
        )

        # Add lat and lon to goes_xarray
        # Ignore boundary warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            goes_xarray = calc_latlon(goes_xarray)

        # Get GOES spatial resolution
        spacial_res = goes_xarray.spatial_resolution

        # Convert x and y projections to x and y projection in meters according to http://meteothink.org/examples/meteoinfolab/satellite/geos-16.html
        sat_height = goes_xarray.goes_imager_projection.attrs[
            "perspective_point_height"
        ]

        # Subset location of interest
        if "goes" in CONFIG:

            # Subset time based on user inputs
            if (
                "min_frame_index" in CONFIG["goes"]
                or "max_frame_index" in CONFIG["goes"]
            ):
                min_frame = (
                    CONFIG["goes"]["min_frame_index"]
                    if "min_frame_index" in CONFIG["goes"]
                    else 0
                )
                max_frame = (
                    CONFIG["goes"]["max_frame_index"] + 1
                    if "max_frame_index" in CONFIG["goes"]
                    else goes_xarray.dims["t"]
                )

                goes_xarray = goes_xarray.isel(
                    t=np.arange(min_frame, max_frame),
                    drop=True,
                )

            if "bounds" in CONFIG["goes"]:

                mask_lon = (goes_xarray.lon >= CONFIG["goes"]["bounds"][0]) & (
                    goes_xarray.lon <= CONFIG["goes"]["bounds"][1]
                )
                mask_lat = (goes_xarray.lat >= CONFIG["goes"]["bounds"][2]) & (
                    goes_xarray.lat <= CONFIG["goes"]["bounds"][3]
                )

                goes_xarray = goes_xarray.CMI.where(mask_lon & mask_lat, drop=True)

            else:
                goes_xarray = goes_xarray.CMI

        else:
            raise Exception('!=====CONFIG Missing "goes" Field=====!')

        # Replace time dimension with minutes since first time and add other x y z coords
        first_time = goes_xarray.t.values[0]
        goes_xarray = goes_xarray.assign_coords(
            t=(goes_xarray.t.values - first_time) / np.timedelta64(1, "m"),
            projection_x_coordinate=("x", goes_xarray.x.values * sat_height),
            projection_y_coordinate=("y", goes_xarray.y.values * sat_height),
            south_north=("y", np.arange(goes_xarray.shape[1])),
            west_east=("x", np.arange(goes_xarray.shape[2])),
            x=("x", np.arange(goes_xarray.shape[2])),
            y=("y", np.arange(goes_xarray.shape[1])),
        )

        # Drop incorrect projection vars
        goes_xarray = goes_xarray.drop_vars(["x_image", "y_image"])

        # Add spatial resolution info
        goes_xarray = goes_xarray.assign_attrs({"spatial_resolution": spacial_res})

        # Adjust dimension names to be standards accepted by iris
        goes_xarray["t"] = goes_xarray.t.assign_attrs(
            {
                "standard_name": "time",
                "long_name": f"minutes since {first_time}",
                "units": f"minutes since {first_time}",
            }
        )
        goes_xarray["lat"] = goes_xarray.lat.assign_attrs({"standard_name": "latitude"})
        goes_xarray["lon"] = goes_xarray.lon.assign_attrs(
            {"standard_name": "longitude"}
        )
        goes_xarray["projection_x_coordinate"] = (
            goes_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
        )
        goes_xarray["projection_y_coordinate"] = (
            goes_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
        )

        return (goes_xarray.to_iris(), goes_xarray)

    raise Exception(
        f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
    )


def goes_load_netcdf(
    path_to_files: str, tracking_var: str, CONFIG: dict
) -> xr.DataArray:
    """


    Parameters
    ----------
    path_to_files : str
        Glob path to input files i.e. "/data/usr/GOES/*.nc".
    tracking_var : str
        ["tb"], variable which is going to be used for tracking--brightness temperature..
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if missing GOES field on CONFIG or invalid tracking variable entered.

    Returns
    -------
    goes_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing GOES brightness temperature data.

    """

    # Convert to iris cube and return
    if tracking_var.lower() == "tb":

        # Import GOES satetellite data as xarray dataset
        goes_xarray = xr.open_mfdataset(
            path_to_files, coords="all", concat_dim="t", combine="nested"
        )

        # Add lat and lon to goes_xarray
        # Ignore boundary warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            goes_xarray = calc_latlon(goes_xarray)

        # Get GOES spatial resolution
        spacial_res = goes_xarray.spatial_resolution

        # Convert x and y projections to x and y projection in meters according to http://meteothink.org/examples/meteoinfolab/satellite/geos-16.html
        sat_height = goes_xarray.goes_imager_projection.attrs[
            "perspective_point_height"
        ]

        # Subset location of interest
        if "goes" in CONFIG:

            # Subset time based on user inputs
            if (
                "min_frame_index" in CONFIG["goes"]
                or "max_frame_index" in CONFIG["goes"]
            ):
                min_frame = (
                    CONFIG["goes"]["min_frame_index"]
                    if "min_frame_index" in CONFIG["goes"]
                    else 0
                )
                max_frame = (
                    CONFIG["goes"]["max_frame_index"] + 1
                    if "max_frame_index" in CONFIG["goes"]
                    else goes_xarray.dims["t"]
                )

                goes_xarray = goes_xarray.isel(
                    t=np.arange(min_frame, max_frame),
                    drop=True,
                )

            if "bounds" in CONFIG["goes"]:

                mask_lon = (goes_xarray.lon >= CONFIG["goes"]["bounds"][0]) & (
                    goes_xarray.lon <= CONFIG["goes"]["bounds"][1]
                )
                mask_lat = (goes_xarray.lat >= CONFIG["goes"]["bounds"][2]) & (
                    goes_xarray.lat <= CONFIG["goes"]["bounds"][3]
                )

                goes_xarray = goes_xarray.CMI.where(mask_lon & mask_lat, drop=True)

            else:
                goes_xarray = goes_xarray.CMI

        else:
            raise Exception('!=====CONFIG Missing "goes" Field=====!')

        # Replace time dimension with minutes since first time and add other x y z coords
        first_time = goes_xarray.t.values[0]
        goes_xarray = goes_xarray.assign_coords(
            t=(goes_xarray.t.values - first_time) / np.timedelta64(1, "m"),
            projection_x_coordinate=("x", goes_xarray.x.values * sat_height),
            projection_y_coordinate=("y", goes_xarray.y.values * sat_height),
            south_north=("y", np.arange(goes_xarray.shape[1])),
            west_east=("x", np.arange(goes_xarray.shape[2])),
            x=("x", np.arange(goes_xarray.shape[2])),
            y=("y", np.arange(goes_xarray.shape[1])),
        )

        # Drop incorrect projection vars
        goes_xarray = goes_xarray.drop_vars(["x_image", "y_image"])

        # Add spatial resolution info
        goes_xarray = goes_xarray.assign_attrs({"spatial_resolution": spacial_res})

        # Adjust dimension names to be standards accepted by iris
        goes_xarray["t"] = goes_xarray.t.assign_attrs(
            {
                "standard_name": "time",
                "long_name": f"minutes since {first_time}",
                "units": f"minutes since {first_time}",
            }
        )
        goes_xarray["lat"] = goes_xarray.lat.assign_attrs({"standard_name": "latitude"})
        goes_xarray["lon"] = goes_xarray.lon.assign_attrs(
            {"standard_name": "longitude"}
        )
        goes_xarray["projection_x_coordinate"] = (
            goes_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
        )
        goes_xarray["projection_y_coordinate"] = (
            goes_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
        )

        return goes_xarray

    raise Exception(
        f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
    )
