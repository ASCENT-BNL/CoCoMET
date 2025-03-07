#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:19:18 2024

@author: thahn
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:28:49 2024

@author: thahn
"""

# =============================================================================
# Loads in pre-gridded radar data which follows the radar standardization set out in the CoCoMET-US Section S1.1.
# =============================================================================

import iris.cube
import numpy as np
import xarray as xr


def standard_radar_load_netcdf_iris(
    path_to_files: str, tracking_var: str, CONFIG: dict
) -> tuple[iris.cube.Cube, xr.DataArray]:
    """


    Parameters
    ----------
    path_to_files : str
        Glob path to gridded input netcdf files--i.e. "/data/usr/KVNX*_V06.nc".
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity..
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if CONFIG missing standard_radar field or invalid tracking variable entered.

    Returns
    -------
    radar_cube : iris.cube.Cube
        Iris cube continaing gridded reflectivity data ready for tobac tracking.
    radar_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing gridded reflectivity data.

    """

    # Convert to iris cube and return
    if tracking_var.lower() == "dbz":
        # Open combined netcdf radar dataarray
        radar_xarray = xr.open_mfdataset(
            path_to_files,  # concat_dim="time", combine="nested"
        ).reflectivity

        # Subset location of interest
        if "standard_radar" in CONFIG:
            # Subset time based on user inputs
            if (
                "min_frame_index" in CONFIG["standard_radar"]
                or "max_frame_index" in CONFIG["standard_radar"]
            ):
                min_frame = (
                    CONFIG["standard_radar"]["min_frame_index"]
                    if "min_frame_index" in CONFIG["standard_radar"]
                    else 0
                )
                max_frame = (
                    CONFIG["standard_radar"]["max_frame_index"] + 1
                    if "max_frame_index" in CONFIG["standard_radar"]
                    else radar_xarray.dims["time"]
                )

                radar_xarray = radar_xarray.isel(
                    time=np.arange(
                        min_frame,
                        max_frame,
                    ),
                    drop=True,
                )

            if "bounds" in CONFIG["standard_radar"]:
                mask_lon = (
                    radar_xarray.lon >= CONFIG["standard_radar"]["bounds"][0]
                ) & (radar_xarray.lon <= CONFIG["standard_radar"]["bounds"][1])
                mask_lat = (
                    radar_xarray.lat >= CONFIG["standard_radar"]["bounds"][2]
                ) & (radar_xarray.lat <= CONFIG["standard_radar"]["bounds"][3])

                try:
                    radar_xarray = radar_xarray.where(mask_lon & mask_lat, drop=True)
                except:
                    radar_xarray = radar_xarray.where(np.logical_and(mask_lon, mask_lat).compute(), drop = True)

        else:
            raise Exception("""!=====CONFIG Missing "standard_radar" Field=====!""")

        # Create DT attribute
        dt_array = np.diff(radar_xarray.time.values).astype('timedelta64[s]').astype(int)
        if len(np.unique(dt_array)) != 1:
            radar_xarray.attrs["DT"] = dt_array
        else:
            radar_xarray.attrs["DT"] = dt_array[0]
            
        first_time = radar_xarray.time.values[0]

        radar_xarray = radar_xarray.assign_coords(
            time=(
                "time",
                (radar_xarray.time.values - first_time)
                .astype("timedelta64[m]")
                .astype(float),
            )
        )
        radar_xarray["time"] = radar_xarray.time.assign_attrs(
            {
                "standard_name": "time",
                "long_name": f"minutes since {first_time}",
                "units": f"minutes since {first_time}",
            }
        )

        if not("projection_x_coordinate" in radar_xarray.coords) and not("projection_y_coordinate" in radar_xarray.coords):

            radar_xarray = radar_xarray.assign_coords(
                projection_x_coordinate=("x", radar_xarray.proj_x.values),
                projection_y_coordinate=("y", radar_xarray.proj_y.values),
            )

            radar_xarray[
                "projection_x_coordinate"
            ] = radar_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            radar_xarray[
                "projection_y_coordinate"
            ] = radar_xarray.projection_y_coordinate.assign_attrs({"units": "m"})

        # Drop altitude coordinate temporarily when making iris cube
        drop_variables = ["altitude", "proj_x", "proj_y"]
        for dv in drop_variables:
            if dv in radar_xarray.coords:
                radar_xarray = radar_xarray.drop_vars(dv)
        radar_cube = radar_xarray.to_iris()

        radar_xarray = radar_xarray.assign_coords(altitude=("z", radar_xarray.z.values))
        radar_xarray["z"] = radar_xarray.z.assign_attrs({"standard_name": ""})
        radar_xarray["altitude"] = radar_xarray.altitude.assign_attrs(
            {"standard_name": "altitude", "units": "m"}
        )

        return (radar_cube, radar_xarray)

    else:
        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )


def standard_radar_load_netcdf(
    path_to_files: str, tracking_var: str, CONFIG: dict
) -> xr.DataArray:
    """


    Parameters
    ----------
    path_to_files : str
        Glob path to gridded input netcdf files--i.e. "/data/usr/KVNX*_V06.nc".
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity..
    CONFIG : dict
        User configuration file.

    Raises
    ------
    Exception
        Exception if CONFIG missing standard_radar field or invalid tracking variable entered.

    Returns
    -------
    radar_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing gridded reflectivity data.

    """

    # Convert to iris cube and return
    if tracking_var.lower() == "dbz":
        # Open combined netcdf radar dataarray
        radar_xarray = xr.open_mfdataset(
            path_to_files, concat_dim="time", combine="nested"
        ).reflectivity

        # Subset location of interest
        if "standard_radar" in CONFIG:
            # Subset time based on user inputs
            if (
                "min_frame_index" in CONFIG["standard_radar"]
                or "max_frame_index" in CONFIG["standard_radar"]
            ):
                min_frame = (
                    CONFIG["standard_radar"]["min_frame_index"]
                    if "min_frame_index" in CONFIG["standard_radar"]
                    else 0
                )
                max_frame = (
                    CONFIG["standard_radar"]["max_frame_index"] + 1
                    if "max_frame_index" in CONFIG["standard_radar"]
                    else radar_xarray.dims["time"]
                )

                radar_xarray = radar_xarray.isel(
                    time=np.arange(
                        min_frame,
                        max_frame,
                    ),
                    drop=True,
                )

            if "bounds" in CONFIG["standard_radar"]:
                mask_lon = (
                    radar_xarray.lon >= CONFIG["standard_radar"]["bounds"][0]
                ) & (radar_xarray.lon <= CONFIG["standard_radar"]["bounds"][1])
                mask_lat = (
                    radar_xarray.lat >= CONFIG["standard_radar"]["bounds"][2]
                ) & (radar_xarray.lat <= CONFIG["standard_radar"]["bounds"][3])

                radar_xarray = radar_xarray.where(mask_lon & mask_lat, drop=True)

        else:
            raise Exception("""!=====CONFIG Missing "standard_radar" Field=====!""")

        first_time = radar_xarray.time.values[0]
        radar_xarray = radar_xarray.assign_coords(
            time=(
                "time",
                (radar_xarray.time.values - first_time)
                .astype("timedelta64[m]")
                .astype(float),
            )
        )
        radar_xarray["time"] = radar_xarray.time.assign_attrs(
            {
                "standard_name": "time",
                "long_name": f"minutes since {first_time}",
                "units": f"minutes since {first_time}",
            }
        )

        radar_xarray["z"] = radar_xarray.z.assign_attrs({"standard_name": ""})
        radar_xarray["altitude"] = radar_xarray.altitude.assign_attrs(
            {"standard_name": "altitude", "units": "m"}
        )

        # Return subseted radar xarray
        return radar_xarray

    else:
        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )
