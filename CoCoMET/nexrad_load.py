#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:28:49 2024

@author: thahn
"""

# =============================================================================
# Loads in and grids the NEXRAD Arhcival Level 2 Data using ARM-Pyart then converts into iris cubes and xarray Datasets for use in trackers
# =============================================================================


import glob
import multiprocessing
import os
import warnings
from functools import partial

import cftime
import iris
import iris.cube
import numpy as np
import pyart
import xarray as xr
from tqdm import tqdm


def gen_and_save_nexrad_grid(
    path_to_files: str,
    save_location: str,
    tracking_var: str,
    CONFIG: dict,
    parallel_processing: bool = False,
    max_cores: int | None = None,
) -> None:
    """


    Parameters
    ----------
    path_to_files : str
        Glob type path to the NEXRAD level 2 archive files.
    save_location : str
        Path to where the gridded NEXRAD files should be saved to, should be a directory and end with "/".
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG : dict
        User configuration file.
    parallel_processing : bool, optional
        Bool determinig whether to use parallel processing when gridding files. The default is False.
    max_cores : int, optional
        Number of cores to use if parallel_processing == True. The default is None.

    Raises
    ------
    Exception
        Exception if no files to grid or invalid tracking variables.

    Returns
    -------
    None

    """

    # Get all archive files and iterate over them
    files = np.sort(glob.glob(path_to_files))

    if len(files) == 0:
        raise Exception("!=====No Files Present to Grid=====!")
        return

    # Create grid folder if it does not already exist
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    # If parallel processing is enabled, run that version and return
    if parallel_processing:
        gen_and_save_nexrad_grid_multi(
            files, save_location, tracking_var, CONFIG, max_cores
        )
        return

    # Extract just the filenames from the paths without the file extensions
    file_names = [os.path.basename(f).split(".")[0] for f in files]

    for idx, ff in tqdm(
        enumerate(files), desc="=====Gridding NEXRAD=====", total=len(files)
    ):
        if tracking_var.lower() == "dbz":
            # Create radar object including only field of interest
            radar = pyart.io.read_nexrad_archive(ff, include_fields="reflectivity")

            # Create radar grid using user-defined params
            radar_grid = pyart.map.grid_from_radars(
                radar, **CONFIG["nexrad"]["gridding"]
            )

            # Save radar grid to save_location as a netcdf file and delete radar and radar_grid objects to free memory
            pyart.io.write_grid(
                save_location + file_names[idx] + "_grid.nc",
                radar_grid,
                arm_alt_lat_lon_variables=True,
                write_point_x_y_z=True,
                write_point_lon_lat_alt=True,
            )

            del radar
            del radar_grid

        else:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )
            return

    return


def create_and_save_grid_single(file, save_location, tracking_var, CONFIG):
    """
    Helper Function for Parallel Processing
    """

    if tracking_var.lower() == "dbz":
        # Create radar object including only field of interest
        radar = pyart.io.read_nexrad_archive(file, include_fields="reflectivity")

        # Create radar grid using user-defined params
        radar_grid = pyart.map.grid_from_radars(radar, **CONFIG["nexrad"]["gridding"])

        # Save radar grid to save_location as a netcdf file and delete radar and radar_grid objects to free memory
        file_name = os.path.basename(file).split(".")[0]
        pyart.io.write_grid(
            save_location + file_name + "_grid.nc",
            radar_grid,
            arm_alt_lat_lon_variables=True,
            write_point_x_y_z=True,
            write_point_lon_lat_alt=True,
        )

        del radar
        del radar_grid

    else:
        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )
        return


def gen_and_save_nexrad_grid_multi(
    files: list[str],
    save_location: str,
    tracking_var: str,
    CONFIG: dict,
    max_cores: int,
) -> None:
    """


    Parameters
    ----------
    files : list[str]
        List containing all paths to NEXRAD level 2 archive files.
    save_location : str
        Path to where the gridded NEXRAD files should be saved to.
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG : dict
        User configuration file.
    max_cores : int
        Number of cores to use for parallel processing

    Returns
    -------
    None

    """

    # Start a pool with max_cores and run the grid function
    with multiprocessing.Pool(max_cores) as multi_pool:
        with tqdm(total=len(files), desc="=====Parallel Gridding NEXRAD=====") as pbar:
            for _ in multi_pool.imap_unordered(
                partial(
                    create_and_save_grid_single,
                    save_location=save_location,
                    tracking_var=tracking_var,
                    CONFIG=CONFIG,
                ),
                files,
            ):
                pbar.update()

    return


def nexrad_load_netcdf_iris(
    path_to_files: str,
    file_type: str,
    tracking_var: str,
    CONFIG: dict,
    save_location: str | None = None,
) -> tuple[iris.cube.Cube, xr.DataArray]:
    """


    Parameters
    ----------
    path_to_files : str
        Glob path to input files, either archival or grided netcdf--i.e. "/data/usr/KVNX*_V06.ar2v".
    file_type : str
        ["ar2v", "nc"] type of input file--either archival or netcdf.
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG : dict
        User configuration file.
    save_location : str, optional
        Where to save gridded NEXRAD data to if file_type=="ar2v". The default is None.

    Returns
    -------
    cube : iris.cube.Cube
        Iris cube continaing gridded reflectivity data ready for tobac tracking.
    nexrad_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing gridded NEXRAD archival data.

    """

    # If data is archival, perform gridding
    if file_type.lower() == "ar2v":
        # Make sure save_location ends with a "/"
        if save_location[-1] != "/":
            save_location = save_location + "//"

        # Create grid
        gen_and_save_nexrad_grid(
            path_to_files,
            save_location,
            tracking_var,
            CONFIG,
            CONFIG["parallel_processing"],
            CONFIG["max_cores"],
        )

        if tracking_var.lower() == "dbz":
            # Read radar objects in and concat into one xarray datarray
            # This is a stupid hacky fix because pyart is dumb
            radar_objects = []
            for file in np.sort(glob.glob(save_location + "*")):
                # Ignore import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    radar_objects.append(pyart.io.read_grid(file).to_xarray())

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity.drop_vars(
                [
                    "origin_altitude",
                    "origin_latitude",
                    "origin_longitude",
                    "ProjectionCoordinateSystem",
                    "projection",
                ]
            )
            del radar_objects

            # Subset location and time of interest
            if "nexrad" in CONFIG:
                # Subset time based on user inputs
                if (
                    "min_frame_index" in CONFIG["nexrad"]
                    or "max_frame_index" in CONFIG["nexrad"]
                ):
                    min_frame = (
                        CONFIG["nexrad"]["min_frame_index"]
                        if "min_frame_index" in CONFIG["nexrad"]
                        else 0
                    )
                    max_frame = (
                        CONFIG["nexrad"]["max_frame_index"] + 1
                        if "max_frame_index" in CONFIG["nexrad"]
                        else nexrad_xarray.dims["time"]
                    )

                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            min_frame,
                            max_frame,
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["nexrad"]:
                    mask_lon = (nexrad_xarray.lon >= CONFIG["nexrad"]["bounds"][0]) & (
                        nexrad_xarray.lon <= CONFIG["nexrad"]["bounds"][1]
                    )
                    mask_lat = (nexrad_xarray.lat >= CONFIG["nexrad"]["bounds"][2]) & (
                        nexrad_xarray.lat <= CONFIG["nexrad"]["bounds"][3]
                    )

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception("""!=====CONFIG Missing "nexrad" Field=====!""")

            # Replace time dimension with minutes since first time and add other x y z coords
            first_time = nexrad_xarray.time.values[0]
            nexrad_xarray = nexrad_xarray.assign_coords(
                time=cftime.date2num(
                    nexrad_xarray.time.values, f"minutes since {first_time}"
                ),
                south_north=("y", np.arange(nexrad_xarray.shape[2])),
                west_east=("x", np.arange(nexrad_xarray.shape[3])),
                projection_x_coordinate=("x", nexrad_xarray.x.values),
                projection_y_coordinate=("y", nexrad_xarray.y.values),
                x=("x", np.arange(nexrad_xarray.shape[3])),
                y=("y", np.arange(nexrad_xarray.shape[2])),
                model_level_number=("z", np.arange(nexrad_xarray.shape[1])),
            )

            # Create DT attribute
            dt_array = np.diff(nexrad_xarray.time.values)
            if len(np.unique(dt_array)) != 1:
                nexrad_xarray.attrs["DT"] = dt_array * 60  # min -> s
            else:
                nexrad_xarray.attrs["DT"] = dt_array[0] * 60  # min -> s

            # Adjust dimension names to be standards accepted by iris
            nexrad_xarray["time"] = nexrad_xarray.time.assign_attrs(
                {
                    "standard_name": "time",
                    "long_name": f"minutes since {first_time}",
                    "units": f"minutes since {first_time}",
                }
            )
            nexrad_xarray["z"] = nexrad_xarray.z.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )
            nexrad_xarray["lat"] = nexrad_xarray.lat.assign_attrs(
                {"standard_name": "latitude"}
            )
            nexrad_xarray["lon"] = nexrad_xarray.lon.assign_attrs(
                {"standard_name": "longitude"}
            )
            nexrad_xarray[
                "projection_x_coordinate"
            ] = nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            nexrad_xarray[
                "projection_y_coordinate"
            ] = nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})

            # Add altitude dimension to xarray but not to cube
            nexrad_cube = nexrad_xarray.to_iris()
            nexrad_xarray = nexrad_xarray.assign_coords(
                altitude=("z", nexrad_xarray.z.values)
            )

            nexrad_xarray["z"] = nexrad_xarray.z.assign_attrs({"standard_name": ""})
            nexrad_xarray["altitude"] = nexrad_xarray.altitude.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )

            return (nexrad_cube, nexrad_xarray)

        else:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )
            return

    # If data already grided, just return concated netcdf dataset
    elif file_type.lower() == "nc":
        # Convert to iris cube and return
        if tracking_var.lower() == "dbz":
            # Read radar objects in and concat into one xarray datarray
            # This is a stupid hacky fix because pyart is dumb
            radar_objects = []
            for file in np.sort(glob.glob(path_to_files)):
                # Ignore import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    radar_objects.append(pyart.io.read_grid(file).to_xarray())

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity.drop_vars(
                [
                    "origin_altitude",
                    "origin_latitude",
                    "origin_longitude",
                    "ProjectionCoordinateSystem",
                    "projection",
                ]
            )
            del radar_objects

            # Subset location and time of interest
            if "nexrad" in CONFIG:
                # Subset time based on user inputs
                if (
                    "min_frame_index" in CONFIG["nexrad"]
                    or "max_frame_index" in CONFIG["nexrad"]
                ):
                    min_frame = (
                        CONFIG["nexrad"]["min_frame_index"]
                        if "min_frame_index" in CONFIG["nexrad"]
                        else 0
                    )
                    max_frame = (
                        CONFIG["nexrad"]["max_frame_index"] + 1
                        if "max_frame_index" in CONFIG["nexrad"]
                        else nexrad_xarray.dims["time"]
                    )

                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            min_frame,
                            max_frame,
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["nexrad"]:
                    mask_lon = (nexrad_xarray.lon >= CONFIG["nexrad"]["bounds"][0]) & (
                        nexrad_xarray.lon <= CONFIG["nexrad"]["bounds"][1]
                    )
                    mask_lat = (nexrad_xarray.lat >= CONFIG["nexrad"]["bounds"][2]) & (
                        nexrad_xarray.lat <= CONFIG["nexrad"]["bounds"][3]
                    )

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception("""!=====CONFIG Missing "nexrad" Field=====!""")

            # Replace time dimension with minutes since first time and add other x y z coords
            first_time = nexrad_xarray.time.values[0]
            nexrad_xarray = nexrad_xarray.assign_coords(
                time=cftime.date2num(
                    nexrad_xarray.time.values, f"minutes since {first_time}"
                ),
                south_north=("y", np.arange(nexrad_xarray.shape[2])),
                west_east=("x", np.arange(nexrad_xarray.shape[3])),
                projection_x_coordinate=("x", nexrad_xarray.x.values),
                projection_y_coordinate=("y", nexrad_xarray.y.values),
                x=("x", np.arange(nexrad_xarray.shape[3])),
                y=("y", np.arange(nexrad_xarray.shape[2])),
                model_level_number=("z", np.arange(nexrad_xarray.shape[1])),
            )

            # Create DT attribute
            dt_array = np.diff(nexrad_xarray.time.values)
            if len(np.unique(dt_array)) != 1:
                nexrad_xarray.attrs["DT"] = dt_array * 60  # min -> s
            else:
                nexrad_xarray.attrs["DT"] = dt_array[0] * 60  # min -> s
                
            # Adjust dimension names to be standards accepted by iris
            nexrad_xarray["time"] = nexrad_xarray.time.assign_attrs(
                {
                    "standard_name": "time",
                    "long_name": f"minutes since {first_time}",
                    "units": f"minutes since {first_time}",
                }
            )
            nexrad_xarray["z"] = nexrad_xarray.z.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )
            nexrad_xarray["lat"] = nexrad_xarray.lat.assign_attrs(
                {"standard_name": "latitude"}
            )
            nexrad_xarray["lon"] = nexrad_xarray.lon.assign_attrs(
                {"standard_name": "longitude"}
            )
            nexrad_xarray[
                "projection_x_coordinate"
            ] = nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            nexrad_xarray[
                "projection_y_coordinate"
            ] = nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})

            # Add altitude dimension to xarray but not to cube
            nexrad_cube = nexrad_xarray.to_iris()#.remove_coord("altitude")
            nexrad_xarray = nexrad_xarray.assign_coords(
                altitude=("z", nexrad_xarray.z.values)
            )

            nexrad_xarray["z"] = nexrad_xarray.z.assign_attrs({"standard_name": ""})
            nexrad_xarray["altitude"] = nexrad_xarray.altitude.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )

            return (nexrad_cube, nexrad_xarray)

        else:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )
            return

    else:
        raise Exception(
            f"!=====Invalid File Type. You Entered: {file_type.lower()}=====!"
        )
        return


def nexrad_load_netcdf(
    path_to_files: str,
    file_type: str,
    tracking_var: str,
    CONFIG: dict,
    save_location: str | None = None,
) -> xr.DataArray:
    """


    Parameters
    ----------
    path_to_files : str
        Glob path to input files, either archival or grided netcdf--i.e. "/data/usr/KVNX*_V06.ar2v".
    file_type : str
        ["ar2v", "nc"] type of input file--either archival or netcdf.
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG : dict
        User configuration file.
    save_location : str, optional
        Where to save gridded NEXRAD data to if file_type=="ar2v". The default is None.

    Returns
    -------
    nexrad_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing gridded NEXRAD archival data.

    """

    # If data is archival, perform gridding
    if file_type.lower() == "ar2v":
        # Make sure save_location ends with a "/"
        if save_location[-1] != "/":
            save_location = save_location + "//"

        # Create grid
        gen_and_save_nexrad_grid(
            path_to_files,
            save_location,
            tracking_var,
            CONFIG,
            CONFIG["parallel_processing"],
            CONFIG["max_cores"],
        )

        # Open them as netcdf file and return
        if tracking_var.lower() == "dbz":
            # Read radar objects in and concat into one xarray datarray
            # This is a stupid hacky fix because pyart is dumb
            radar_objects = []
            for file in np.sort(glob.glob(save_location + "*")):
                # Ignore import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    radar_objects.append(pyart.io.read_grid(file).to_xarray())

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity.drop_vars(
                [
                    "origin_altitude",
                    "origin_latitude",
                    "origin_longitude",
                    "ProjectionCoordinateSystem",
                    "projection",
                ]
            )
            del radar_objects

            # Subset location and time of interest
            if "nexrad" in CONFIG:
                # Subset time based on user inputs
                if (
                    "min_frame_index" in CONFIG["nexrad"]
                    or "max_frame_index" in CONFIG["nexrad"]
                ):
                    min_frame = (
                        CONFIG["nexrad"]["min_frame_index"]
                        if "min_frame_index" in CONFIG["nexrad"]
                        else 0
                    )
                    max_frame = (
                        CONFIG["nexrad"]["max_frame_index"] + 1
                        if "max_frame_index" in CONFIG["nexrad"]
                        else nexrad_xarray.dims["time"]
                    )

                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            min_frame,
                            max_frame,
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["nexrad"]:
                    mask_lon = (nexrad_xarray.lon >= CONFIG["nexrad"]["bounds"][0]) & (
                        nexrad_xarray.lon <= CONFIG["nexrad"]["bounds"][1]
                    )
                    mask_lat = (nexrad_xarray.lat >= CONFIG["nexrad"]["bounds"][2]) & (
                        nexrad_xarray.lat <= CONFIG["nexrad"]["bounds"][3]
                    )

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception("""!=====CONFIG Missing "nexrad" Field=====!""")

            # Replace time dimension with minutes since first time and add other x y z coords
            first_time = nexrad_xarray.time.values[0]
            nexrad_xarray = nexrad_xarray.assign_coords(
                time=cftime.date2num(
                    nexrad_xarray.time.values, f"minutes since {first_time}"
                ),
                south_north=("y", np.arange(nexrad_xarray.shape[2])),
                west_east=("x", np.arange(nexrad_xarray.shape[3])),
                projection_x_coordinate=("x", nexrad_xarray.x.values),
                projection_y_coordinate=("y", nexrad_xarray.y.values),
                x=("x", np.arange(nexrad_xarray.shape[3])),
                y=("y", np.arange(nexrad_xarray.shape[2])),
                model_level_number=("z", np.arange(nexrad_xarray.shape[1])),
                altitude=("z", nexrad_xarray.z.values),
            )

            # Adjust dimension names to be standards accepted by iris
            nexrad_xarray["time"] = nexrad_xarray.time.assign_attrs(
                {
                    "standard_name": "time",
                    "long_name": f"minutes since {first_time}",
                    "units": f"minutes since {first_time}",
                }
            )
            nexrad_xarray["z"] = nexrad_xarray.z.assign_attrs(
                {"standard_name": "", "units": "m"}
            )
            nexrad_xarray["lat"] = nexrad_xarray.lat.assign_attrs(
                {"standard_name": "latitude"}
            )
            nexrad_xarray["lon"] = nexrad_xarray.lon.assign_attrs(
                {"standard_name": "longitude"}
            )
            nexrad_xarray[
                "projection_x_coordinate"
            ] = nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            nexrad_xarray[
                "projection_y_coordinate"
            ] = nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
            nexrad_xarray["altitude"] = nexrad_xarray.altitude.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )

            return nexrad_xarray

        else:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )
            return

    # If data already grided, just return concated netcdf dataset
    elif file_type.lower() == "nc":
        if tracking_var.lower() == "dbz":
            # Read radar objects in and concat into one xarray datarray
            # This is a stupid hacky fix because pyart is dumb
            radar_objects = []
            for file in np.sort(glob.glob(path_to_files)):
                # Ignore import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    radar_objects.append(pyart.io.read_grid(file).to_xarray())

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity.drop_vars(
                [
                    "origin_altitude",
                    "origin_latitude",
                    "origin_longitude",
                    "ProjectionCoordinateSystem",
                    "projection",
                ]
            )
            del radar_objects

            # Subset location and time of interest
            if "nexrad" in CONFIG:
                # Subset time based on user inputs
                if (
                    "min_frame_index" in CONFIG["nexrad"]
                    or "max_frame_index" in CONFIG["nexrad"]
                ):
                    min_frame = (
                        CONFIG["nexrad"]["min_frame_index"]
                        if "min_frame_index" in CONFIG["nexrad"]
                        else 0
                    )
                    max_frame = (
                        CONFIG["nexrad"]["max_frame_index"] + 1
                        if "max_frame_index" in CONFIG["nexrad"]
                        else nexrad_xarray.dims["time"]
                    )

                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            min_frame,
                            max_frame,
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["nexrad"]:
                    mask_lon = (nexrad_xarray.lon >= CONFIG["nexrad"]["bounds"][0]) & (
                        nexrad_xarray.lon <= CONFIG["nexrad"]["bounds"][1]
                    )
                    mask_lat = (nexrad_xarray.lat >= CONFIG["nexrad"]["bounds"][2]) & (
                        nexrad_xarray.lat <= CONFIG["nexrad"]["bounds"][3]
                    )

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception("""!=====CONFIG Missing "nexrad" Field=====!""")

            # Replace time dimension with minutes since first time and add other x y z coords
            first_time = nexrad_xarray.time.values[0]
            nexrad_xarray = nexrad_xarray.assign_coords(
                time=cftime.date2num(
                    nexrad_xarray.time.values, f"minutes since {first_time}"
                ),
                south_north=("y", np.arange(nexrad_xarray.shape[2])),
                west_east=("x", np.arange(nexrad_xarray.shape[3])),
                projection_x_coordinate=("x", nexrad_xarray.x.values),
                projection_y_coordinate=("y", nexrad_xarray.y.values),
                x=("x", np.arange(nexrad_xarray.shape[3])),
                y=("y", np.arange(nexrad_xarray.shape[2])),
                model_level_number=("z", np.arange(nexrad_xarray.shape[1])),
                altitude=("z", nexrad_xarray.z.values),
            )

            # Adjust dimension names to be standards
            nexrad_xarray["time"] = nexrad_xarray.time.assign_attrs(
                {
                    "standard_name": "time",
                    "long_name": f"minutes since {first_time}",
                    "units": f"minutes since {first_time}",
                }
            )
            nexrad_xarray["z"] = nexrad_xarray.z.assign_attrs(
                {"standard_name": "", "units": "m"}
            )
            nexrad_xarray["lat"] = nexrad_xarray.lat.assign_attrs(
                {"standard_name": "latitude"}
            )
            nexrad_xarray["lon"] = nexrad_xarray.lon.assign_attrs(
                {"standard_name": "longitude"}
            )
            nexrad_xarray[
                "projection_x_coordinate"
            ] = nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            nexrad_xarray[
                "projection_y_coordinate"
            ] = nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
            nexrad_xarray["altitude"] = nexrad_xarray.altitude.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )

            return nexrad_xarray

        else:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )
            return

    else:
        raise Exception(
            f"!=====Invalid File Type. You Entered: {file_type.lower()}=====!"
        )
        return
