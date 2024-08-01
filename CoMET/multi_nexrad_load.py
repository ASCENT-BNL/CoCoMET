#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:57:43 2024

@author: thahn
"""

# =============================================================================
# This file contains scripts for loading (and gridding) multiple NEXRAD radars at once
# =============================================================================

import glob
import multiprocessing
import warnings
from functools import partial

import cftime
import iris
import iris.cube
import numpy as np
import pyart
import xarray as xr
from tqdm import tqdm


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def gen_and_save_multi_nexrad_grid(
    paths_to_files: list[str],
    save_location: str,
    tracking_var: str,
    CONFIG: dict,
    parallel_processing: bool = False,
    max_cores: int | None = None,
) -> None:
    """


    Parameters
    ----------
    paths_to_files : list[str]
        Array of glob path to archival NEXRAD level 2 input files--i.e. ["/data/usr/KVNX*_V06.ar2v", "/data/usr/KIVX*_V06.ar2v"].
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
        Exception if there are no files to grid or invalid tracking variable entered.

    Returns
    -------
    None

    """

    # No parallel processing for this as the time of arrival matters
    radar_list = []
    radar_time_list = []

    for ii, radar_path in enumerate(paths_to_files):

        radar_list.append([])
        radar_time_list.append([])

        # Get all archive files and iterate over them
        files = np.sort(glob.glob(radar_path))

        if len(files) == 0:
            raise Exception("!=====No Files Present to Grid=====!")

        # Get radars and their respective times. Don"t load them all at once to conserve memory
        for ff in tqdm(
            files, desc=f"=====Loading NEXRAD #{ii+1}=====", total=len(files)
        ):

            if tracking_var.lower() == "dbz":

                # Create radar object including only field of interest
                radar = pyart.io.read_nexrad_archive(ff, include_fields="reflectivity")

                radar_list[ii].append(ff)

                # Add datetime of radar
                radar_time_list[ii].append(
                    pyart.util.datetime_from_radar(
                        radar,
                    )
                )

                del radar

            else:
                raise Exception(
                    f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
                )

    # First get all lenghts
    radar_lens = np.array([len(f) for f in radar_list])

    # If parallel processing, save grids multi
    if parallel_processing:

        # Start a pool with max_cores and run the grid function
        with multiprocessing.Pool(max_cores) as multi_pool:

            with tqdm(
                total=np.min(radar_lens),
                desc="=====Parallel Gridding Combined NEXRAD=====",
            ) as pbar:

                for _ in multi_pool.imap_unordered(
                    partial(
                        parallel_save_grid,
                        save_location=save_location,
                        radar_list=radar_list,
                        radar_time_list=radar_time_list,
                        CONFIG=CONFIG,
                    ),
                    np.arange(0, np.min(radar_lens)),
                ):

                    pbar.update()

        # Delete remaining radar arrays
        del radar_list
        del radar_time_list

        return

    # Now loop over shortest subset of radar objects

    # Find the shortest radar and get the index in the radar list of said radar
    shortest_radar_idx = np.where(radar_lens == np.min(radar_lens))[0][0]

    # Do this to make the for loop indexing slightly easier
    shortest_radar_list = radar_list[shortest_radar_idx]
    shortest_radar_time_list = radar_time_list[shortest_radar_idx]

    radar_list.pop(shortest_radar_idx)
    radar_time_list.pop(shortest_radar_idx)

    # Loop over shortest length
    for ii in tqdm(
        range(np.min(radar_lens)),
        desc="=====Gridding Combined NEXRAD Data=====",
        total=np.min(radar_lens),
    ):

        # Keep track of the radars we are merging and the average position of them
        merge_radar_list = []
        lats = []
        lons = []

        # Append the base radar
        base_radar = pyart.io.read_nexrad_archive(
            shortest_radar_list[ii], include_fields="reflectivity"
        )
        merge_radar_list.append(base_radar)
        lats.append(base_radar.latitude["data"][0])
        lons.append(base_radar.longitude["data"][0])

        # For each additional radar, add the nearest time and then its position
        for jj, additional_radar in enumerate(radar_list):

            # Grab the nearest radar object to the shortest iith one for every other radar
            nearest_time_idx = find_nearest(
                radar_time_list[jj], shortest_radar_time_list[ii]
            )

            addit_radar = pyart.io.read_nexrad_archive(
                additional_radar[nearest_time_idx], include_fields="reflectivity"
            )
            merge_radar_list.append(addit_radar)

            lats.append(addit_radar.latitude["data"][0])
            lons.append(addit_radar.longitude["data"][0])

            del addit_radar

        # Now merge the radars
        combined_grid = pyart.map.grid_from_radars(
            merge_radar_list,
            grid_origin=(np.mean(lats), np.mean(lons)),
            **CONFIG["multi_nexrad"]["gridding"],
        )

        # Save radar grid to save_location as a netcdf file
        pyart.io.write_grid(
            save_location
            + f"combined_grid_{shortest_radar_time_list[ii].strftime("%Y_%m_%d_%H:%M:%S")}.nc",
            combined_grid,
            arm_alt_lat_lon_variables=True,
            write_point_x_y_z=True,
            write_point_lon_lat_alt=True,
        )

        del base_radar
        del combined_grid
        del merge_radar_list

    # Delete remaining radar arrays
    del radar_list
    del radar_time_list

    return


def parallel_save_grid(ii, save_location, radar_list, radar_time_list, CONFIG):
    """
    Helper function for multi-processing
    """

    # Now loop over shortest subset of radar objects
    radar_lens = np.array([len(f) for f in radar_list])

    # Find the shortest radar and get the index in the radar list of said radar
    shortest_radar_idx = np.where(radar_lens == np.min(radar_lens))[0][0]

    # Do this to make the for loop indexing slightly easier
    shortest_radar_list = radar_list[shortest_radar_idx]
    shortest_radar_time_list = radar_time_list[shortest_radar_idx]

    radar_list.pop(shortest_radar_idx)
    radar_time_list.pop(shortest_radar_idx)

    # Keep track of the radars we are merging and the average position of them
    merge_radar_list = []
    lats = []
    lons = []

    # Append the base radar
    base_radar = pyart.io.read_nexrad_archive(
        shortest_radar_list[ii], include_fields="reflectivity"
    )
    merge_radar_list.append(base_radar)
    lats.append(base_radar.latitude["data"][0])
    lons.append(base_radar.longitude["data"][0])

    # For each additional radar, add the nearest time and then its position
    for jj, additional_radar in enumerate(radar_list):

        # Grab the nearest radar object to the shortest iith one for every other radar
        nearest_time_idx = find_nearest(
            radar_time_list[jj], shortest_radar_time_list[ii]
        )

        addit_radar = pyart.io.read_nexrad_archive(
            additional_radar[nearest_time_idx], include_fields="reflectivity"
        )
        merge_radar_list.append(addit_radar)

        lats.append(addit_radar.latitude["data"][0])
        lons.append(addit_radar.longitude["data"][0])

        del addit_radar

    # Now merge the radars
    combined_grid = pyart.map.grid_from_radars(
        merge_radar_list,
        grid_origin=(np.mean(lats), np.mean(lons)),
        **CONFIG["multi_nexrad"]["gridding"],
    )

    # Save radar grid to save_location as a netcdf file
    pyart.io.write_grid(
        save_location
        + f"""combined_grid_{shortest_radar_time_list[ii].strftime("%Y_%m_%d_%H:%M:%S")}.nc""",
        combined_grid,
        arm_alt_lat_lon_variables=True,
        write_point_x_y_z=True,
        write_point_lon_lat_alt=True,
    )

    del base_radar
    del combined_grid
    del merge_radar_list

    return


def multi_nexrad_load_netcdf_iris(
    paths_to_files: list[str] | str,
    file_type: str,
    tracking_var: str,
    CONFIG: dict,
    save_location: str | None = None,
) -> tuple[iris.cube.Cube, xr.DataArray]:
    """


    Parameters
    ----------
    paths_to_files : list[str] | str
        Array of glob path to input files, either archival or grided netcdf--i.e. ["/data/usr/KVNX*_V06.ar2v", "/data/usr/KIVX*_V06.ar2v"]. ONLY AN ARRAY WHEN NOT GRIDDED YET.
    file_type : str
        ["ar2v", "nc"] type of input file--either archival or netcdf.
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG : dict
        User configuration file.
    save_location : str, optional
        Where to save gridded NEXRAD data to if file_type=="ar2v". The default is None.

    Raises
    ------
    Exception
        Exception if missing MULTI-NEXRAD field in CONFIG, invalid tracking variable, and/or invalid file type.

    Returns
    -------
    nexrad_cube : iris.cube.Cube
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
        gen_and_save_multi_nexrad_grid(
            paths_to_files,
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

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity
            del radar_objects

            # Subset location and time of interest
            if "multi_nexrad" in CONFIG:

                if "min_frame" in CONFIG["multi_nexrad"]:
                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            CONFIG["multi_nexrad"]["min_frame"],
                            nexrad_xarray.dims["time"],
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["multi_nexrad"]:

                    mask_lon = (
                        nexrad_xarray.lon >= CONFIG["multi_nexrad"]["bounds"][0]
                    ) & (nexrad_xarray.lon <= CONFIG["multi_nexrad"]["bounds"][1])
                    mask_lat = (
                        nexrad_xarray.lat >= CONFIG["multi_nexrad"]["bounds"][2]
                    ) & (nexrad_xarray.lat <= CONFIG["multi_nexrad"]["bounds"][3])

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception('!=====CONFIG Missing "multi_nexrad" Field=====!')

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
            nexrad_xarray["projection_x_coordinate"] = (
                nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            )
            nexrad_xarray["projection_y_coordinate"] = (
                nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
            )

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

        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )

    # If data already grided, just return concated netcdf dataset
    if file_type.lower() == "nc":

        # Convert to iris cube and return
        if tracking_var.lower() == "dbz":

            # Read radar objects in and concat into one xarray datarray
            # This is a stupid hacky fix because pyart is dumb
            radar_objects = []
            for file in np.sort(glob.glob(paths_to_files)):

                # Ignore import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    radar_objects.append(pyart.io.read_grid(file).to_xarray())

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity
            del radar_objects

            # Subset location and time of interest
            if "multi_nexrad" in CONFIG:

                if "min_frame" in CONFIG["multi_nexrad"]:
                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            CONFIG["multi_nexrad"]["min_frame"],
                            nexrad_xarray.dims["time"],
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["multi_nexrad"]:

                    mask_lon = (
                        nexrad_xarray.lon >= CONFIG["multi_nexrad"]["bounds"][0]
                    ) & (nexrad_xarray.lon <= CONFIG["multi_nexrad"]["bounds"][1])
                    mask_lat = (
                        nexrad_xarray.lat >= CONFIG["multi_nexrad"]["bounds"][2]
                    ) & (nexrad_xarray.lat <= CONFIG["multi_nexrad"]["bounds"][3])

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception('!=====CONFIG Missing "multi_nexrad" Field=====!')

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
            nexrad_xarray["projection_x_coordinate"] = (
                nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            )
            nexrad_xarray["projection_y_coordinate"] = (
                nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
            )

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

    raise Exception(f"!=====Invalid File Type. You Entered: {file_type.lower()}=====!")


def multi_nexrad_load_netcdf(
    paths_to_files: list[str] | str,
    file_type: str,
    tracking_var: str,
    CONFIG: dict,
    save_location: str | None = None,
) -> xr.DataArray:
    """


    Parameters
    ----------
    paths_to_files : list[str] | str
        Array of glob path to input files, either archival or grided netcdf--i.e. ["/data/usr/KVNX*_V06.ar2v", "/data/usr/KIVX*_V06.ar2v"]. ONLY AN ARRAY WHEN NOT GRIDDED YET.
    file_type : str
        ["ar2v", "nc"] type of input file--either archival or netcdf.
    tracking_var : str
        ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG : dict
        User configuration file.
    save_location : str, optional
        Where to save gridded NEXRAD data to if file_type=="ar2v". The default is None.

    Raises
    ------
    Exception
        Exception if missing MULTI-NEXRAD field in CONFIG, invalid tracking variable, and/or invalid file type.

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
        gen_and_save_multi_nexrad_grid(
            paths_to_files,
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

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity
            del radar_objects

            # Subset location and time of interest
            if "multi_nexrad" in CONFIG:

                if "min_frame" in CONFIG["multi_nexrad"]:
                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            CONFIG["multi_nexrad"]["min_frame"],
                            nexrad_xarray.dims["time"],
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["multi_nexrad"]:

                    mask_lon = (
                        nexrad_xarray.lon >= CONFIG["multi_nexrad"]["bounds"][0]
                    ) & (nexrad_xarray.lon <= CONFIG["multi_nexrad"]["bounds"][1])
                    mask_lat = (
                        nexrad_xarray.lat >= CONFIG["multi_nexrad"]["bounds"][2]
                    ) & (nexrad_xarray.lat <= CONFIG["multi_nexrad"]["bounds"][3])

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception('!=====CONFIG Missing "nexrad" Field=====!')

            # Replace time dimension with minutes since first time and add other x y z coords
            first_time = nexrad_xarray.time.values[0]
            nexrad_xarray = nexrad_xarray.assign_coords(
                time=cftime.date2num(
                    nexrad_xarray.time.values, f"minutes since {first_time}"
                ),
                south_north=("y", np.arange(nexrad_xarray.shape[2])),
                west_east=("x", np.arange(nexrad_xarray.shape[2])),
                projection_x_coordinate=("x", nexrad_xarray.x.values),
                projection_y_coordinate=("y", nexrad_xarray.y.values),
                x=("x", np.arange(nexrad_xarray.shape[2])),
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
            nexrad_xarray["projection_x_coordinate"] = (
                nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            )
            nexrad_xarray["projection_y_coordinate"] = (
                nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
            )
            nexrad_xarray["altitude"] = nexrad_xarray.altitude.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )

            return nexrad_xarray

        else:
            raise Exception(
                f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
            )

    # If data already grided, just return concated netcdf dataset
    if file_type.lower() == "nc":

        if tracking_var.lower() == "dbz":

            # Read radar objects in and concat into one xarray datarray
            # This is a stupid hacky fix because pyart is dumb
            radar_objects = []
            for file in np.sort(glob.glob(paths_to_files)):

                # Ignore import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    radar_objects.append(pyart.io.read_grid(file).to_xarray())

            nexrad_xarray = xr.concat(radar_objects, dim="time").reflectivity
            del radar_objects

            # Subset location and time of interest
            if "multi_nexrad" in CONFIG:

                if "min_frame" in CONFIG["multi_nexrad"]:
                    nexrad_xarray = nexrad_xarray.isel(
                        time=np.arange(
                            CONFIG["multi_nexrad"]["min_frame"],
                            nexrad_xarray.dims["time"],
                        ),
                        drop=True,
                    )

                if "bounds" in CONFIG["multi_nexrad"]:

                    mask_lon = (
                        nexrad_xarray.lon >= CONFIG["multi_nexrad"]["bounds"][0]
                    ) & (nexrad_xarray.lon <= CONFIG["multi_nexrad"]["bounds"][1])
                    mask_lat = (
                        nexrad_xarray.lat >= CONFIG["multi_nexrad"]["bounds"][2]
                    ) & (nexrad_xarray.lat <= CONFIG["multi_nexrad"]["bounds"][3])

                    nexrad_xarray = nexrad_xarray.where(mask_lon & mask_lat, drop=True)

            else:
                raise Exception('!=====CONFIG Missing "multi_nexrad" Field=====!')

            # Replace time dimension with minutes since first time and add other x y z coords
            first_time = nexrad_xarray.time.values[0]
            nexrad_xarray = nexrad_xarray.assign_coords(
                time=cftime.date2num(
                    nexrad_xarray.time.values, f"minutes since {first_time}"
                ),
                south_north=("y", np.arange(nexrad_xarray.shape[2])),
                west_east=("x", np.arange(nexrad_xarray.shape[2])),
                projection_x_coordinate=("x", nexrad_xarray.x.values),
                projection_y_coordinate=("y", nexrad_xarray.y.values),
                x=("x", np.arange(nexrad_xarray.shape[2])),
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
            nexrad_xarray["projection_x_coordinate"] = (
                nexrad_xarray.projection_x_coordinate.assign_attrs({"units": "m"})
            )
            nexrad_xarray["projection_y_coordinate"] = (
                nexrad_xarray.projection_y_coordinate.assign_attrs({"units": "m"})
            )
            nexrad_xarray["altitude"] = nexrad_xarray.altitude.assign_attrs(
                {"standard_name": "altitude", "units": "m"}
            )

            return nexrad_xarray

        raise Exception(
            f"!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!"
        )

    raise Exception(f"!=====Invalid File Type. You Entered: {file_type.lower()}=====!")
