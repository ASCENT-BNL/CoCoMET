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
# Loads in pre-gridded radar data which follows the radar standardization set out in the CoMET-UDAF Section S1.1.
# =============================================================================


"""
Inputs:
    path_to_files: Glob path to gridded input netcdf files--i.e. "/data/usr/KVNX*_V06.nc"
    tracking_var: ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG: User configuration file
Outputs:
    radar_cube: iris cube continaing gridded reflectivity data ready for tobac tracking 
    radar_xarray: Xarray dataset containing gridded reflectivity data
"""


def standard_radar_load_netcdf_iris(path_to_files, tracking_var, CONFIG):
    import xarray as xr

    # Convert to iris cube and return
    if tracking_var.lower() == "dbz":

        # Open combined netcdf radar dataarray
        radar_xarray = xr.open_mfdataset(
            path_to_files, concat_dim="time", combine="nested"
        ).reflectivity

        # Subset location of interest
        if "standard_radar" in CONFIG:

            if "bounds" in CONFIG["standard_radar"]:

                mask_lon = (
                    radar_xarray.lon >= CONFIG["standard_radar"]["bounds"][0]
                ) & (radar_xarray.lon <= CONFIG["standard_radar"]["bounds"][1])
                mask_lat = (
                    radar_xarray.lat >= CONFIG["standard_radar"]["bounds"][2]
                ) & (radar_xarray.lat <= CONFIG["standard_radar"]["bounds"][3])

                radar_xarray = radar_xarray.where(mask_lon & mask_lat, drop=True)

        else:
            raise Exception('!=====CONFIG Missing "standard_radar" Field=====!')

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

        # Drop altitude coordinate temporarily when making iris cube
        radar_xarray = radar_xarray.drop_vars(["altitude"])
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


"""
Inputs:
    path_to_files: Glob path to gridded input netcdf files--i.e. "/data/usr/KVNX*_V06.nc"
    tracking_var: ["dbz"], variable which is going to be used for tracking--reflectivity.
    CONFIG: User configuration file
Outputs:
    radar_xarray: Xarray dataset containing gridded reflectivity data
"""


def standard_radar_load_netcdf(path_to_files, tracking_var, CONFIG):
    import xarray as xr

    # Convert to iris cube and return
    if tracking_var.lower() == "dbz":

        # Open combined netcdf radar dataarray
        radar_xarray = xr.open_mfdataset(
            path_to_files, concat_dim="time", combine="nested"
        ).reflectivity

        # Subset location of interest
        if "standard_radar" in CONFIG:

            if "bounds" in CONFIG["standard_radar"]:

                mask_lon = (
                    radar_xarray.lon >= CONFIG["standard_radar"]["bounds"][0]
                ) & (radar_xarray.lon <= CONFIG["standard_radar"]["bounds"][1])
                mask_lat = (
                    radar_xarray.lat >= CONFIG["standard_radar"]["bounds"][2]
                ) & (radar_xarray.lat <= CONFIG["standard_radar"]["bounds"][3])

                radar_xarray = radar_xarray.where(mask_lon & mask_lat, drop=True)

        else:
            raise Exception('!=====CONFIG Missing "standard_radar" Field=====!')

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
