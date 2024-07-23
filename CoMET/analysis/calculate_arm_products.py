#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:55:18 2024

@author: thahn
"""

# =============================================================================
# This file contains the functions used to calculate statistics and data related to ARM products
# =============================================================================


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def extract_arm_product(analysis_object, path_to_files, variable_names, **args):
    """
    

    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks.
    path_to_files : string
        A glob-like path to the VDISQUANTS ARM product output.
    variable_name : list of strings
        Case sensitive name of variable you want to extract from the ARM data

    Returns
    -------
    output_data : xarray.core.dataset.Dataset
        An xarray Dataset with the following: frame, tracking_time, vdisquants_time, time_delta, closest_feature_id (km), variable_names list

    """
    
    import numpy as np
    import xarray as xr
    from tqdm import tqdm
    from vincenty import vincenty

    # Ensure input is a list
    if (type(variable_names) != list): variable_names = [variable_names]
    
    # Open vap product
    vap = xr.open_mfdataset(
        path_to_files, coords="all", concat_dim="time", combine="nested"
    )
    
    vap_info = {
        "frame": [],
        "tracking_time": [],
        "vdisquants_time": [],
        "time_delta": [],
        "closest_feature_id": [],
        "closest_cell_id": [],
        "distance_to_closest_feature": [],
    }
    
    for var in variable_names:
        vap_info[var] = []

    frame_groups = analysis_object["UDAF_linking"].groupby("frame")

    # If 1D input data, don't use heights
    if (len(vap.dims) == 1):

        # Loop over frames
        for ii, frame in tqdm(
            enumerate(frame_groups),
            desc="=====Extracting ARM Data=====",
            total=frame_groups.ngroups,
        ):
    
            # Get VAP at current time step
            time_idx = find_nearest(vap.time.values, frame[1].time.values[0])
            time_delta = abs(vap.time.values[time_idx] - frame[1].time.values[0])
    
            # Get the position of the device
            lat_pos = vap.lat.values[time_idx]
            lon_pos = vap.lon.values[time_idx]
    
            feature_id = []
            cell_distance = []
    
            # Loop over features
            for feature in frame[1].groupby("feature_id"):
    
                # Get distance from each feature to the device
                dis_to_vdis = vincenty(
                    (lat_pos, lon_pos),
                    (feature[1].latitude.values, feature[1].longitude.values),
                )
                cell_distance.append(dis_to_vdis)
                feature_id.append(feature[0])
    
            closest_feature_id = feature_id[
                np.where(cell_distance == np.nanmin(cell_distance))[0][0]
            ]
            closest_feature = frame[1].query("feature_id==@closest_feature_id")
    
            vap_info["frame"].append(frame[0])
            vap_info["tracking_time"].append(frame[1].time.values[0])
            vap_info["vdisquants_time"].append(vap.time.values[time_idx])
            vap_info["time_delta"].append(time_delta)
            vap_info["closest_feature_id"].append(closest_feature.feature_id.values[0])
            vap_info["closest_cell_id"].append(closest_feature.cell_id.values[0])
            vap_info["distance_to_closest_feature"].append(np.nanmin(cell_distance))
            
            for var in variable_names:
                vap_info[var].append(vap[var].values[time_idx])


        data_vars = {
            "time_delta":("time", vap_info["time_delta"]),
            "closest_feature_id":("time", vap_info["closest_feature_id"]),
            "closest_cell_id":("time", vap_info["closest_cell_id"]),
            "distance_to_closest_feature":(
                "time",
                vap_info["distance_to_closest_feature"],
            ),
        }
        
        for var in variable_names:
            data_vars[var] = ("time", vap_info[var])

        # Create output Dataset
        output_data = xr.Dataset(
            coords=dict(
                frame=("time", vap_info["frame"]),
                tracking_time=("time", vap_info["tracking_time"]),
                vdisquants_time=("time", vap_info["vdisquants_time"]),
            ),
            data_vars=data_vars,
            attrs=dict(description=f"ARM Data {vap.input_datastreams}, Variables: {variable_names}"),
        )
    
        return output_data
    
    
    # If 2D data, add heights
    # Loop over frames
    for ii, frame in tqdm(
        enumerate(frame_groups),
        desc="=====Extracting ARM Data=====",
        total=frame_groups.ngroups,
    ):

        # Get VAP at current time step
        time_idx = find_nearest(vap.time.values, frame[1].time.values[0])
        time_delta = abs(vap.time.values[time_idx] - frame[1].time.values[0])

        # Get the position of the device
        lat_pos = vap.lat.values[time_idx]
        lon_pos = vap.lon.values[time_idx]

        feature_id = []
        cell_distance = []

        # Loop over features
        for feature in frame[1].groupby("feature_id"):

            # Get distance from each feature to the device
            dis_to_vdis = vincenty(
                (lat_pos, lon_pos),
                (feature[1].latitude.values, feature[1].longitude.values),
            )
            cell_distance.append(dis_to_vdis)
            feature_id.append(feature[0])

        closest_feature_id = feature_id[
            np.where(cell_distance == np.nanmin(cell_distance))[0][0]
        ]
        closest_feature = frame[1].query("feature_id==@closest_feature_id")

        vap_info["frame"].append(frame[0])
        vap_info["tracking_time"].append(frame[1].time.values[0])
        vap_info["vdisquants_time"].append(vap.time.values[time_idx])
        vap_info["time_delta"].append(time_delta)
        vap_info["closest_feature_id"].append(closest_feature.feature_id.values[0])
        vap_info["closest_cell_id"].append(closest_feature.cell_id.values[0])
        vap_info["distance_to_closest_feature"].append(np.nanmin(cell_distance))
        
        for var in variable_names:
            vap_info[var].append(vap[var].values[time_idx,:])


    data_vars = {
        "time_delta":("time", vap_info["time_delta"]),
        "closest_feature_id":("time", vap_info["closest_feature_id"]),
        "closest_cell_id":("time", vap_info["closest_cell_id"]),
        "distance_to_closest_feature":(
            "time",
            vap_info["distance_to_closest_feature"],
        ),
    }
    
    for var in variable_names:
        data_vars[var] = (["time", "height"], vap_info[var])

    # Create output Dataset
    output_data = xr.Dataset(
        coords=dict(
            frame=("time", vap_info["frame"]),
            tracking_time=("time", vap_info["tracking_time"]),
            vdisquants_time=("time", vap_info["vdisquants_time"]),
            height=((vap.height.values) * 1000 - vap.alt.values.min())
        ),
        data_vars=data_vars,
        attrs=dict(description=f"ARM Data {vap.input_datastreams}, Variables: {variable_names}"),
    )

    return output_data


def calculate_arm_interpsonde(analysis_object, path_to_files, **args):
    """
    

    Parameters
    ----------
    analysis_object : dict
        A CoMET-UDAF standard analysis object containing at least UDAF_tracks.
    path_to_files : string
        A glob-like path to the VDISQUANTS ARM product output.
    **args : dict, optional
        Parameters to pass to the calculations of convective initation properties.

    Returns
    -------
    sonde_output_data : xarray.core.dataset.Dataset
        An xarray Dataset with the following: frame, tracking_time, vdisquants_time, height (m AGL), time_delta, closest_feature_id (km), temperature (C), relative_humidity (%),
                                                            barometric_pressure (hPA), wind_speed (m/s), wind_direction (degrees), northward_wind (m/s), eastward_wind (m/s).
    sonde_output_init_data : xarray.core.dataset.Dataset
        An xarray Dataset with the following: sonde_time, ...
    """
    
    import numpy as np
    import xarray as xr
    from tqdm import tqdm
    from vincenty import vincenty
    from .calculate_convective_properties import (
        calculate_interp_sonde_convective_properties,
    )

    # Open video disdrometer product
    sonde = xr.open_mfdataset(
        path_to_files, coords="all", concat_dim="time", combine="nested"
    )

    sonde_info = {
        "frame": [],
        "tracking_time": [],
        "sonde_time": [],
        "time_delta": [],
        "closest_feature_id": [],
        "closest_cell_id": [],
        "distance_to_closest_feature": [],
        "temperature": [],
        "relative_humidity": [],
        "barometric_pressure": [],
        "wind_speed": [],
        "wind_direction": [],
        "northward_wind": [],
        "eastward_wind": [],
    }

    frame_groups = analysis_object["UDAF_linking"].groupby("frame")

    # Loop over frames
    for ii, frame in tqdm(
        enumerate(frame_groups),
        desc="=====Calculating INTERPSONDE Data=====",
        total=frame_groups.ngroups,
    ):

        # Get sonde at current time step
        time_idx = find_nearest(sonde.time.values, frame[1].time.values[0])
        time_delta = abs(sonde.time.values[time_idx] - frame[1].time.values[0])

        # Get the position of the sonde
        lat_pos = sonde.lat.values[time_idx]
        lon_pos = sonde.lon.values[time_idx]

        feature_id = []
        cell_distance = []

        # Loop over features
        for feature in frame[1].groupby("feature_id"):

            # Get distance from each feature to the video disdrometer
            dis_to_sonde = vincenty(
                (lat_pos, lon_pos),
                (feature[1].latitude.values, feature[1].longitude.values),
            )
            cell_distance.append(dis_to_sonde)
            feature_id.append(feature[0])

        closest_feature_id = feature_id[
            np.where(cell_distance == np.nanmin(cell_distance))[0][0]
        ]
        closest_feature = frame[1].query("feature_id==@closest_feature_id")

        sonde_info["frame"].append(frame[0])
        sonde_info["tracking_time"].append(frame[1].time.values[0])
        sonde_info["sonde_time"].append(sonde.time.values[time_idx])
        sonde_info["time_delta"].append(time_delta)
        sonde_info["closest_feature_id"].append(closest_feature.feature_id.values[0])
        sonde_info["closest_cell_id"].append(closest_feature.cell_id.values[0])
        sonde_info["distance_to_closest_feature"].append(np.nanmin(cell_distance))
        sonde_info["temperature"].append(sonde.temp.values[time_idx])
        sonde_info["relative_humidity"].append(sonde.rh.values[time_idx])
        sonde_info["barometric_pressure"].append(sonde.bar_pres.values[time_idx] * 10)
        sonde_info["wind_speed"].append(sonde.wspd.values[time_idx])
        sonde_info["wind_direction"].append(sonde.wdir.values[time_idx])
        sonde_info["northward_wind"].append(sonde.v_wind.values[time_idx])
        sonde_info["eastward_wind"].append(sonde.u_wind.values[time_idx])

    # Create sonde data output Dataset
    sonde_output_data = xr.Dataset(
        coords=dict(
            frame=("time", sonde_info["frame"]),
            tracking_time=("time", sonde_info["tracking_time"]),
            sonde_time=("time", sonde_info["sonde_time"]),
            height=((sonde.height.values) * 1000 - sonde.alt.values.min()),
        ),
        data_vars=dict(
            time_delta=("time", sonde_info["time_delta"]),
            closest_feature_id=("time", sonde_info["closest_feature_id"]),
            closest_cell_id=("time", sonde_info["closest_cell_id"]),
            distance_to_closest_feature=(
                "time",
                sonde_info["distance_to_closest_feature"],
            ),
            temperature=(["time", "height"], sonde_info["temperature"]),
            relative_humidity=(["time", "height"], sonde_info["relative_humidity"]),
            barometric_pressure=(["time", "height"], sonde_info["barometric_pressure"]),
            wind_speed=(["time", "height"], sonde_info["wind_speed"]),
            wind_direction=(["time", "height"], sonde_info["wind_direction"]),
            northward_wind=(["time", "height"], sonde_info["northward_wind"]),
            eastward_wind=(["time", "height"], sonde_info["eastward_wind"]),
        ),
        attrs=dict(description="Tracking Linked INTERPSONDE Data"),
    )

    convective_init_info = {
        "sonde_time": [],
        "cell_id": [],
        "cape": [],
        "ncape": [],
        "cin": [],
        "lnb": [],
        "lfc": [],
        "lcl": [],
        "wind_shear": [],
        "low_rh": [],
        "mid_rh": [],
        "richardson": [],
        "elr_0_3": [],
    }

    init_groups = (
        analysis_object["UDAF_linking"].query("lifetime_percent==0").groupby("cell_id")
    )

    # Loop over cell initations
    for ii, cell in tqdm(
        enumerate(init_groups),
        desc="=====Calculating INTERPSONDE Initation Properties=====",
        total=init_groups.ngroups,
    ):

        # Get sonde data at time step before initiation if possible
        time_idx = find_nearest(sonde.time.values, cell[1].time.values[0])
        time_idx_init = time_idx - 1 if time_idx > 0 else 0

        # Calculate CAPE, CIN, etc.
        properties = calculate_interp_sonde_convective_properties(
            sonde.isel(time=time_idx_init), **args
        )

        convective_init_info["sonde_time"].append(sonde.time.values[time_idx_init])
        convective_init_info["cell_id"].append(cell[0])
        convective_init_info["cape"].append(properties["CAPE"])
        convective_init_info["ncape"].append(properties["NCAPE"])
        convective_init_info["cin"].append(properties["CIN"])
        convective_init_info["lnb"].append(properties["LNB"])
        convective_init_info["lfc"].append(properties["LFC"])
        convective_init_info["lcl"].append(properties["LCL"])
        convective_init_info["wind_shear"].append(properties["Wind_Shear"])
        convective_init_info["low_rh"].append(properties["Low_RH"])
        convective_init_info["mid_rh"].append(properties["Mid_RH"])
        convective_init_info["richardson"].append(properties["Richardson"])
        convective_init_info["elr_0_3"].append(properties["ELR_0-3km"])

    # Create sonde data output Dataset
    sonde_output_init_data = xr.Dataset(
        coords=dict(cell_id=convective_init_info["cell_id"]),
        data_vars=dict(
            sonde_time=("cell_id", convective_init_info["sonde_time"]),
            cape=("cell_id", convective_init_info["cape"]),
            ncape=("cell_id", convective_init_info["ncape"]),
            cin=("cell_id", convective_init_info["cin"]),
            lnb=("cell_id", convective_init_info["lnb"]),
            lfc=("cell_id", convective_init_info["lfc"]),
            lcl=("cell_id", convective_init_info["lcl"]),
            wind_shear=("cell_id", convective_init_info["wind_shear"]),
            low_rh=("cell_id", convective_init_info["low_rh"]),
            mid_rh=("cell_id", convective_init_info["mid_rh"]),
            richardson=("cell_id", convective_init_info["richardson"]),
            elr_0_3=("cell_id", convective_init_info["elr_0_3"]),
        ),
        attrs=dict(
            description="Convective properties calculated at the time before the initation of cell formations"
        ),
    )

    return (sonde_output_data, sonde_output_init_data)
