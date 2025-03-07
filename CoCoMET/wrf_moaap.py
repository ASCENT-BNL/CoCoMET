#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:29:56 2024

@author: thahn
"""

# =============================================================================
# This defines the methods for running MOAAP on WRF data processed using wrf_load.py
# =============================================================================


import os

import numpy as np
import pandas as pd
import xarray as xr

from CoCoMET.MOAAP import moaap

from .wrf_calculate_products import (
    wrf_calculate_brightness_temp,
    wrf_calculate_precip_rate,
)


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def wrf_run_moaap(wrf_xarray: xr.Dataset, CONFIG: dict) -> xr.Dataset:
    """


    Parameters
    ----------
    wrf_xarray : xarray.core.dataset.Dataset
        Xarray Dataset containing WRF data calculated from wrf_load.py.
    CONFIG : dict
        User configuration file.

    Returns
    -------
    mask_file : xarray.core.dataset.Dataset
        The xarray object containing the default MOAAP outputs.

    """

    # Get basic setup variables including lat/lon, delta time, a pandas time range vector (TODO: adjust output to )
    latitudes = wrf_xarray.XLAT[0].values
    longitudes = wrf_xarray.XLONG[0].values
    dt = wrf_xarray.DT # in s
    times = pd.date_range(
        start=wrf_xarray.XTIME[0].values,#.astype('datetime64[s]'),
        end=wrf_xarray.XTIME[-1].values,#.astype('datetime64[s]'),
        freq=str(dt / 60) + "min",
    )
    mask = np.ones(latitudes.shape)

    # Get all necessary variables from WRF output to input into MOAAP

    # Get pressure heights
    p = wrf_xarray["P"]
    pb = wrf_xarray["PB"]

    total_p = (p + pb) / 100
    avg_geo_pres = [np.mean(h.values) for h in total_p[0]]

    # Get height_idx of 850hPA, 500hPA, and 200hPA
    height_idx_850 = find_nearest(avg_geo_pres, 850)
    height_idx_500 = find_nearest(avg_geo_pres, 500)
    height_idx_200 = find_nearest(avg_geo_pres, 200)

    # get destaggered 850hPA wind speeds and 200hPA wind speeds
    v_winds_850 = (
        0.5 * wrf_xarray["V"][:, height_idx_850, 1:, :]
        + 0.5 * wrf_xarray["V"][:, height_idx_850, :-1, :]
    ).values
    u_winds_850 = (
        0.5 * wrf_xarray["U"][:, height_idx_850, :, 1:]
        + 0.5 * wrf_xarray["U"][:, height_idx_850, :, :-1]
    ).values
    v_winds_200 = (
        0.5 * wrf_xarray["V"][:, height_idx_200, 1:, :]
        + 0.5 * wrf_xarray["V"][:, height_idx_200, :-1, :]
    ).values
    u_winds_200 = (
        0.5 * wrf_xarray["U"][:, height_idx_200, :, 1:]
        + 0.5 * wrf_xarray["U"][:, height_idx_200, :, :-1]
    ).values

    # Get 850hPA air temperature
    t = wrf_xarray["T"]

    # Calculate proper pressures and actual temperature
    full_t = t + 300
    full_p = p + pb

    air_temp = (full_t * (full_p / 1e5) ** (287.0 / 1004.5))[:, height_idx_850].values

    # Get geopotential heights
    ph = wrf_xarray["PH"]
    phb = wrf_xarray["PHB"]
    geopt = ph + phb

    # DESTAGGER geopt
    geopt = 0.5 * geopt[:, 1:] + 0.5 * geopt[:, :-1]

    # Get brightness temp
    tb = (
        wrf_xarray.TB.values
        if "TB" in wrf_xarray
        else wrf_calculate_brightness_temp(wrf_xarray)
    )

    # Get total mixing ratio at 850hPA
    mr = (
        wrf_xarray.QVAPOR
        + wrf_xarray.QCLOUD
        + wrf_xarray.QRAIN
        + wrf_xarray.QICE
        + wrf_xarray.QSNOW
        + wrf_xarray.QGRAUP
    )[:, height_idx_850]

    # Get precipitation rate
    pr = wrf_calculate_precip_rate(wrf_xarray) / (3600 / wrf_xarray.DT)

    # Create output folder if it does not already exist
    if not os.path.exists(CONFIG["wrf"]["moaap"]["tracking_save_path"]):
        os.makedirs(CONFIG["wrf"]["moaap"]["tracking_save_path"])

    moaap(
        longitudes,
        latitudes,
        times,
        dt / 3600, # convert to hr
        mask,
        DataName="CoMET_WRF_MOAAP_TRACKING",
        OutputFolder=CONFIG["wrf"]["moaap"]["tracking_save_path"],
        # Data Variables
        v850=v_winds_850,
        u850=u_winds_850,
        t850=air_temp,
        q850=mr,
        slp=full_p[:, 0],
        ivte=None,
        ivtn=None,
        z500=geopt.values[:, height_idx_500],
        v200=v_winds_200,
        u200=u_winds_200,
        pr=pr,
        tb=tb,
        # Any user defined params
        **CONFIG["wrf"]["moaap"],
    )

    output_filepath = (
        CONFIG["wrf"]["moaap"]["tracking_save_path"]
        + str(times[0].year)
        + str(times[0].month).zfill(2)
        + "_CoMET_WRF_MOAAP_TRACKING_ObjectMasks_dt-%.2f" % (dt / 60)
        + "min_MOAAP-masks"
        + ".nc"
    )

    mask_file = xr.open_mfdataset(
        output_filepath,
        coords="all",
        concat_dim="time",
        combine="nested",
        decode_times=False,
    )

    return mask_file
