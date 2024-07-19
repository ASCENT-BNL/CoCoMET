#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:01:10 2024

@author: thahn
"""

# =============================================================================
# This file contains the functions used to calculate additional values from wrf output. Including reflectivity, mass point updrafts, and altitudes
# =============================================================================


def wrf_calculate_reflectivity(wrf_xarray):
    """
    Inputs:
        wrf_xarray: xarray Dataset containing default WRF values
    Ouputs:
        dBZ: DataArray containing calculated reflectivity values
    """

    import warnings
    import numpy as np

    # Get variables from WRF
    t = wrf_xarray["T"]
    p = wrf_xarray["P"]
    pb = wrf_xarray["PB"]
    qv = wrf_xarray["QVAPOR"]
    qr = wrf_xarray["QRAIN"]
    qs = wrf_xarray["QSNOW"]
    qg = wrf_xarray["QGRAUP"]

    # Calculate proper pressures and actual temperature
    full_t = t + 300
    full_p = p + pb
    tmk = full_t * (full_p / 1e5) ** (287.0 / 1004.5)

    # Supress divide by zero warnings
    with warnings.catch_warnings():

        warnings.filterwarnings(
            action="ignore", message="divide by zero encountered in divide"
        )

        # Calculate density of dry air at points
        virtual_t = tmk * (0.622 + qv) / (0.622 * (1 + qv))

        dry_air_density = full_p / (287 * virtual_t)

        # Slope intercept constants
        N0r = 8e6
        N0s = 2e7
        N0g = 4e6

        # Calculate individual reflectivites for different types of particles
        gamma_r = ((np.pi * N0r * 1000) / (qr * dry_air_density)) ** (1 / 4)
        Z_er = 720 * N0r * (gamma_r ** (-7)) * 1e18

        gamma_s = ((np.pi * N0s * 100) / (qs * dry_air_density)) ** (1 / 4)
        Z_es = 161.28 * N0s * (gamma_s ** (-7)) * ((100 / 1000) ** 2) * 1e18

        gamma_g = ((np.pi * N0g * 400) / (qg * dry_air_density)) ** (1 / 4)
        Z_eg = 161.28 * N0g * (gamma_g ** (-7)) * ((400 / 1000) ** 2) * 1e18

        # Sum them up
        Z_e = Z_er + Z_es + Z_eg

        # Make sure minimum value is -30dBZ and remove any NaNs (replace NaNs with -30dBZ)
        Z_e.values = np.clip(Z_e.values, 0.001, 1e99)
        np.nan_to_num(Z_e.values, copy=False, nan=0.001)

        dBZ = 10 * np.log10(Z_e)

    # Assign attributes
    dBZ = dBZ.assign_attrs(
        {
            "FieldType": 104,
            "MemoryOrder": "XYZ",
            "description": "radar reflectivity",
            "units": "dBZ",
            "stagger": "",
            "coordinates": "XLONG XLAT XTIME",
        }
    )

    return dBZ.chunk(t.chunksizes)


def wrf_calculate_brightness_temp(wrf_xarray):
    """
    Inputs:
        wrf_xarray:xarray Dataset containing default WRF values
    Outputs:
        TB: numpy array containing brightness temperature at each point and time--same dimension as input
    """

    import numpy as np
    from tqdm import tqdm

    OLR = wrf_xarray["OLR"].values

    TB = np.empty(OLR.shape)

    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8  # W m^-2 K^-4

    for tt, ix, iy in tqdm(
        np.ndindex(OLR.shape),
        desc="=====Calculating WRF Brightness Temperatures=====",
        total=np.prod(OLR.shape),
    ):
        tf = (OLR[tt, ix, iy] / sigma) ** 0.25
        TB[tt, ix, iy] = (-a + np.sqrt(a**2 + 4 * b * tf)) / (2 * b)

    return TB


def wrf_calculate_agl_z(wrf_xarray):
    """
    Inputs:
        wrf_xarray: xarray Dataset containing default WRF values
    Outputs:
        geopt: Dataarray of heights AGL
    """

    ph = wrf_xarray["PH"]
    phb = wrf_xarray["PHB"]
    hgt = wrf_xarray["HGT"][0].squeeze().values

    # Make sure we only take one time dimension
    geopt = (ph + phb)[0].squeeze()

    # DESTAGGER geopt
    geopt = 0.5 * geopt[1:] + 0.5 * geopt[:-1]
    geopt = geopt.rename(bottom_top_stag="bottom_top")

    # Account for terrain to convert from MSL to AGL, hence the - hgt
    return (geopt / 9.81) - hgt


def wrf_calculate_wa(wrf_xarray):
    """
    Inputs:
        wrf_xarray: xarray Dataset containing default WRF values
    Outputs:
        wa: Dataarray of vertical wind components at mass points
    """

    # Destagger vertical winds
    wa = (0.5 * wrf_xarray["W"][:, 1:] + 0.5 * wrf_xarray["W"][:, :-1]).rename(
        bottom_top_stag="bottom_top"
    )
    wa = wa.assign_attrs(
        {
            "units": "m s-1",
            "coordinates": "XLONG XLAT XTIME",
            "description": "updraft velocity",
            "MemoryOrder": "XYZ",
        }
    )

    return wa


def wrf_calculate_precip_rate(wrf_xarray):
    """
    Inputs:
        wrf_xarray: xarray Dataset containing default WRF values
    Outputs:
        pr: Numpy array of precipitation rate in mm/hr
    """

    import numpy as np
    from tqdm import tqdm

    total_precip = (wrf_xarray.RAINC + wrf_xarray.RAINNC).values
    precip_rate = np.zeros(total_precip.shape)

    for ii in tqdm(
        range(total_precip.shape[0] - 1),
        desc="=====Calculating WRF Precipitation Rate=====",
        total=total_precip.shape[0] - 1,
    ):

        precip_rate[ii] = (total_precip[ii + 1] - total_precip[ii]) * (
            60 / wrf_xarray.DT
        )

    return precip_rate
