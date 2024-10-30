#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:18:18 2024

@author: thahn
"""

# =============================================================================
# This file contains the functions used to calculate additional values from MesoNH output. Including reflectivity and altitudes
# =============================================================================


def mesonh_calculate_reflectivity(mesonh_xarray):
    """
    Inputs:
        mesonh_xarray: xarray Dataset containing default MesoNH values
    Ouputs:
        dBZ: DataArray containing calculated reflectivity values
    """

    import warnings
    import numpy as np

    # Get variables from MesoNH
    t = mesonh_xarray["T"]
    p = mesonh_xarray["P"]
    qv = mesonh_xarray["qv"]
    qr = mesonh_xarray["qr"]
    qs = mesonh_xarray["qs"]
    qg = mesonh_xarray["qg"]

    # Calculate proper pressures and actual temperature
    full_p = p
    tmk = t

    # Suppress divide by zero warnings
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
            "coordinates": "lon lat time",
        }
    )

    return dBZ.chunk(t.chunksizes)


def mesonh_calculate_brightness_temp(mesonh_xarray):
    """
    Inputs:
        mesonh_xarray:xarray Dataset containing default MesoNH values
    Outputs:
        TB: numpy array containing brightness temperature at each point and time--same dimension as input
    """

    import numpy as np
    from tqdm import tqdm

    OLR = mesonh_xarray["LWup_TOA"].values

    TB = np.empty(OLR.shape)

    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8  # W m^-2 K^-4

    for tt, ix, iy in tqdm(
        np.ndindex(OLR.shape),
        desc="=====Calculating MesoNH Brightness Temperatures=====",
        total=np.prod(OLR.shape),
    ):
        tf = (OLR[tt, ix, iy] / sigma) ** 0.25
        TB[tt, ix, iy] = (-a + np.sqrt(a**2 + 4 * b * tf)) / (2 * b)

    return TB


def mesonh_calculate_agl_z(mesonh_xarray):
    """
    Inputs:
        mesonh_xarray: xarray Dataset containing default MesoNH values
    Outputs:
        geopt: Dataarray of heights AGL in m
    """

    return mesonh_xarray.Z[0].squeeze()
