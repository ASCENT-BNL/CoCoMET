# =============================================================================
# This defines the methods for running MOAAP on RAMS data processed using rams_load.py
# =============================================================================

import os

import numpy as np
import pandas as pd
import xarray as xr

from CoCoMET.MOAAP import moaap

from .rams_calculate_products import (
    rams_calculate_brightness_temp,
    rams_calculate_precip_rate,
    rams_calculate_reflectivity,
    rams_calculate_wa,
)


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def rams_run_moaap(rams_xarray, CONFIG):
    """
    Inputs:
        rams_xarray: xarray Dataset containing RAMS data calculated from rams_load.py
        CONFIG: User configuration file
    Outputs:
        mask_file: The xarray object containing the default MOAAP outputs
    """

    # Get basic setup variables including lat/lon, delta time, a pandas time range vector (TODO: adjust output to )
    latitudes = rams_xarray.GLAT[0].values
    longitudes = rams_xarray.GLON[0].values
    dt = rams_xarray.DT
    times = pd.date_range(
        start=rams_xarray["Time"].values[0],
        end=rams_xarray["Time"].values[-1],
        freq=str(dt) + "min",
    )
    mask = np.ones(latitudes.shape)

    # Get all necessary variables from RAMS output to input into MOAAP

    # Get pressure heights
    PI = rams_xarray["PI"]
    p00 = 1e5  # Pa
    cp = 1004  # J / kg / K
    Rd = 287  # J / kg / K

    total_p = (PI / cp) ** (Rd / cp) * p00 / 100
    avg_geo_pres = np.mean(total_p[0, :, :, :].values, axis=(1, 2))

    # Get height_idx of 850hPA, 500hPA, and 200hPA
    height_idx_850 = find_nearest(avg_geo_pres, 850)
    height_idx_500 = find_nearest(avg_geo_pres, 500)
    height_idx_200 = find_nearest(avg_geo_pres, 200)

    # get destaggered 850hPA wind speeds and 200hPA wind speeds
    v_winds_850 = (rams_xarray["VC"][:, height_idx_850, :, :]).values
    u_winds_850 = (rams_xarray["UC"][:, height_idx_850, :, :]).values
    v_winds_200 = (rams_xarray["VC"][:, height_idx_200, :, :]).values
    u_winds_200 = (rams_xarray["UC"][:, height_idx_200, :, :]).values

    # Calculate proper pressures and actual temperature
    full_t = rams_xarray["THETA"]

    air_temp = (full_t * (total_p / 1e5) ** (287.0 / 1004.5))[:, height_idx_850].values

    # Get geopt
    geopt = rams_xarray["geopt"]

    # Get brightness temp
    tb = (
        rams_xarray.TB.values
        if "TB" in rams_xarray
        else rams_calculate_brightness_temp(rams_xarray)
    )

    # Get total mixing ratio at 850hPA
    mr = (
        rams_xarray.RDP
        + rams_xarray.RCP
        + rams_xarray.RRP
        + rams_xarray.RPP
        + rams_xarray.RSP
        + rams_xarray.RGP
        + rams_xarray.RHP
        + rams_xarray.RAP
    )[
        :, height_idx_850
    ] * 1000  # kg / kg -> g / kg

    # Get precipitation rate
    pr = rams_calculate_precip_rate(rams_xarray) / (60 / rams_xarray.DT)

    # Create output folder if it does not already exist
    if not os.path.exists(CONFIG["rams"]["moaap"]["tracking_save_path"]):
        os.makedirs(CONFIG["rams"]["moaap"]["tracking_save_path"])

    moaap(
        longitudes,
        latitudes,
        times,
        dt / 3600,  # I think it's supposed to be in hours
        mask,
        DataName="CoMET_RAMS_MOAAP_TRACKING",
        OutputFolder=CONFIG["rams"]["moaap"]["tracking_save_path"],
        # Data Variables
        v850=v_winds_850,
        u850=u_winds_850,
        t850=air_temp,
        q850=mr,
        slp=total_p[:, 0, :, :],
        ivte=None,
        ivtn=None,
        z500=geopt.values[:, height_idx_500],
        v200=v_winds_200,
        u200=u_winds_200,
        pr=pr,
        tb=tb,
        # Any user defined params
        **CONFIG["rams"]["moaap"],
    )

    if CONFIG["rams"]["moaap"]["tracking_save_path"][-1] != "/":
        CONFIG["rams"]["moaap"]["tracking_save_path"] += "/"

    output_filepath = (
        CONFIG["rams"]["moaap"]["tracking_save_path"]
        + str(times[0].year)
        + str(times[0].month).zfill(2)
        + "_CoMET_RAMS_MOAAP_TRACKING_ObjectMasks_dt-%.2f" % dt
        + "min_MOAAP-masks"
        + ".nc"
    )

    mask_file = xr.open_mfdataset(
        output_filepath,
        coords="all",
        concat_dim="time",
        combine="nested",
        decode_time=False,
    )

    return mask_file
