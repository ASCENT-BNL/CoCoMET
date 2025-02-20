# =============================================================================
# This file contains the functions used to calculate additional values from rams output
# =============================================================================

import numpy as np
import xarray as xr
from tqdm import tqdm

# Calculate nearest item in list to given pivot
def find_nearest(array: np.ndarray, pivot) -> int:
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx

# TODO: This file needs a lot more explanation
def rams_calculate_brightness_temp(rams_xarray: xr.Dataset) -> xr.DataArray:
    """


    Parameters
    ----------
    rams_xarray : xarray.core.dataset.Dataset
        Xarray Dataset containing default RAMS values.

    Returns
    -------
    TB_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray containing brightness temperature at each point and time--same dimension as input [K].

    """

    rams_xarray["TOA_OLR"] = (
        ["Time", "south_north", "west_east"],
        rams_xarray["LWUP"][:, -1, :, :].values,
    )  # the top of atmosphere surface radiation, needed to TB calculation
    TOA_OLR = rams_xarray["TOA_OLR"].values

    TB = np.zeros(TOA_OLR.shape)

    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8  # W m^-2 K^-4

    print("=====Calculating RAMS Brightness Temperatures=====")
    tf = (TOA_OLR[:, :, :] / sigma) ** 0.25
    TB[:, :, :] = (np.sqrt(4 * b * tf + a**2) - a) / (2 * b)

    TB_xarray = xr.DataArray(TB, dims=["Time", "south_north", "west_east"])
    return TB_xarray


def rams_calculate_precip_rate(
    rams_xarray: xr.Dataset, height : int
) -> xr.DataArray:
    """


    Parameters
    ----------
    rams_xarray : xarray.core.dataset.Dataset
        Xarray Dataset containing default RAMS values.
    height : int
        The height to calculate the precipitation rate at

    Raises
    ------
        ValueError if the height is None

    Returns
    -------
    total2D_precip_rate_xarray : xarray.core.dataarray.DataArray
        Xarray DataArray of precipitation rate in mm/hr.

    """
    # If the height is not at ground level, calculate the index it should be
    if height != 0:
        altitudes = rams_xarray["altitudes"].values
        height_indx = find_nearest(altitudes, height * 1000)
    # If the height is ground level, take the second level, since the first is unphysical
    else:
        height_indx = 1

    if height is not None:
        water_precip_rate = rams_xarray["PCPVR"].values[:, height_indx, :, :] * (1 / 1000) * (1000) * (60 * 60) # take the water precipitation rate and divide by the density of water (kg/ m^2 /s -> m/s), convert m->mm, then convert 1/s -> 1/hr
        snow_precip_rate = rams_xarray["PCPVS"].values[:, height_indx, :, :] * (1 / 1000) * (1000) * (60 * 60) # take the snow precipitation rate and divide by the density of snow, convert m->mm, then convert 1/s -> 1/hr 
        aggregate_precip_rate = rams_xarray["PCPVA"].values[:, height_indx, :, :] * (1 / 1000) * (1000) * (60 * 60)# take the aggregate precipitation rate and divide by the average density of aggregates, convert m->mm, then convert 1/s -> 1/hr 
        graupel_precip_rate = rams_xarray["PCPVG"].values[:, height_indx, :, :] * (1 / 1000) * (1000) * (60 * 60)# take the graupel precipitation rate and divide by the average density of graupel, convert m->mm, then convert 1/s -> 1/hr 
        hail_precip_rate = rams_xarray["PCPVH"].values[:, height_indx, :, :] * (1 / 1000) * (1000) * (60 * 60)# take the hail precipitation rate and divide by the average density of hails, convert m->mm, then convert 1/s -> 1/hr 
        drizzle_precip_rate = rams_xarray["PCPVD"].values[:, height_indx, :, :] * (1 / 1000) * (1000) * (60 * 60)# take the water precipitation rate and divide by the average density of water, convert m->mm, then convert 1/s -> 1/hr
    
        print('=====Calculating RAMS Precipitation Rate=====')

        total2D_precip_rate = water_precip_rate + snow_precip_rate + aggregate_precip_rate + graupel_precip_rate + hail_precip_rate + drizzle_precip_rate

        total2D_precip_rate_xarray = xr.DataArray(
            total2D_precip_rate, dims=["Time", "south_north", "west_east"]
        )
    
    else:
        raise ValueError("Height cannot be None")

    return total2D_precip_rate_xarray.chunk(rams_xarray["TOPT"].chunksizes)


# Why does this function exist?
def rams_calculate_wa(rams_xarray: xr.Dataset) -> xr.DataArray:
    """


    Parameters
    ----------
    rams_xarray : xarray.core.dataset.Dataset
        xarray Dataset containing default RAMS values.

    Returns
    -------
    wa : xarray.core.dataarray.DataArray
        Xarray DataArray of vertical wind components at mass points.

    """

    wa = rams_xarray["WC"]  # rams winds are unstaggered in the header file already
    wa = wa.assign_attrs(
        {
            "units": "m s-1",
            "coordinates": "GLON GLAT Times",
            "description": "updraft velocity",
            "MemoryOrder": "XYZ",
        }
    )
    return wa


# Citation needed for this methodology
def rams_calculate_reflectivity(rams_xarray: xr.Dataset) -> xr.DataArray:
    """


    Parameters
    ----------
    rams_xarray : xarray.core.dataset.Dataset
        Xarray Dataset containing default RAMS values.

    Returns
    -------
    dBZ: xarray.core.dataarray.DataArray
        Xarray DataArray of calculated reflectivity values.

    """

    # constants defined by RAMS
    p00 = 1.0e5  # Pa
    rgas = 287  # J/gk/K
    cp = 1004  # J/kg/K
    cpor = cp / rgas

    # find temperature, pressure, and grid point density
    T_pot = rams_xarray["THETA"]  # potential temperature
    PI = rams_xarray["PI"]
    P = ((PI / cp) ** cpor) * p00 * 0.01  # convert pressure to mb
    T = T_pot * PI / cp  # temperature in K
    dens = (P * 100) / (T * rgas)

    # mass coefficients defined by RAMS
    alpha_mr = 524.0  # rain mass coeff
    alpha_mg = 157.0  # graupel mass coeff
    alpha_mh = 471.0  # hail mass coeff
    alpha_mp = 110.8  # pris mass coeff
    alpha_ms = 2.739e-3  # snow mass coeff
    alpha_ma = 0.496  # aggregates mass coeff

    # need mixing ratios and concentrations and ensure they are above 0
    rainMix = xr.where(rams_xarray["RRP"] < 0, 0, rams_xarray["RRP"])
    pris_iceMix = xr.where(rams_xarray["RPP"] < 0, 0, rams_xarray["RPP"])
    snowMix = xr.where(rams_xarray["RSP"] < 0, 0, rams_xarray["RSP"])
    aggregatesMix = xr.where(rams_xarray["RAP"] < 0, 0, rams_xarray["RAP"])
    graupelMix = xr.where(rams_xarray["RGP"] < 0, 0, rams_xarray["RGP"])
    hailMix = xr.where(rams_xarray["RHP"] < 0, 0, rams_xarray["RHP"])
    rainConc = xr.where(rams_xarray["CRP"] < 0, 0, rams_xarray["CRP"])
    pris_iceConc = xr.where(rams_xarray["CPP"] < 0, 0, rams_xarray["CPP"])
    snowConc = xr.where(rams_xarray["CSP"] < 0, 0, rams_xarray["CSP"])
    aggregatesConc = xr.where(rams_xarray["CAP"] < 0, 0, rams_xarray["CAP"])
    graupelConc = xr.where(rams_xarray["CGP"] < 0, 0, rams_xarray["CGP"])
    hailConc = xr.where(rams_xarray["CHP"] < 0, 0, rams_xarray["CHP"])

    # define the gamma shape parameters
    gamma = rams_xarray.gnu  # in order: cld rain pris snow aggr graup hail driz

    dicToIterateThrough = {
        "rain": [alpha_mr, rainMix, rainConc, gamma[1]],
        "graupel": [alpha_mg, graupelMix, graupelConc, gamma[5]],
        "pris_ice": [alpha_mp, pris_iceMix, pris_iceConc, gamma[2], 2.91],
        "snow": [alpha_ms, snowMix, snowConc, gamma[3], 1.74],
        "hail": [alpha_mh, hailMix, hailConc, gamma[6]],
        "aggregates": [alpha_ma, aggregatesMix, aggregatesConc, gamma[4], 2.4],
    }

    # define an array of the reflectivity and iteratively add to it
    dbz_total = np.zeros_like(rainMix)
    Z_total = np.zeros_like(rainMix)

    # thresholds for the mixing ratios and concentration numbers
    q = 1e-10  # kg / kg
    qn = 1e-3  # (# / kg)

    for key in tqdm(
        dicToIterateThrough.keys(),
        desc="=====Calculating RAMS Reflectivity=====",
        total=len(dicToIterateThrough.keys()),
    ):
        alpha = dicToIterateThrough[key][0]
        mixRatio = dicToIterateThrough[key][1]
        conc = dicToIterateThrough[key][2]
        gsp = dicToIterateThrough[key][3]

        F_gnu1 = (5.0 + gsp) * (4.0 + gsp) * (3.0 + gsp)
        F_gnu2 = (2.0 + gsp) * (1.0 + gsp) * gsp
        F_gnu = F_gnu1 / F_gnu2

        # if the mixing ratio or number concentration is below a threshold there is no contribution to reflectivity
        mixRatiobool = mixRatio > q
        concbool = conc > qn
        FullBool = mixRatiobool * concbool

        if key in ["pris_ice", "snow", "aggregates"]:
            factor = dicToIterateThrough[key][4]
            M = mixRatio / conc
            D = (M / alpha) ** (1 / factor)
            alpha = M / (D**3)
        tmp0 = mixRatio / alpha
        tmp1 = (tmp0**2) * dens * F_gnu
        tmp2 = (tmp1 / conc) * 1.0e18
        tmp2 = np.where(FullBool, tmp2, 0)

        Z_total += tmp2

    Z_total = np.clip(Z_total, 0.001, 1e99)
    np.nan_to_num(Z_total, copy=False, nan=0.001)

    dbz_total = 10 * np.log10(Z_total)

    dbz_total_xarray = xr.DataArray(
        dbz_total, dims=["Time", "bottom_top", "south_north", "west_east"]
    )
    # Assign attributes
    dBZ = dbz_total_xarray.assign_attrs(
        {
            "FieldType": 104,
            "MemoryOrder": "XYZ",
            "description": "radar reflectivity",
            "units": "dBZ",
            "stagger": "",
            "coordinates": "Times GLON GLAT",
        }
    )

    return dBZ.chunk(T.chunksizes)
