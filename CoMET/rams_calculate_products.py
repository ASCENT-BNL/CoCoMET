# =============================================================================
# This file contains the functions used to calculate additional values from rams output
# =============================================================================

def rams_calculate_brightness_temp(rams_xarray):
    """
    Inputs:
        rams_xarray (xarray) - xarray Dataset containing default RAMS values
    Outputs:
        TB (numpy array) - array containing brightness temperature at each point and time--same dimension as input [K]
    """
    import numpy as np
    import xarray as xr
    rams_xarray["TOA_OLR"] = (['Time', 'south_north', 'west_east'], rams_xarray["LWUP"][:, -1, :, :].values) # the top of atmosphere surface radiation, needed to TB calculation
    TOA_OLR = rams_xarray["TOA_OLR"].values

    TB = np.zeros(TOA_OLR.shape)

    a = 1.228
    b = -1.106e-3
    sigma = 5.67e-8  # W m^-2 K^-4

    print('=====Calculating RAMS Brightness Temperatures=====')
    tf = (TOA_OLR[:, :, :] / sigma) ** 0.25
    TB[:, :, :] = (np.sqrt(4 * b * tf + a**2) - a) / (2 * b)

    TB_xarray = xr.DataArray(TB, dims=["Time", "south_north", "west_east"])
    return TB_xarray

def rams_calculate_precip_rate(rams_xarray, pr_calc = 'integral'):
    """
    Inputs:
        rams_xarray: xarray Dataset containing default rams values
    Outputs:
        pr: Numpy array of precipitation rate in mm/hr
    """
    import numpy as np
    import xarray as xr

    if pr_calc == '3d':
        # Do you want to take the three dimensional rates and sum over them?
        water_precip_rate = rams_xarray["PCPVR"].values * (1 / 1000) * (1000) * (60 * 60) # take the water precipitation rate and divide by the density of water, convert m->mm, then convert 1/s -> 1/hr
        snow_precip_rate = rams_xarray["PCPVS"].values * (1 / 1000) * (1000) * (60 * 60) # take the snow precipitation rate and divide by the density of snow, convert m->mm, then convert 1/s -> 1/hr 
        aggregate_precip_rate = rams_xarray["PCPVA"].values * (1 / 1000) * (1000) * (60 * 60)# take the aggregate precipitation rate and divide by the average density of aggregates, convert m->mm, then convert 1/s -> 1/hr 
        graupel_precip_rate = rams_xarray["PCPVG"].values * (1 / 1000) * (1000) * (60 * 60)# take the graupel precipitation rate and divide by the average density of graupel, convert m->mm, then convert 1/s -> 1/hr 
        hail_precip_rate = rams_xarray["PCPVH"].values * (1 / 1000) * (1000) * (60 * 60)# take the hail precipitation rate and divide by the average density of hails, convert m->mm, then convert 1/s -> 1/hr 
        drizzle_precip_rate = rams_xarray["PCPVD"].values * (1 / 1000) * (1000) * (60 * 60)# take the water precipitation rate and divide by the average density of water, convert m->mm, then convert 1/s -> 1/hr
        print('=====Calculating RAMS Precipitation Rate=====')

        total3D_precip_rate = water_precip_rate + snow_precip_rate + aggregate_precip_rate + graupel_precip_rate + hail_precip_rate + drizzle_precip_rate
        total2D_precip_rate = np.sum(total3D_precip_rate, axis=1)

    elif pr_calc == 'integral':
        # Aryeh's calculation suggestion
        DN0 = rams_xarray['DN0']
            # liquid water path
        cloudMix = rams_xarray['RCP']
        drizzleMix = rams_xarray['RDP']
        rainMix = rams_xarray['RRP']
            # ice water path
        pris_iceMix = rams_xarray['RPP']
        snowMix = rams_xarray['RSP']
        aggregatesMix = rams_xarray['RAP']
        graupelMix = rams_xarray['RGP']
        hailMix = rams_xarray['RHP']

        total2D_precip_rate = np.sum((cloudMix + drizzleMix + rainMix + pris_iceMix + snowMix + aggregatesMix + graupelMix + hailMix) * DN0, axis=1)

    total2D_precip_rate_xarray = xr.DataArray(total2D_precip_rate, dims=["Time", "south_north", "west_east"])
    return total2D_precip_rate_xarray

def rams_calculate_wa(rams_xarray):
    """
    Inputs:
        rams_xarray: xarray Dataset containing default WRF values
    Outputs:
        wa: Dataarray of vertical wind components at mass points
    """

    wa = rams_xarray['WC'] # rams winds are unstaggered in the header file already
    wa = wa.assign_attrs(
        {
            "units": "m s-1",
            "coordinates": "GLON GLAT Times",
            "description": "updraft velocity",
            "MemoryOrder": "XYZ",
        }
    )
    return wa

def rams_calculate_reflectivity(rams_xarray):

    import numpy as np
    from tqdm import tqdm
    import xarray as xr

    # constants defined by RAMS
    p00 = 1.e5 # Pa
    rgas = 287 # J/gk/K
    cp = 1004 # J/kg/K
    cpor = cp / rgas

    # find temperature, pressure, and grid point density
    T = rams_xarray["THETA"]
    P = (rams_xarray["PI"] / cp)**cpor * p00 * 0.01 # convert to mb
    dens = (P * 100) / (T * rgas)

    # mass coefficients defined by RAMS
    alpha_mr=524.0    #rain mass coeff
    alpha_mg=157.0    #graupel mass coeff
    alpha_mh=471.0    #hail mass coeff
    alpha_mp=110.8    #pris mass coeff
    alpha_ms=2.739e-3 #snow mass coeff
    alpha_ma=0.496    #aggregates mass coeff

    # need mixing ratios and concentrations
    rainMix = rams_xarray['RRP']
    pris_iceMix = rams_xarray['RPP']
    snowMix = rams_xarray['RSP']
    aggregatesMix = rams_xarray['RAP']
    graupelMix = rams_xarray['RGP']
    hailMix = rams_xarray['RHP']
    rainConc = rams_xarray['CRP']
    pris_iceConc = rams_xarray['CPP']
    snowConc = rams_xarray['CSP']
    aggregatesConc = rams_xarray['CAP']
    graupelConc = rams_xarray['CGP']
    hailConc = rams_xarray['CHP']

    # define the gamma shape parameters
    gamma = rams_xarray.gnu #in order: cld rain pris snow aggr graup hail driz

    dicToIterateThrough = {
        'rain': [alpha_mr, rainMix, rainConc, gamma[1]],
        'graupel': [alpha_mg, graupelMix, graupelConc, gamma[5]],
        'pris_ice': [alpha_mp, pris_iceMix, pris_iceConc, gamma[2], 2.91],
        'snow': [alpha_ms, snowMix, snowConc, gamma[3], 1.74], 
        'hail': [alpha_mh, hailMix, hailConc, gamma[6]],
        'aggregates': [alpha_ma, aggregatesMix, aggregatesConc, gamma[4], 2.4],
        }

    # define an array of the reflectivity and iteratively add to it
    dbz_total = np.zeros_like(rainMix)
    Z_total = np.zeros_like(rainMix)

    # thresholds for the mixing ratios and concentration numbers
    q = 1e-10 # kg / kg
    qn = 1e-3 # (# / kg)

    for key in tqdm(
        dicToIterateThrough.keys(),
        desc="=====Calculating RAMS Reflectivity=====",
        total=len(dicToIterateThrough.keys())
        ):

        alpha = dicToIterateThrough[key][0]
        mixRatio = dicToIterateThrough[key][1]
        conc = dicToIterateThrough[key][2]
        gsp = dicToIterateThrough[key][3]

        F_gnu1 = (5. + gsp) * (4. + gsp) * (3. + gsp)
        F_gnu2 = (2. + gsp) * (1. + gsp) * gsp
        F_gnu = F_gnu1 / F_gnu2

        # if the mixing ratio or number concentration is below a threshold there is no contribution to reflectivity
        mixRatiobool = mixRatio > q
        concbool = conc > qn
        FullBool = mixRatiobool * concbool

        if key in ['pris_ice', 'snow', 'aggregates']:
            factor = dicToIterateThrough[key][4]
            M = mixRatio / conc
            D = (M / alpha)**(1 / factor)
            alpha = M / (D**3)
        tmp0 = mixRatio / alpha
        tmp1 = (tmp0**2) * dens * F_gnu
        tmp2 = (tmp1 / conc) * 1.e18
        tmp2 = np.where(FullBool, tmp2, 0)

        Z_total+= tmp2

    Z_total = np.clip(Z_total, 0.001, 1e99)
    np.nan_to_num(Z_total, copy=False, nan=0.001)

    dbz_total = 10 * np.log10(Z_total)

    dbz_total_xarray = xr.DataArray(dbz_total, dims=['Time', 'bottom_top', 'south_north', 'west_east'])
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