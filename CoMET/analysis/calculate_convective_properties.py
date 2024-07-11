#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:23:21 2024

@author: thahn
"""

# NAME: calc_cape.m (v1.0)
#
# PURPOSE: Determine the CAPE, CIN, LCL, LFC, LNB, and W of a parcel using
# parcel theory. User defines how to determine parcel starting properies
# and which processes to include (e.g., freezing, deposition, and
# hydrometeor loading).
#   
# INPUT: Requires data that contains pressure, height, temperature, and
# either relative humidity or dew point (Declared in 'Input Data' section).
#
# OUTPUT: Displays the CAPE, CIN, LCL, LFC, LNB, and Wmax of the parcel
# (variables: cape, cin, lcl, lfc, lnb, w_p, respectively). Also generates
# vectors of the parcel sensible temperature and vertical velocity with
# height (variables: temp_log and w_log, respectively).
#
# NOTES:
#   -Can select suface based, mixed-layer, or most unstable parcel defined
#   in 'User Settings' section.
#   -Surface-based parcel defined as: lowest possible data point.
#   -Mixed-layer parcel defined as: parcel with properties of the mean of
#   the mixed-layer that is a user defined depth in 'User Settings'
#   section [Default: 1.5 km]. Parcel starts at surface.
#   -Most-Unstable parcel defined as: parcel that has
#   the greatest Theta-E (using Bolton 1980) below a certain pressure level
#   that is a user defined level in 'User Settings' section [Default:
#   700mb].
#   -Can have pseudoadiabatic/irreversible (i.e., all hydrometeors(water/ice
#   particles formed from condensation) fall out) ascent or moist adiabatic
#   /reversible (i.e,, all hydrometeors are retained with parcel) ascent.
#   pseudoadiabatic meaning 
#   -Can have liquid-only or liquid and ice processes. Ice processes
#   include deposition and can include freezing (if loading turned on).
#   -Ice processes are very simplistic, and condensation/deposition rate is
#   based on a linear function of temperature. At 0 deg C: all extra mass
#   resulting from a supersaturated environment is condensed into water. At
#   minus 40 deg C: all extra mass is deposited into ice.
#   -Freezing follows same temperature function as depositon, but only
#   works when liquid hydrometeors are present.
#   -Parcel ascent includes accounting for subsaturated conditions
#   (even though these currently should not happen). 
#   -Variables in the code are all defined when first introduced; however,
#   generally variables ending in '_e' represent the environment and '_p'
#   represent the parcel.
#   -Parcel ascent is similar to Bryan and Fritsch (2002).
#
# UPDATE (YYYYMMDD): 
#   20180410 - Written Matlab - Mariusz Starzec (M.S.) [v1.0]
#   20190206 - Python 3.6 version - Die Wang 
#              read in ARM sounding
#              add in wind shear, midlevel RH, ELR, etc
#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

#Without ice:
#For pseudo-adiabatic (irreversible) ascent, you assume no hydrometeor loading/drag (flag_adiabat = 1, flag_heat =1).
#For moist-adiabatic (reversible) ascent, you retain all the hydrometeors with the parcel (flag_adiabt = 2, flag_heat = 1). This setting essentially retains all hydrometeors and used them to except drag on the parcel. The drag is simplistic, in that it is essentially considered as negative buoyancy where you take the mass of liquid water and apply gravity to it, but it does have a very strong effect.

#With ice:
#For pseudo-adiabatic ascent, only latent heat from deposition (i.e., water vapor to ice) is added in (flag_adiabat = 1, flag_heat = 2).
#For moist adiabatic ascent, there are two options, both include hydrometer loading from both water and ice hydrometeors:
#Only latent heat from deposition is added in (flag_adiabat = 2, flag_heat = 2).
#Latent heat from deposition and freezing of pre-existing liquid water is added in (flag_adiabat = 2, flag_heat = 2, flag_convert = 1). Freezing of pre-existing water can only happen when flag_adiabat = 2, otherwise it has no effect, since you need to carry hydrometeors with you if you want to be able to freeze them. However, this process is overly intense, and ends up releasing a significant amount of heat (which is honestly not as realistic since we end up converting all the hydrometeors to ice which is too much). I really wouldn’t use this option if you are going for realism.

# =========================

import numpy as np

def find_closest(A, target):
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def calc_tv(t,q):
    ''' Virt. pot. temp. of environment'''
    return t*(1.0+0.61*q)

def calc_r(e,p):
    '''# Water vapor mixing ratio [kg kg-1]'''
    return (0.622*e)/(p-e)

def calc_es(t):
    '''# Vapor pressure - Mangus equation [Pa]''' 
    return 610.94*np.exp((17.625*t)/(243.04+t))

def calc_shear(wspd, wdir, alt):
    '''wind shear (vector)'''
    index = find_closest(alt, 5000)
    a = wspd[0]
    b = wspd[index]
    c = wdir[0]
    d = wdir[index]
    return (a ** 2 + b ** 2 - 2 * a * b * np.cos(np.radians(abs(d - c)))) ** 0.5

def calc_speed_shear(wspd, alt):
    '''wind speed shear'''
    a = np.nanmean(wspd[(alt>0) & (alt<1000)])
    b = np.nanmean(wspd[(alt>4000) & (alt<5000)])    
    return b-a

def calc_midlevel_rh(rh,alt):
    return np.nanmean(rh[(alt>3000) & (alt<6000)])

def calc_lowlevel_rh(rh,alt):
    return np.nanmean(rh[(alt>0) & (alt<3000)])

def calc_richardson(theta_v,u,v,alt):
    '''Bulk Richardson Number
       buoyancy / wind shear
       Buoyancy tends to suppress turbulence, while the wind shear tends to generate turbulence mechanically.'''
    index = find_closest(alt, 6000)
    mask  = (alt>0) & (alt<=500)
    theta_v0 = np.nanmean(theta_v[mask])
    delta_thetav = theta_v[index]-np.nanmean(theta_v[mask]) 
    delta_z = alt[index]
    delta_u = u[index]-np.nanmean(u[mask])
    delta_v = v[index]-np.nanmean(v[mask])
    return (9.81*delta_thetav*delta_z)/(np.nanmean(theta_v[mask])*(delta_u**2+delta_v**2))

def calc_elr(T,alt):
    # sfc-3km
    ind_3km = find_closest(alt, 3000)
    # find the maximum T st lower altitudes 
    ind_max = np.argmax(T[:ind_3km])
    lap_3km = -(T[ind_3km]-T[ind_max])/(alt[ind_3km]-alt[ind_max])*1e3
    # 3km-6km    
    ind_6km = find_closest(alt, 6000)
    lap_6km = -(T[ind_6km]-T[ind_3km])/(alt[ind_6km]-alt[ind_3km])*1e3
    return round(lap_3km,1), round(lap_6km,1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Input Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Input data (ARM sounding) 
# Need Pressure, Height, Temperature and either RH or Dew point.
# Note: Variable names and units need to stay the same. Final units are in
# brackets (e.g., finalized pressure needs to be in Pa, Temp in C, etc.)

# test on the original file
#data = ascii.read('smodel_fine.txt')
#p_e = np.array(data['col1']) 
#p_e = p_e*100.   # Pressure [Pa]
#z   = np.array(data['col2']) # Height [m]
#ts_e = np.array(data['col3']) # Temperature [C]
#rh_e = np.array(data['col4']) 
#rh_e = rh_e/100               # Relative humidity
#td_e = np.array(data['col4'])


def calculate_interp_sonde_convective_properties(sonde, 
                                                 parcel=3, 
                                                 ml_depth=482, 
                                                 mu_depth=700.0, 
                                                 start=0,
                                                 flag_heat=2,
                                                 flag_convert=0,
                                                 flag_adiabat=1,
                                                 **args):

    tdryo  = sonde["temp"].values    # Dry Bulb Temperature [C]
    p_e  = sonde["bar_pres"].values*1000      # Pressure [Pa]
    rh_eo = sonde["rh"].values      # Relative Humidity [%]
    rh_eo = rh_eo/100  
    z    = sonde["height"].values*1000      # Altitude above mean sea level [m]
    #td_e = ncfile.variables["dp"][:]

    ts_e = tdryo
    rh_e = rh_eo
       

    u_wind = sonde["u_wind"].values # eastward wind comp [m/s]
    v_wind = sonde["v_wind"].values # northward wind comp [m/s]
    wspd   = sonde["wspd"].values   # wind speed [m/s]
    wdir   = sonde["wdir"].values    # wind direction [degree]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% User Settings %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Which parcel to use?
    # 1 - Surface / 2 - Mixed-Layer / 3 - Most Unstable
    parcel = parcel
    # depth of mixed-layer [meters] (when parcel = 2)
    ml_depth = ml_depth
    # look below this pressure level for MU parcel [hPa] (when parcel = 3)
    mu_depth = mu_depth

    # Index to start at for determining parcel properties (e.g. 1 = z(1)).
    # This can be used to remove surface data or bad data at low levels.
    # Example: If want to ignore first 4 data levels completely, set start = 5.
    start = start

    # Latent heating only due to liquid or liquid and ice?
    # 1 - Liquid only / 2 - Liquid and Ice
    flag_heat = flag_heat # To follow Parcel Theory = 1;
    # Convert pre-existing cloud water to cloud ice?
    # 0 = No / 1 - Yes (using linear function of temperature)
    # Requires flag_adiabat = 2 below (so we have hydrometeors to freeze).
    # Can have a very pronounced effect just above the freezing level.
    flag_convert = flag_convert # To follow Parcel Theory = 0;

    # Reversible or Irreversible parcel path?
    # 1 - Pseudoadiabatic (Irreversible) / 2 - Moist Adiabatic (Reversible)
    flag_adiabat = flag_adiabat # To follow Parcel Theory = 1;

    #%%%%%%%%%%%%%%%%%%%%%%%%%%% Minor Error Checks %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Heights in meters or km? (Should be in meters)
    #if max(z) < 100.:
    #    warning('Expected heights in meters; Assuming data is in kilometers')
    #    z = z*1000.

    # Correct parcel flag?
    #if (parcel>3) | (parcel<1): 
    #    error('Incorrect parcel flag') 

    # Start index?
    #if start>len(z):
    #    error('Start index exceeds max index')

    # Reasonable depths if ML/MU parcels chosen?
    #if ((parcel==2) & (ml_depth >=max(z))) | ((parcel==3) & (mu_depth >max(z))):
    #    error('ML/MU parcel depths do not match height data')

    # Correct heating flag?
    #if (flag_heat < 1) | (flag_heat > 2):
    #    error('Incorrect heating flag')

    # Correct parcel adiabat?
    #if (flag_adiabat <1) | (flag_adiabat >2):
    #    error('Incorrect adiabat flag')

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Constants
    g = 9.81 # gravity [m s-2]
    cpd = 1004.0 # specific heat of air [J kg-1 K-1]
    ci = 2106.0 # specific heat of ice water [J kg-1 K-1]
    cl = 4186.0 # specific heat of liquid water [J kg-1 K-1]
    cpv = 1885.0 # specific heat of water vapor [J kg-1 K-1]
    Rd = 287.0 # gas constant of dry air [J kg-1 K-1]
    Rv = 461.0 # gas constant of water vapor [J kg-1 K-1]
    eps = Rd/Rv # epsilon 
    lv0 = 2.5e6 # reference latent heat of vaporization [J kg-1]
    lf0 = 3.337e5 # reference latent heat of fusion [J kg-1]
    ls0 = lf0+lv0 # reference latent heat of sublimation [J kg-1]
    p0 = 1.e5 # reference pressure [Pa]
    t0 = 273.15 # reference temperature [K]

    # Exner function
    pii = (p_e/p0)**(Rd/cpd)

    # Environmental Attributes
    # Vapor pressure - Mangus equation [Pa]
    es_e = calc_es(ts_e)

    # If we have RH but not Dew Point
    # This prioritizes RH if both RH and dew point exist since RH is usually
    # the provided/measured variable from soundings and not derived
    #if rh_e.any():
    e_e  = rh_e*es_e
    td_e = np.log(e_e/610.94)*243.04/(17.625-np.log(e_e/610.94))

    # If we have Dew Point but not RH
    #elif td_e.any():
    #e_e  = calc_es(td_e)
    #rh_e = e_e/es_e

    # Water vapor mixing ratio [kg kg-1]
    r_e = calc_r(e_e,p_e)
    # Saturation Water vapor mixing ratio [kg kg-1]
    rs_e = calc_r(es_e,p_e)
    # Specific humidity [kg kg-1]
    q_e = r_e/(1+r_e)
    # Saturation Specific humidity [kg kg-1]
    qs_e = rs_e/(1+rs_e)
    # Convert environmental temperature; [C] to [K]
    ts_e = ts_e+273.15
    # Pot. temp.
    t_e = ts_e/pii
    # Virt. pot. temp. of environment
    tv_e = calc_tv(t_e,q_e)

    # Get info regarding the starting location/heights of parcel
    if parcel == 1: # surface parcel
        depth_index = start
    elif parcel == 2: # mixed-layer parcel
        depth_index = np.where(z <= ml_depth)
        print (depth_index[0])
        if len(depth_index[0]) < start:
            depth_index[0] = []
        if len(depth_index[0]) < 1:
            print ('error: Mixed-layer depth and starting index are incorrect.',
                   ' Values result in no starting parcel.')
        elif len(depth_index[0]) == 1:
            print ('warning: Only 1 starting height is valid for mixed-layer parcel')
    elif parcel == 3: # most unstable parcel
        # Get Theta-E of environment to choose parcel
        # Temperature of lcl (Bolton 1980)
        tlcl_e = 56.0+(1.0/((1.0/((td_e+t0)-56.0))+(np.log(ts_e/(td_e+t0))/800.0)))
        # Theta-E of the environment (Bolton 1980)
        thetae_e = t_e*np.exp(2.675*q_e*1000.0/tlcl_e)
        thetae_ee = thetae_e[start:]
        depth_index = np.argmax(thetae_ee[p_e[start:]/100.0 >= mu_depth])
        # If in an extrememly unlikely situation we end up with two heights
        # having the same theta-E, choose the higher one.
        #depth_index = max(depth_index)
        start = depth_index

    # Boolean vector holding starting parcel locations (1 = Valid)
    temp_z = np.zeros(len(z))
    temp_z[depth_index] = 1

    # Starting Parcel Attributes
    # Pot. temp. of parcel
    t_p = t_e*temp_z 
    t_p[t_p==0] = np.nan
    t_p = np.nanmean(t_p)

    # Water vapor mixing ratio of parcel
    q_p = q_e*temp_z
    q_p[q_p==0] = np.nan 
    q_p = np.nanmean(q_p)

    # Virt. pot. temp. of parcel
    tv_p = calc_tv(t_p,q_p)
    flag1 = 0

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parcel Ascent %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Preallocate Arrays 
    # (buoyancy, cape, cin)
    B    = np.zeros(len(z)-(start))
    cape = np.zeros(len(B))
    cin  = np.zeros(len(B))
    
    # Initialize values 
    # (lcl, lfc, parcel vertical velocity, liquid water, ice water)
    lcl = 9999.9
    lfc = 9999.9
    q_l = 0.0 
    q_i = 0.0
    
    # Logging: Track parcel ascent sensible temperature and vertical velocity
    temp_log = np.empty(len(z)-(start))*np.nan
    w_log    = np.empty(len(z)-(start))*np.nan
    
    # Starting parcel values
    temp_log[start] = t_p*pii[start]
    w_log[start] = 0 # assuming parcel is starting with no momentum
    
    # Raise parcel and determine buoyancy
    for k in range (start+1,len(z)-start):
        # Current sensible temperature of parcel
        ts_p = t_p*pii[k]
        
        # Current latent heat of vaporization (based in parcel temperature)
        # Bryan and Fritsch (2002)
        lv_p = lv0-(cl-cpv)*(ts_p-t0)
        ls_p = ls0-(ci-cpv)*(ts_p-t0)
    
        # Current saturation mixing ratio of liquid
        # Magnus equation for saturation vapor pressure
        es_p = calc_es(ts_p-t0) # vapor pres.
        rs_p = calc_r(es_p,p_e[k]) # mixing ratio
        qs_p = rs_p/(1+rs_p) # specific humidity
    
        # Ice saturation mixing ratio
        esi_p = 611.2*np.exp(21.8745584*(ts_p-t0)/(ts_p-7.66)) # vapor pres.
        rsi_p = eps*esi_p/(p_e[k]-esi_p) # mixing ratio
        qsi_p = rsi_p/(1+rsi_p) # specific humidity
    
        #% % % FOR LIQUID PROCESSES ONLY % % % 
        # Adjust by latent heat of vaporization by condensing in new mass when
        # parcel is supersaturated
        # Rate of conversion from Bryan and Fritsch (2002)
        if flag_heat == 1:
            # % IRREVERSIBLE ASCENT % %
            # No hydrometeors
            delqvl = 0
            if q_p > qs_p:
                lcl = min(lcl,z[k]/1e3) # LCL height [km]
                delqvl = (q_p-qs_p)/(1.0 + (qs_p*lv_p**2)/(cpd*Rv*ts_p**2))
                t_p = t_p+(delqvl*lv_p/cpd)/pii[k]
                q_p = q_p-delqvl
    
            #% % REVERSIBLE ASCENT % %
            #% Keep track of hydrometeors mass
            if flag_adiabat == 2:
                q_l = q_l + delqvl
                # Allow for evaporation of hydrometers if for some unlikely
                # reason we do end up subsaturated
                if q_p < qs_p: 
                    delqvl = min(q_l,abs(q_p-qs_p))/(1.0 + (qs_p*lv_p**2)/(cpd*Rv*ts_p**2))
                    t_p = t_p - (delqvl*lv_p/cpd)/pii[k]
                    q_l = q_l - delqvl
                    q_p = q_p + delqvl
    
        #% % % FOR LIQUID AND ICE PROCESSES % % %
        #% Where parcel is saturated: adjust by latent heat of condensation
        if flag_heat == 2:
            #% % IRREVERSIBLE ASCENT % %
            # No hydrometeors
            # The amount of vapor to convert to water or ice is based on a
            # linear function of temperature. All ice when T <= -40 deg C and
            # all water when T >= 0 deg C.
            ratio = max(min((ts_p-233.15)/(t0-233.15),1.0),0.0)
            delq  = min(q_p,ratio*qs_p+(1.0-ratio)*qsi_p)
            ice_gain = max((1.0-ratio)*(q_p-delq),0.0)
            liq_gain = max(q_p-delq-ice_gain,0.0)
            if (liq_gain > 0) and (lcl == 9999.9):
                lcl = min(lcl,(z[k]/1e3))
    
            delqvi = (ice_gain)/(1.0+(qsi_p*ls_p**2)/(cpd*Rv*ts_p**2))
            delqvl = (liq_gain)/(1.0+(qs_p*lv_p**2)/(cpd*Rv*ts_p**2))
            t_p = t_p + ((delqvi*ls_p+delqvl*lv_p)/cpd)/(pii[k])
            q_p = q_p - (delqvi+delqvl)
    
            # % REVERSIBLE ASCENT % %
            # Keep track of hydrometeors mass
            if flag_adiabat == 2:
                #% % % FREEZING OF PRE-EXISTING WATER % % %
                if flag_convert == 1:
                    # Convert a fraction of water to ice based on the same
                    # linear temperature function.
                    ratio = max(min((ts_p-233.15)/(t0-233.15),1.0),0.0)
                    liq_conv = min(q_l,(1.0-ratio)*q_l)
                    t_p = t_p + ((liq_conv*(ls_p-lv_p))/cpd)/(pii[k])
                else:
                    liq_conv = 0
            
                q_l = q_l+delqvl-liq_conv
                q_i = q_i+delqvi+liq_conv
    
                # Allow for evaporation of hydrometers if for some unlikely reason we do end up subsaturated.
                # Ratio of how much ice loss
                # should occur relative to liquid follows similar linear
                # temperature curve (but inverse, so more liquid loss at cold temps)
                ratio = max(min((ts_p-233.15)/(t0-233.15),1.0),0.0)
                delq = max(q_p,ratio*qs_p+(1.0-ratio)*qsi_p)
                ice_loss = min(q_i,abs((1.0-ratio)*(q_p-delq)))
                liq_loss = min(q_l,abs(q_p-delq+ice_loss))
                ice_loss = min(q_i,ice_loss+(delq-q_p-liq_loss-ice_loss))
                delqvi = (ice_loss)/(1.0 + (qsi_p*ls_p**2)/(cpd*Rv*ts_p**2))
                delqvl = (liq_loss)/(1.0 + (qs_p*lv_p**2)/(cpd*Rv*ts_p**2))
                t_p = t_p - ((delqvi*ls_p+delqvl*lv_p)/cpd)/(pii[k])
                q_p = q_p + (delqvi+delqvl)
                q_l = q_l - delqvl
                q_i = q_i - delqvi
    
        # Get new adjusted virt. pot. temp. of parcel
        tv_p = t_p*(1.0+0.61*q_p-(q_l+q_i))
        # Determine Buoyancy
        B[k] = g*((tv_p-tv_e[k])/tv_e[k])
        # Determine CAPE/CIN [J kg-1]
        cape[k] = B[k]*(z[k]-z[k-1])*(B[k] > 0)
        cin[k]  = B[k]*(z[k]-z[k-1])*(B[k] < 0)
    
        # Get new sensible temperature
        # Note: A sort of "equivalent" sensible temperature of parcel: the
        # temperature of the parcel if all effects (such as loading) were
        # accounted for and represented only as a change in temperature
        ts_p = (tv_p/(1.0+0.61*q_p))*pii[k]
    
        # Get "traditional" temperature-based LFC if LCL exists
        # This will pick up an embedded pockets of instability as the LFC if
        # there is such a pocket present.
        if (lcl<9999.9) and (ts_p >= ts_e[k]):
            lfc = min(lfc,z[k]/1.e3) # [km]
            # Alternative LFC: Mean between two points, 1st unstable and last
            # stable point. Probably more "accurate" representation of actual
            # LFC if the grid is coarse.
            # lfc = min(lfc,((z(k)+z(k-1))/2)/1e3);
    
        # Save logs of parcel sensible temperature and vertical velocity
        temp_log[k] = ts_p # [K]
        w_log[k]    = np.sqrt(2.*np.nansum(cape[start+1:k+1]))-np.sqrt(2*np.nansum(abs(cin[start+1:k+1]))) # [m/s]
    
    # Determine LNB [km]: Height where parcel no longer buoyant.
    # Look for: 1st negatively buoyany height with no positive buoyancy above
    lnb = np.nan # [km]
    for k in range(start+1,len(z)-start):
        #print (k,lnb,z[k-1]/1000.)
        if (B[k] < 0) & (B[k-1] > 0) & (np.nansum(B[k:] > 0) == 0):
            hgt = z[k-1]/1.e3
            lnb = np.nanmin([hgt,lnb])
            if lnb == 0:
                lnb = np.nan 
    
    # "Trim" CAPE/CIN to include everything below LNB only and get one value
    # cape = sum(cape.*(z/1e3 < lnb));
    CIN = np.nansum(cin[z[:len(z)-start]/1.e3 < lnb])
    
    # Alternative CAPE: Only positive buoyancy between LFC and LNB. This
    # ignores pockets of instability between parcel starting point and the LFC.
    # Would eliminate CAPE resulting from surface "superadiabatic" or dry
    # convective CAPE. 
    CAPE = np.nansum(cape[(z[:len(z)-start]/1.e3 < lnb) & (z[:len(z)-start]/1.e3 >= lfc)])
    # low-level cape
    LCAPE = np.nansum(cape[(z[:len(z)-start]/1.e3 < lfc+4) & (z[:len(z)-start]/1.e3 >= lfc)])
    
    #print ('lfc=', round(ZLFc,0))
    NCAPE = CAPE/(lnb*1.e3-lfc*1.e3)
    
    # Vertical Velocity of Parcel
    w_p = np.sqrt(2*np.nansum(CAPE))-np.sqrt(2*np.nansum(abs(CIN)))
    
    # wind shear
    wshr  = calc_shear(wspd, wdir, z)
    wsshr = 0.
    
    # midlevel humidity   
    if (rh_e[(z>3000) & (z<6000)].shape[0]==0):
        mid_rh = np.nan
    else:
        mid_rh = calc_midlevel_rh(rh_e,z)
    
    # lowlevel humidity
    low_rh = calc_lowlevel_rh(rh_e,z)
    
    # Richardson number
    richardson = calc_richardson(tv_e,u_wind,v_wind,z)
    
    # calculate the average environmental lapse rate 
    try:
        elr3, elr6 = calc_elr(ts_e,z)
    except: 
        elr3 = np.nan
        elr6 = np.nan
    
    results ={'CAPE':CAPE,'CIN':CIN,'LNB':lnb,'LFC':lfc,'LCL':lcl,'Wind_Shear':wshr,'Low_RH':low_rh,'Mid_RH':mid_rh,'Richardson':richardson,'ELR_0-3km':elr3,'NCAPE':NCAPE}
    
    return(results)