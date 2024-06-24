#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:34:29 2024

@author: thahn
"""

# =============================================================================
# Creates a dictionary which the CoMET analysis module can understand
# =============================================================================
# TODO: Update this to convert values first to CoMET-UDAF before returning analysis object


"""
Inputs:
    Various outputs from different CoMET functions
Outpus:
    analysis_object_dict: A dictionary setup for intake into the CoMET analysis module if you wanted to run the CoMET functions independently of CoMET_start
"""
def create_analysis_object(
        wrf_tracking_xarray = None, # Mandatory for WRF
        wrf_tracking_cube = None, # Mandatory
        wrf_segmentation_xarray = None, #Optional
        wrf_segmentation_cube = None, # Optional
        wrf_tobac_features = None, # Optional
        wrf_tobac_tracks = None, # Optional
        wrf_tobac_segmentation_2d = None, # Optional
        wrf_tobac_segmentation_3d = None, # Optional
        
        nexrad_tracking_xarray = None, # Mandatory for NEXRAD
        nexrad_tracking_cube = None, # Mandatory
        nexrad_segmentation_xarray = None, # Optional
        nexrad_segmentation_cube = None, # Optional
        nexrad_tobac_features = None, # Optional
        nexrad_tobac_tracks = None, # Optional
        nexrad_tobac_segmentation_2d = None, # Optional
        nexrad_tobac_segmentation_3d = None, # Optional
        
        goes_tracking_xarray = None, # Mandatory for GOES
        goes_tracking_cube = None, # Mandatory
        goes_segmentation_xarray = None, # Optional
        goes_segmentation_cube = None, # Optional
        goes_tobac_features = None, # Optional
        goes_tobac_tracks = None, # Optional
        goes_tobac_segmentation_2d = None, # Optional
        ):
    
    # Create return dictionary
    analysis_object_dict = {}
    
    # Check to see if wrf_tracking_xarray and wrf_tracking_cube exist
    if (wrf_tracking_xarray is not None and wrf_tracking_cube is not None):
        
        analysis_object_dict["wrf"] = {
            "tracking_xarray": wrf_tracking_xarray,
            "tracking_cube": wrf_tracking_cube,
            "segmentation_xarray": wrf_segmentation_xarray,
            "segmentation_cube": wrf_segmentation_cube    
        }
        
        # Check if any tobac fields are not None
        if (wrf_tobac_features is not None or
            wrf_tobac_tracks is not None or
            wrf_tobac_segmentation_2d is not None or
            wrf_tobac_segmentation_3d is not None):
            
            analysis_object_dict["wrf"]["tobac"] = {
                "feature_id": wrf_tobac_features,
                "linking": wrf_tobac_tracks,
                "segmentation_2d": wrf_tobac_segmentation_2d,
                "segmentation_3d": wrf_tobac_segmentation_3d
            }
    
    
    # Check to see if nexrad_tracking_xarray and nexrad_tracking_cube exist
    if (nexrad_tracking_xarray is not None and nexrad_tracking_cube is not None):
        
        analysis_object_dict["nexrad"] = {
            "tracking_xarray": nexrad_tracking_xarray,
            "tracking_cube": nexrad_tracking_cube,
            "segmentation_xarray": nexrad_segmentation_xarray,
            "segmentation_cube": nexrad_segmentation_cube    
        }
        
        # Check if any tobac fields are not None
        if (nexrad_tobac_features is not None or
            nexrad_tobac_tracks is not None or
            nexrad_tobac_segmentation_2d is not None or
            nexrad_tobac_segmentation_3d is not None):
            
            analysis_object_dict["nexrad"]["tobac"] = {
                "feature_id": nexrad_tobac_features,
                "linking": nexrad_tobac_tracks,
                "segmentation_2d": nexrad_tobac_segmentation_2d,
                "segmentation_3d": nexrad_tobac_segmentation_3d
            }
    
    
    # Check to see if goes_tracking_xarray and goes_tracking_cube exist
    if (goes_tracking_xarray is not None and goes_tracking_cube is not None):
        
        analysis_object_dict["goes"] = {
            "tracking_xarray": goes_tracking_xarray,
            "tracking_cube": goes_tracking_cube,
            "segmentation_xarray": goes_segmentation_xarray,
            "segmentation_cube": goes_segmentation_cube
        }
        
        # Check if any tobac fields are not None
        if (goes_tobac_features is not None or
            goes_tobac_tracks is not None or
            goes_tobac_segmentation_2d is not None):
            
            analysis_object_dict["goes"]["tobac"] = {
                "feature_id": goes_tobac_features,
                "linking": goes_tobac_tracks,
                "segmentation_2d": goes_tobac_segmentation_2d,
            }
            
    
    # Return analysis object
    return (analysis_object_dict)