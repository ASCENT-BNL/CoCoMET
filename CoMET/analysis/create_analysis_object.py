#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:34:29 2024

@author: thahn
"""

# =============================================================================
# Creates a dictionary which the CoMET analysis module can understand
# =============================================================================


"""
Inputs:
    Various outputs from different CoMET functions
Outpus:
    analysis_object_dict: A dictionary setup for intake into the CoMET analysis module if you wanted to run the CoMET functions independently of CoMET_start. Follows CoMET-UDAF Section 3.1.
"""
def create_analysis_object(
        wrf_tracking_xarray = None, # Mandatory for WRF
        wrf_segmentation_xarray = None, #Optional
        wrf_tobac_features = None, # Optional
        wrf_tobac_tracks = None, # Optional
        wrf_tobac_segmentation_2d = None, # Optional
        wrf_tobac_segmentation_3d = None, # Optional
        
        nexrad_tracking_xarray = None, # Mandatory for NEXRAD
        nexrad_segmentation_xarray = None, # Optional
        nexrad_tobac_features = None, # Optional
        nexrad_tobac_tracks = None, # Optional
        nexrad_tobac_segmentation_2d = None, # Optional
        nexrad_tobac_segmentation_3d = None, # Optional
        
        goes_tracking_xarray = None, # Mandatory for GOES
        goes_segmentation_xarray = None, # Optional
        goes_tobac_features = None, # Optional
        goes_tobac_tracks = None, # Optional
        goes_tobac_segmentation_2d = None, # Optional
        ):
    
    from .tracker_output_translation_layer import feature_id_to_UDAF, linking_to_UDAF, segmentation_to_UDAF
    
    # Create return dictionary
    analysis_object_dict = {}
    
    # Check to see if wrf_tracking_xarray and wrf_tracking_cube exist
    if (wrf_tracking_xarray is not None):
        
        analysis_object_dict["wrf"] = {}
        
        # Check if any tobac fields are not None
        if (wrf_tobac_features is not None or
            wrf_tobac_tracks is not None or
            wrf_tobac_segmentation_2d is not None or
            wrf_tobac_segmentation_3d is not None):
            
            analysis_object_dict["wrf"]["tobac"] = {
                "tracking_xarray": wrf_tracking_xarray,
                "segmentation_xarray": wrf_segmentation_xarray,
                "UDAF_features": feature_id_to_UDAF(wrf_tobac_features, "tobac"),
                "UDAF_linking": linking_to_UDAF(wrf_tobac_tracks, "tobac"),
                "UDAF_segmentation_2d": segmentation_to_UDAF(wrf_tobac_segmentation_2d[0], linking_to_UDAF(wrf_tobac_tracks, "tobac"), "tobac"),
                "UDAF_segmentation_3d": segmentation_to_UDAF(wrf_tobac_segmentation_3d[0], linking_to_UDAF(wrf_tobac_tracks, "tobac"), "tobac")
            }
            
    
    
    # Check to see if nexrad_tracking_xarray and nexrad_tracking_cube exist
    if (nexrad_tracking_xarray is not None):
        
        analysis_object_dict["nexrad"] = {}
        
        # Check if any tobac fields are not None
        if (nexrad_tobac_features is not None or
            nexrad_tobac_tracks is not None or
            nexrad_tobac_segmentation_2d is not None or
            nexrad_tobac_segmentation_3d is not None):
            
            analysis_object_dict["nexrad"]["tobac"] = {
                "tracking_xarray": nexrad_tracking_xarray,
                "segmentation_xarray": nexrad_tracking_xarray,
                "UDAF_features": feature_id_to_UDAF(nexrad_tobac_features, "tobac"),
                "UDAF_linking": linking_to_UDAF(nexrad_tobac_tracks, "tobac"),
                "UDAF_segmentation_2d": segmentation_to_UDAF(nexrad_tobac_segmentation_2d[0], linking_to_UDAF(nexrad_tobac_tracks, "tobac"), "tobac"),
                "UDAF_segmentation_3d": segmentation_to_UDAF(nexrad_tobac_segmentation_3d[0], linking_to_UDAF(nexrad_tobac_tracks, "tobac"), "tobac")
            }
    
    
    # Check to see if goes_tracking_xarray and goes_tracking_cube exist
    if (goes_tracking_xarray is not None):
        
        analysis_object_dict["goes"] = {}
        
        # Check if any tobac fields are not None
        if (goes_tobac_features is not None or
            goes_tobac_tracks is not None or
            goes_tobac_segmentation_2d is not None):
            
            analysis_object_dict["goes"]["tobac"] = {
                "tracking_xarray": goes_tracking_xarray,
                "segmentation_xarray": goes_tracking_xarray,
                "UDAF_features": feature_id_to_UDAF(goes_tobac_features, "tobac"),
                "UDAF_linking": linking_to_UDAF(goes_tobac_tracks, "tobac"),
                "UDAF_segmentation_2d": segmentation_to_UDAF(goes_tobac_segmentation_2d[0], linking_to_UDAF(goes_tobac_tracks, "tobac"), "tobac"),
                "UDAF_segmentation_3d": None
            }
            
    
    # Return analysis object
    return (analysis_object_dict)