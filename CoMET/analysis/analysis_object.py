#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:03:37 2024

@author: thahn
"""

# =============================================================================
# This defines the analysis object class
# =============================================================================


class Analysis_Object:

    def __init__(
        self,
        tracking_xarray,
        segmentation_xarray,
        UDAF_features,
        UDAF_tracks,
        UDAF_segmentation_2d,
        UDAF_segmentation_3d,
    ):
        self.tracking_xarray = tracking_xarray
        self.segmentation_xarray = segmentation_xarray
        self.UDAF_features = UDAF_features
        self.UDAF_tracks = UDAF_tracks
        self.UDAF_segmentation_2d = UDAF_segmentation_2d
        self.UDAF_segmentation_3d = UDAF_segmentation_3d

    def return_analysis_dictionary(self) -> dict:

        return {
            "tracking_xarray": self.tracking_xarray,
            "segmentation_xarray": self.segmentation_xarray,
            "UDAF_features": self.UDAF_features,
            "UDAF_tracks": self.UDAF_tracks,
            "UDAF_segmentation_2d": self.UDAF_segmentation_2d,
            "UDAF_segmentation_3d": self.UDAF_segmentation_3d,
        }
