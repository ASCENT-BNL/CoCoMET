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
        US_features,
        US_tracks,
        US_segmentation_2d,
        US_segmentation_3d,
    ):
        self.tracking_xarray = tracking_xarray
        self.segmentation_xarray = segmentation_xarray
        self.US_features = US_features
        self.US_tracks = US_tracks
        self.US_segmentation_2d = US_segmentation_2d
        self.US_segmentation_3d = US_segmentation_3d

    def return_analysis_dictionary(self) -> dict:

        return {
            "tracking_xarray": self.tracking_xarray,
            "segmentation_xarray": self.segmentation_xarray,
            "US_features": self.US_features,
            "US_tracks": self.US_tracks,
            "US_segmentation_2d": self.US_segmentation_2d,
            "US_segmentation_3d": self.US_segmentation_3d,
        }
