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
        UDAF_linking,
        UDAF_segmentation_2d,
        UDAF_segmentation_3d,

        **kwargs
    ):
        self.tracking_xarray = tracking_xarray
        self.segmentation_xarray = segmentation_xarray
        self.UDAF_features = UDAF_features
        self.UDAF_linking = UDAF_linking
        self.UDAF_segmentation_2d = UDAF_segmentation_2d
        self.UDAF_segmentation_3d = UDAF_segmentation_3d

        if 'rams_segmentation_3d' in kwargs:
            self.rams_segmentation_3d = kwargs['rams_segmentation_3d']

    def return_analysis_dictionary(self):

        return {
            "tracking_xarray": self.tracking_xarray,
            "segmentation_xarray": self.segmentation_xarray,
            "UDAF_features": self.UDAF_features,
            "UDAF_linking": self.UDAF_linking,
            "UDAF_segmentation_2d": self.UDAF_segmentation_2d,
            "UDAF_segmentation_3d": self.UDAF_segmentation_3d,

            "second_3d_segmentation_variable" : self.rams_segmentation_3d
        }
