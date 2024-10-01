#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:00:20 2024

@author: thahn
"""

# =============================================================================
# Functions to filter output from CoMET into something clearner and easier to use
# =============================================================================


def filter_cells(track_output, domain_shapely=None):

    # Remove cells which exist at frame 0 and cells which exist at the very last frame
    # Remove cells which get too close to the boundary edge
    # Remove cells which travel outside of the domain defined by an input shapely file

    print("=====In Progress=====")
