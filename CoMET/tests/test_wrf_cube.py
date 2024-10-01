#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:12:33 2024

@author: thahn
"""

# =============================================================================
# TODO: Implement all these tests once we can create our test cases
# =============================================================================

from CoMET.create_test_objects import create_isolated_cell, create_test_wrf_xarray


# This test should always pass for now
def test_wrf_cube_load():

    import xarray as xr

    cell = create_isolated_cell(
        grid_shape=(10, 200, 200, 50), cell_var="tb", cell_radius=10, max_dbz=62
    )
    wrf_xarray = create_test_wrf_xarray(
        cell_grid=cell, dt=5, dx=1000, dy=1000, dz=500, cell_var="dbz"
    )

    assert type(wrf_xarray) == type(xr.Dataset())
