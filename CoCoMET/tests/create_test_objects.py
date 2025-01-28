#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:48:43 2024

@author: thahn
"""

# =============================================================================
# TODO: Create a basic case study of one isolated cell and one MCS to input in the tracks
# Should also convert this one case into the various model types / CoCoMET input types to test.
# This can be used later for unit testing of individual functions
# =============================================================================


def create_isolated_cell(grid_shape, cell_var, cell_radius, max_dbz=None, min_tb=None):
    """

    Parameters
    ----------
    grid_shape : length 4 tuple
        The shape of the domain you want to create in number of grid points (t,x,y,z). i.e. (20,500,500,50).
    cell_var : string, ["dbz", "tb"]
        The variable you want to use to define the cell, either reflectivity or brightness temperature.
    cell_radius : int
        The radius of the cell in # of grid points.
    max_dbz : float, optional
        The maximum desired reflectivity value if using dbz as cell_var. The default is None.
    min_tb : float, optional
        The minimum desired brightness temperature value if using tb as cell_var. The default is None.

    Returns
    -------
    cell_grid : numpy.ndarray
        A numpy array of shape equivelant to grid_shape but has either reflectivity or brightness temperature values.

    """

    import numpy as np
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    cell_grid = np.zeros(grid_shape)

    # Cell should start in corner at mid height and propagate to the center of the grid
    cell_location = [0, 0, int(np.median(np.arange(0, grid_shape[3])))]

    # Determine the number of grid points that need to be moved per frame
    target_point = [int(grid_shape[1] / 2), int(grid_shape[2] / 2)]

    x_steps_per_time = np.ceil(target_point[0] / grid_shape[0])
    y_steps_per_time = np.ceil(target_point[1] / grid_shape[0])

    for tt in tqdm(
        range(grid_shape[0]),
        desc="=====Creating Isolated Cell Grid=====",
        total=grid_shape[0],
    ):

        # Create sphere of radius r
        xx, yy, zz = np.mgrid[: grid_shape[1], : grid_shape[2], : grid_shape[3]]

        sphere = (
            (xx - cell_location[0]) ** 2
            + (yy - cell_location[1]) ** 2
            + (zz - cell_location[2]) ** 2
        )
        sphere = sphere <= (cell_radius**2)

        cell_grid[tt] = sphere

        # Adjust cell position
        cell_location[0] += x_steps_per_time
        cell_location[1] += y_steps_per_time

    # If cell_var is "dbz" set reflectivities
    if cell_var.lower() == "dbz":

        cell_grid[cell_grid == 0] = -30
        cell_grid[cell_grid == 1] = max_dbz

        # Add some filter to make it more cell-like
        for tt in tqdm(
            range(cell_grid.shape[0]),
            desc="=====Filtering Isolated Cell=====",
            total=cell_grid.shape[0],
        ):
            cell_grid[tt] = gaussian_filter(cell_grid[tt], 7)

        return cell_grid

    # Otherwise set brightness temperatures
    cell_grid[cell_grid == 0] = 300
    cell_grid[cell_grid == 1] = min_tb

    # Add some filter to make it more cell-like
    for tt in tqdm(
        range(cell_grid.shape[0]),
        desc="=====Filtering Isolated Cell=====",
        total=cell_grid.shape[0],
    ):
        cell_grid[tt] = gaussian_filter(cell_grid[tt], 5)

    return cell_grid


def create_mcs():

    print("=====In Progress=====")


def create_test_wrf_xarray(cell_grid, dt, dx, dy, dz, cell_var):
    """

    Parameters
    ----------
    cell_grid : numpy.ndarray
        A numpy array of shape (t,x,y,z) but has either reflectivity or brightness temperature values.
    dt : float
        The temporal spacing of the grid in minutes.
    dx : float
        The x spacing of the grid in meters.
    dy : float
        The y spacing of the grid in meters.
    dz : float
        The z spacing of the grid in meters.
    cell_var : string ["dbz", "tb"]
        The variable you want to use to define the cell, either reflectivity or brightness temperature.

    Returns
    -------
    wrf_xarray : xarray.core.dataset.Dataset
        DESCRIPTION.

    """

    import xarray as xr

    print("=====In Progress=====")

    return xr.Dataset()
