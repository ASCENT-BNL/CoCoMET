#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:37:09 2024

@author: thahn
"""

# =============================================================================
# This file contains the functionality for detecting merging and splitting cells
# =============================================================================

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from dask.array import isin
from tqdm import tqdm


# Calculate nearest item in list to given pivot
def find_nearest(array, pivot):
    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx


def merge_split_tracking(
    analysis_object: dict,
    variable: str,
    invert: bool = False,
    cell_footprint_height: float = 2,
    touching_threshold: float = 0.20,
    flood_background: float = 20,
    score_threshold: float = 0,
    score_weight_1: float = 1,
    score_weight_2: float = 1,
    radius_multiplyer: float = 0.1,
    overlap_threshold: float = 0.5,
    steps_forward_back: int = 2,
    **args: dict,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """


    Parameters
    ----------
    analysis_object : dict
        A  CoMET-UDAF standard analysis object containing at least segmentation_xarray, UDAF_tracks. and UDAF_segmentation_2d or UDAF_segmentation_3d, and segmentation_xarray.
    variable : str
        The variable which defines the background field.
    invert : bool, optional
        If we care about tracking lower values such as brightness temperature (i.e. we want to flood fill stuff less than a threshold), we need to invert the data so we can use the same algorithm. Will also invert background value. The default is False.
    cell_footprint_height : float, optional
        The height at which we want to find the newly calculated cell areas in kilometers. The default is 2km.
    touching_threshold : float, optional
        The percentage of cell border which must be shared by two cells. The default is 0.20.
    flood_background : float, optional
        The minimum background value for area calculation. The default is 20.
    score_threshold : float, optional
        The minimum value of the score function to be considered a part of the cell. The default is 0.
    score_weight_1 : float, optional
        The weighting of the relative strength of the cell. The default is 1.
    score_weight_2 : float, optional
        The weighting of the relative distance of the cell. The default is 1.
    radius_multiplyer : float, optional
        The multiplyer to increase the search radius by. Is equal to 1 + radius_multiplyer. The default is 0.1.
    overlap_threshold : float, optional
        The amount two cells need to overlap [0 to 1]. The default is 0.5.
    steps_forward_back : int, optional
        The number of steps to look forward or back to detect mergers/splitters. The default is 2.
    **args : dict
        Throw away params.

    Raises
    ------
    Exception
        Exception if missing segmentation input.

    Returns
    -------
    merged_df : pandas.core.frame.DataFrame
        A pandas dataframe with these columns: frame, parent_cells, merged_cell
    split_df : pandas.core.frame.DataFrame
        A pandas dataframe with these columns: frame, split_cell, child_cells
    """

    # TODO: Rewrite this to be more efficent / better in general

    Tracks = analysis_object["UDAF_linking"]

    # If input variable field is 2D return None. Also, if DataArray, use those values for calculations. If Dataset, use tracking_var to get variable
    if type(analysis_object["segmentation_xarray"]) == xr.core.dataarray.DataArray:

        if len(analysis_object["segmentation_xarray"].shape) < 3:
            warnings.warn("!=====Input Variable has Too Low Dimensionality=====!")
            return None

        else:
            variable_field = analysis_object["segmentation_xarray"]

    else:

        if len(analysis_object["segmentation_xarray"][variable].shape) < 3:
            warnings.warn("!=====Input Variable has Too Low Dimensionality=====!")
            return None

        else:
            variable_field = analysis_object["segmentation_xarray"][variable]

    # If 3D segmentation is available, use that to calculate cell footprint, otherwise use 2D segmentation
    if analysis_object["UDAF_segmentation_3d"] is not None:

        height_index = find_nearest(
            analysis_object["UDAF_segmentation_3d"].altitude.values,
            cell_footprint_height * 1000,
        )

        footprint_data = analysis_object["UDAF_segmentation_3d"].Feature_Segmentation[
            :, height_index
        ]

    elif analysis_object["UDAF_segmentation_2d"] is not None:

        footprint_data = analysis_object["UDAF_segmentation_2d"].Feature_Segmentation

    else:
        raise Exception("!=====Missing Segmentation Input=====!")

    # If we care about tracking lower values such as brightness temperature (i.e. we want to flood fill stuff less than a threshold), we need to invert the data so we can use the same algorithm. Will also invert background value
    if invert:

        variable_field = -1 * variable_field
        flood_background = -1 * flood_background

    output_frame_list_merge = []
    output_init_cells_merge = []
    output_merged_cells = []

    output_frame_list_split = []
    output_init_cells_split = []
    output_split_cells = []

    frame_groups = Tracks.groupby("frame")

    for frame in tqdm(
        frame_groups,
        desc="=====Calculating Mergers and Splitters=====",
        total=frame_groups.ngroups,
    ):

        if frame[1].frame.min() >= (Tracks.frame.max() - (steps_forward_back - 1)):
            continue
        if frame[1].frame.min() <= (Tracks.frame.min() + (steps_forward_back - 1)):
            continue

        # Replace empty grid boxes with -1s to keep consistent with UDAF
        final_mask = deepcopy(footprint_data[frame[1].frame.min()])
        final_mask.values[:] = -1

        for feature in frame[1].iterrows():

            feature_mask = deepcopy(footprint_data[frame[1].frame.min()])
            feature_mask.values[~isin(feature_mask.values, feature[1].feature_id)] = -1

            # Replace with cell id instead of feature id and merge with final_mask
            final_mask.values[feature_mask.values != -1] = feature[1].cell_id

            # If cell has no area, skip
            if np.all(feature_mask.values == -1):
                continue

        # Now find touching cells
        valid_cells = np.unique(final_mask.values)[1:]
        cell_data = final_mask.values

        ci = []
        ne = []
        tu = []
        for cell_id in valid_cells:

            num_of_edges = 0
            touching = []

            # Get indices of valid cells in mask
            # Loop over indices
            for nx, ny in zip(*np.where(cell_data == cell_id)):

                # Find if location is on edge of cell (i.e. any touching cells not)
                neighboring_cells = []

                for mx in range(nx - 1, nx + 2):

                    for my in range(ny - 1, ny + 2):

                        try:
                            t = cell_data[mx, my]
                        except:
                            t = -1

                        neighboring_cells.append(t)

                # Get all 9 adjecent data cells to the current data cell of interest
                neighboring_cells = np.array(neighboring_cells).reshape((3, 3))

                # If touching any data cells not belonging to the cell of interest, consider that cell an edge
                if np.sum(neighboring_cells != cell_id) != 0:
                    num_of_edges += 1

                temp = neighboring_cells[neighboring_cells != -1]
                temp = temp[temp != cell_id]
                touching.append(np.unique(temp))

            touching = np.concatenate(touching)

            ci.append(cell_id)
            ne.append(num_of_edges)
            tu.append(touching)

        cell_info_df = pd.DataFrame(
            data={"cell": ci, "Num_Edges": ne, "Touching_Edges": tu}
        )

        valid_touching_cell_sets = []

        # As long as one of the cells exceeds out threshold of touching, it will get added to tracked list, so no need to do anything more complex
        for cell in cell_info_df.iterrows():

            uu = np.unique(cell[1].Touching_Edges, return_counts=True)

            for touching_cellid, touching_edge_count in zip(uu[0], uu[1]):

                # If touching by over a certain percent threshold, add to valid touching set
                if (touching_edge_count / cell[1].Num_Edges) > touching_threshold:

                    touching_tuple = tuple(np.sort((cell[1].cell, touching_cellid)))
                    valid_touching_cell_sets.append(touching_tuple)

        # Save only unique tuples so tracking doesn"t repeat
        valid_touching_cell_sets = list(dict.fromkeys(valid_touching_cell_sets))

        valid_overlap_cell_sets = []

        for cell_set in valid_touching_cell_sets:

            # Cell 1 and 2 checks
            cell1_data = frame[1].query("cell_id==@cell_set[0]")
            cell2_data = frame[1].query("cell_id==@cell_set[1]")

            # If 3d, choose height, otherwise, use whole thing
            if (
                len(variable_field.shape) == 4
                and analysis_object["UDAF_segmentation_3d"] is not None
            ):

                height_index = find_nearest(
                    analysis_object["UDAF_segmentation_3d"].altitude.values,
                    cell_footprint_height * 1000,
                )
                reflectivity_data = deepcopy(
                    variable_field[cell1_data.frame.min()]
                ).values[height_index]

            else:

                reflectivity_data = deepcopy(
                    variable_field[cell1_data.frame.min()].squeeze()
                ).values

            # flood fill cell 1
            row_1 = int(np.round(cell1_data.south_north.values[0]))
            col_1 = int(np.round(cell1_data.west_east.values[0]))

            # Calculate the cell adjusted variable field, radius and then get the max distance of the search radius
            adj_Rmax_1 = np.round(reflectivity_data[row_1, col_1] - flood_background)
            cell_1_radius = np.ceil(
                cell_info_df.query("cell==@cell_set[0]").Num_Edges.values[0]
                / (2 * np.pi)
            )
            cell_1_radius = int(np.ceil((1 + radius_multiplyer) * cell_1_radius))
            max_dist_1 = np.sqrt(2) * cell_1_radius
            segmented_1 = np.zeros(reflectivity_data.shape)

            # Loop over search radius
            for mx in range(row_1 - cell_1_radius, row_1 + cell_1_radius + 1):

                for my in range(col_1 - cell_1_radius, col_1 + cell_1_radius + 1):

                    try:

                        # Calculate score function
                        R_adj = reflectivity_data[mx, my] - flood_background
                        dis = np.sqrt((my - col_1) ** 2 + (mx - row_1) ** 2)

                        # If the adjusted variable field value is not finite, skip it
                        if not np.isfinite(R_adj):
                            continue

                        # If the cell is the center of the search radius, will be a part of the cell
                        if dis == 0:
                            segmented_1[mx, my] = 1
                            continue

                        # If the maximum variable value or maximum distance is 0, just skip since every data cell would be counted
                        if adj_Rmax_1 == 0 or max_dist_1 == 0:
                            continue

                        # Calculate the score and see if it exceeds the threshold
                        score = score_weight_1 * (
                            R_adj / adj_Rmax_1
                        ) - score_weight_2 * (dis / max_dist_1)

                        if score > score_threshold:
                            segmented_1[mx, my] = 1

                    except:
                        continue

            segmented_1_area = np.sum(segmented_1)

            # flood fill cell 2
            row_2 = int(np.round(cell2_data.south_north.values[0]))
            col_2 = int(np.round(cell2_data.west_east.values[0]))

            # Calculate adjusted max variable, radius values and then max distance
            adj_Rmax_2 = np.round(reflectivity_data[row_2, col_2] - flood_background)
            cell_2_radius = np.ceil(
                cell_info_df.query("cell==@cell_set[1]").Num_Edges.values[0]
                / (2 * np.pi)
            )
            cell_2_radius = int(np.ceil(1.1 * cell_2_radius))
            max_dist_2 = np.sqrt(2) * cell_2_radius
            segmented_2 = np.zeros(reflectivity_data.shape)

            for mx in range(row_2 - cell_2_radius, row_2 + cell_2_radius + 1):

                for my in range(col_2 - cell_2_radius, col_2 + cell_2_radius + 1):

                    try:

                        # Repeat same process but for the second cell in the pair
                        R_adj = reflectivity_data[mx, my] - flood_background
                        dis = np.sqrt((my - col_2) ** 2 + (mx - row_2) ** 2)

                        if not np.isfinite(R_adj):
                            continue

                        if dis == 0:
                            segmented_2[mx, my] = 1
                            continue

                        if adj_Rmax_2 == 0 or max_dist_2 == 0:
                            continue

                        score = (R_adj / adj_Rmax_2) - (dis / max_dist_2)

                        if score > score_threshold:
                            segmented_2[mx, my] = 1

                    except:
                        continue

            segmented_2_area = np.sum(segmented_2)

            overlap_mask = np.logical_and(segmented_1, segmented_2)
            overlap_mask_area = np.sum(overlap_mask)

            if segmented_1_area == 0 or segmented_2_area == 0:
                continue

            max_ol_p = np.max(
                (
                    overlap_mask_area / segmented_1_area,
                    overlap_mask_area / segmented_2_area,
                )
            )

            if max_ol_p > overlap_threshold:
                valid_overlap_cell_sets.append(cell_set)

        # Check if all valid overlap cells still exist in the next frame(s), if no, whichever still exists merged
        for cell_set in valid_overlap_cell_sets:
            cell_1_status = [False] * steps_forward_back
            cell_2_status = [False] * steps_forward_back

            for step in range(1, steps_forward_back + 1):
                nf_id = frame[1].frame.min() + step
                next_frame = Tracks.query("frame==@nf_id")

                cell_1_status[step - 1] = (
                    next_frame.query("cell_id==@cell_set[0]").shape[0] != 0
                )
                cell_2_status[step - 1] = (
                    next_frame.query("cell_id==@cell_set[1]").shape[0] != 0
                )

            if np.all(cell_1_status) and not np.any(cell_2_status):

                output_frame_list_merge.append(
                    np.array([frame[1].frame.min(), frame[1].frame.min() + 1])
                )
                output_init_cells_merge.append(np.array((cell_set[0], cell_set[1])))
                output_merged_cells.append(cell_set[0])

            elif not np.any(cell_1_status) and np.all(cell_2_status):

                output_frame_list_merge.append(
                    np.array([frame[1].frame.min(), frame[1].frame.min() + 1])
                )
                output_init_cells_merge.append(np.array((cell_set[0], cell_set[1])))
                output_merged_cells.append(cell_set[1])

        # Check if all valid overlap cells still exist in the previous frame(s), if no, whichever didn"t exist is the split
        for cell_set in valid_overlap_cell_sets:
            cell_1_status = [False] * steps_forward_back
            cell_2_status = [False] * steps_forward_back

            for step in range(1, steps_forward_back + 1):
                nf_id = frame[1].frame.min() - step
                next_frame = Tracks.query("frame==@nf_id")

                cell_1_status[step - 1] = (
                    next_frame.query("cell_id==@cell_set[0]").shape[0] != 0
                )
                cell_2_status[step - 1] = (
                    next_frame.query("cell_id==@cell_set[1]").shape[0] != 0
                )

            if np.all(cell_1_status) and not np.any(cell_2_status):

                output_frame_list_split.append(
                    np.array([frame[1].frame.min() - 1, frame[1].frame.min()])
                )
                output_init_cells_split.append(cell_set[0])
                output_split_cells.append((cell_set[0], cell_set[1]))
            elif not np.any(cell_1_status) and np.all(cell_2_status):

                output_frame_list_split.append(
                    np.array([frame[1].frame.min() - 1, frame[1].frame.min()])
                )
                output_init_cells_split.append(cell_set[1])
                output_split_cells.append((cell_set[0], cell_set[1]))

    merged_df = pd.DataFrame(
        data={
            "frame": output_frame_list_merge,
            "parent_cells": output_init_cells_merge,
            "merged_cell": output_merged_cells,
        }
    )
    split_df = pd.DataFrame(
        data={
            "frame": output_frame_list_split,
            "split_cell": output_init_cells_split,
            "child_cells": output_split_cells,
        }
    )

    return (merged_df, split_df)
