#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:00:20 2024

@author: thahn
"""

# =============================================================================
# Functions to filter output from CoMET into something clearner and easier to use
# =============================================================================

import numpy as np
from tqdm import tqdm
from .analysis import Analysis_Object

def filter_cells(
        analysis_object: dict | Analysis_Object,
        **args: dict,
) -> Analysis_Object :
    """


    Parameters
    ----------
    analysis_object : dict
        A  CoMET-UDAF standard analysis object containing at least UDAF_features, UDAF_tracks, UDAF_segmentation_2d, tracking_xarray, and segmentation_xarray.
    **args : dict
        Throw away inputs
            
    Raises
    ------
    Exceptioon
        Exception if missing segmentation data from the analysis object.
    
    Returns
    -------
    analysis_object_filtered : Analysis_Object
        A  CoMET-UDAF standard analysis object containing UDAF_features, UDAF_tracks, UDAF_segmentation_2d, tracking_xarray, and segmentation_xarray.
    """

    # Remove cells which exist at frame 0 and cells which exist at the very last frame
    # Remove cells which get too close to the boundary edge
    # Remove cells which travel outside of the domain defined by an input shapely file

    if type(analysis_object) != dict:
        analysis_object = analysis_object.return_analysis_dictionary()

    frame_groups = analysis_object["UDAF_linking"].groupby("frame")

    if analysis_object['UDAF_segmentation_3d'] is not None:
    
        cell_seg = analysis_object['UDAF_segmentation_3d'].Cell_Segmentation
        dim = '3d'
    
    elif ['UDAF_segmentation_2d'] is not None:

        cell_seg = analysis_object['UDAF_segmentation_2d'].Cell_Segmentation
        dim = '2d'
    
    else:
        raise Exception("!=====Missing Segmentation Input=====!")


    edge_cells_to_delete = []

    # Loop over frames
    for frameg in frame_groups:

        frame = frameg[0]

        # Remove all cells in the first or last frames
        if frame == 0 or (frame == frame_groups.ngroups - 1):
        #     # add all cells in these frames to the delete list
            for cellg in frameg[1].groupby('cell_id'):
                edge_cells_to_delete.append(cellg[0])

        # Go through the sides of the 2D or 3D space and remove 
        else:      

            # go through and check the top, bottom, left, right, forward, back faces
            
            bottom_features = np.unique(cell_seg[frame, 0, :, :])[1:] # each will have -1 for a lack of a feature
            top_features = np.unique(cell_seg[frame, -1, :, :])[1:]
            forward_features = np.unique(cell_seg[frame, :, 0, :])[1:]
            backward_features = np.unique(cell_seg[frame, :, -1, :])[1:]

            edge_features = np.unique(np.concatenate((bottom_features, top_features,
                                            forward_features, backward_features)))
            if dim == '3d':
                left_features = np.unique(cell_seg[frame, :, :, 0])[1:]
                right_features = np.unique(cell_seg[frame, :, :, -1])[1:]

                edge_features = np.unique(np.concatenate((edge_features,
                                                left_features, right_features)))
            
            edge_cells_to_delete = edge_cells_to_delete + edge_features.tolist()
    
    # Make sure the each edge is only listed once
    edge_cells_to_delete = np.unique(edge_cells_to_delete)

    # Dataframes to remove features from
    UDAF_features = analysis_object['UDAF_features']
    UDAF_linking = analysis_object['UDAF_linking']

    # Xarrays to remove features from converted to numpy arrays
        # create numpy arrays so hopefully computation will be faster, then revert back to DataArrays when everything is finished

    UDAF_segmentation_2d = analysis_object['UDAF_segmentation_2d']
    UDAF_segmentation_2d_cell_numpy = UDAF_segmentation_2d.Cell_Segmentation.values
    UDAF_segmentation_2d_feature_numpy = UDAF_segmentation_2d.Feature_Segmentation.values    
    if dim == '3d':
        UDAF_segmentation_3d = analysis_object['UDAF_segmentation_3d']
        UDAF_segmentation_3d_cell_numpy = UDAF_segmentation_3d.Cell_Segmentation.values
        UDAF_segmentation_3d_feature_numpy = UDAF_segmentation_3d.Feature_Segmentation.values
    else:
        UDAF_segmentation_3d = None

    # get dimensions for re-conversion of numpy arrays to DataArrays
    dims2d = UDAF_segmentation_2d.dims
    if dim == '3d':
        dims3d = UDAF_segmentation_3d.dims

    for cell in tqdm(edge_cells_to_delete,
        desc = '=====Deleting Edge Cells=====',
        total = len(edge_cells_to_delete)):
        
        cell_indices_2d = where_replacement(UDAF_segmentation_2d_cell_numpy == cell)
        features = -np.ones_like(UDAF_segmentation_2d_cell_numpy)

        for t, y, x in cell_indices_2d:
            features[t, y, x] = UDAF_segmentation_2d_feature_numpy[t, y, x]

            UDAF_segmentation_2d_cell_numpy[t, y, x] = -1
            UDAF_segmentation_2d_feature_numpy[t, y, x] = -1
        
        if dim == '3d':
            cell_indices_3d = where_replacement(UDAF_segmentation_3d_cell_numpy == cell)
            features = -np.ones_like(UDAF_segmentation_3d_feature_numpy)

            for t, z, y, x in cell_indices_3d:
                UDAF_segmentation_3d_cell_numpy[t, z, y, x] =  -1
                UDAF_segmentation_3d_feature_numpy[t, z, y, x] =  -1

                features[t, z, y, x] = UDAF_segmentation_3d_feature_numpy[t, z, y, x]

        features = np.unique(features)[1:]
        UDAF_features.drop(features, inplace = True)
        UDAF_linking.drop(features, inplace = True)


   # convert back to data arrays
    UDAF_segmentation_2d['Cell_Segmentation'] = ([*dims2d], UDAF_segmentation_2d_cell_numpy)
    UDAF_segmentation_2d['Feature_Segmentation'] = ([*dims2d], UDAF_segmentation_2d_feature_numpy)
    if dim == '3d':
        UDAF_segmentation_3d['Cell_Segmentation'] = ([*dims3d], UDAF_segmentation_3d_cell_numpy)
        UDAF_segmentation_3d['Feature_Segmentation'] = ([*dims3d], UDAF_segmentation_3d_feature_numpy)

    # Recreate the analysis object
    analysis_object_filtered = Analysis_Object(
    analysis_object['tracking_xarray'],
    analysis_object['segmentation_xarray'],
    UDAF_features,
    UDAF_linking,
    UDAF_segmentation_2d,
    UDAF_segmentation_3d,
    )

    return analysis_object_filtered

def where_replacement(bool_arr):
    """
    A replacement function for numpy.where in attempts to speed up the process
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    def compute_M(data):
        cols = np.arange(data.size)
        return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))

    def get_indices_sparse(data):
        M = compute_M(data)
        return [np.unravel_index(row.data, data.shape) for row in M]
    
    found_inds = get_indices_sparse(bool_arr)
    if len(found_inds) == 2:
        indices_to_replace = np.asarray(found_inds[1]).T
        return indices_to_replace
    elif len(found_inds) == 1:
        return []
    else:
        raise Exception("Maybe you dont actually know how this works")
