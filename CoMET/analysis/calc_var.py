#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:17:45 2024

@author: thahn
"""

# =============================================================================
# This defines the primary functionality of the analysis module for CoMET. Including the primary get_var function which works similar to wrf-python's getvar
# =============================================================================

from CoMET.analysis.analysis_object import Analysis_Object

from .calculate_arm_products import calculate_convective_indices, extract_arm_product
from .calculate_bulk_cell_statistics import (
        calculate_ETH,
        calculate_area,
        calculate_volume,
        calculate_max_height,
        calculate_max_intensity,
        calculate_velocity,
        calculate_perimeter,
        calculate_cell_growth,
        calculate_irregularity,
    )
from .merge_split_detection import merge_split


def calc_var(analysis_object: Analysis_Object, var: str, **args: dict):
    """


    Parameters
    ----------
    analysis_object : Analysis_Object
        The output of CoMET_start and/or an object generated by create_analysis_object following CoMET-UDAF Specification 1.x.
    var : str
        Which variable to calculate. See list below.
    **args : dict
        The additional arguments of indeterminate length required for calculating request variables.

    Raises
    ------
    Exception
        Exception if invalid variable name.

    Returns
    -------
    Type depends on variable
        N/A.

    Valid Variables
    ---------------
        "eth",
        "max_height",
        "max_intensity",
        "area",
        "volume",
        "velocity",
        "perimeter",
        "cell_growth",
        "merge_split",
        "arm",

    """

    # Map the correct functions to the proper variables. This is a list of all the calculatable variables as well.
    variable_call_mechanism = {
        "eth": calculate_ETH,
        "max_height" : calculate_max_height,
        "max_intensity" : calculate_max_intensity,
        "area": calculate_area,
        "volume": calculate_volume,
        "velocity": calculate_velocity,
        "perimeter": calculate_perimeter,
        "cell_growth": calculate_cell_growth,
        "irregularity": calculate_irregularity,
        "merge_split": merge_split,
        "arm": extract_arm_product,
        "convective_indices": calculate_convective_indices,
    }

    # Check for valid variables
    if var.lower() not in variable_call_mechanism:
        raise Exception(
            f"!=====Invalid Variable Requested. You Entered: {var.lower()}=====!"
        )

    # Call the proper function and return its output
    return variable_call_mechanism[var.lower()](
        analysis_object=analysis_object.return_analysis_dictionary(),
        **args,
    )
