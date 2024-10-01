#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:17:45 2024

@author: thahn
"""

# =============================================================================
# This defines the primary functionality of the analysis module for CoMET. Including the primary get_var function which works similar to wrf-python's getvar
# =============================================================================


def calc_var(analysis_object, var, **args):
    """
    Inputs:
        analysis_object: The output of CoMET_start and/or an object generated by create_analysis_object following CoMET-UDAF Specification 1.x.
        var: Which variable to calculate. See list below.
        verbose: Determins if output should be printed during the calculation of selected variable
        **args: The additional arguments of indeterminate length required for calculating request variables.
    Outputs:
        Depends on requested variable.
    Valid Variables:
        "eth",
        "area",
        "volume",
        "merge_split",
        "arm",
        "interpsonde"
    """

    from .merge_split_detection import merge_split_tracking
    from .calculate_arm_products import (
        extract_arm_product,
        calculate_arm_interpsonde,
    )
    from .calculate_bulk_cell_statistics import (
        calculate_ETH,
        calculate_area,
        calculate_volume,
        # calculate_max_height
    )

    # Map the correct functions to the proper variables. This is a list of all the calculatable variables as well.
    variable_call_mechanism = {
        "eth": calculate_ETH,
        # "max_height" : calculate_max_height,
        "area": calculate_area,
        "volume": calculate_volume,
        "merge_split": merge_split_tracking,
        "arm": extract_arm_product,
        "interpsonde": calculate_arm_interpsonde,
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
