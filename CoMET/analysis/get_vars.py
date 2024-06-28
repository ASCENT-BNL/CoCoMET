#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:17:45 2024

@author: thahn
"""

# =============================================================================
# This defines the primary functionality of the analysis module for CoMET. Including the primary get_var function which works similar to wrf-python's getvar
# =============================================================================


"""
Inputs:
    analysis_object: The output of CoMET_start and/or an object generated by create_analysis_object following CoMET-UDAF Specification 1.x.
    variable: Which variable to calculate. See list below.
    *args: 
Outputs:
    Depends on requested variable.
Valid Variables:
    "eth"

"""
def get_var(analysis_object, variable, *args):
    from .calculate_bulk_cell_statistics import calculate_ETH
    
    print("=====In Progress=====")
    
    # Map the correct functions to the proper variables. This is a list of all the calculatable variables as well.
    variable_call_mechanism = {
        "eth": calculate_ETH,
    }
    
    # Check for valid variables
    if (variable.lower() not in variable_call_mechanism):
        raise Exception(f"!=====Invalid Variable Requested. You Entered: {variable.lower()}=====!")
        return
    
    
    # Call the proper function and return its output
    return (variable_call_mechanism[variable.lower()](analysis_object, *args))