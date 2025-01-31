#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:12:33 2024

@author: thahn
"""

# =============================================================================
# TODO: Implement all these tests once we can create our test cases
# =============================================================================

import glob
import os
import shelve
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from six.moves import urllib

import CoCoMET


# Returns true if there is a difference between input objects
def diff_output(d1: object, d2: object, path: str = "") -> bool:
    """


    Parameters
    ----------
    d1 : object
        DESCRIPTION.
    d2 : object
        DESCRIPTION.
    path : str, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    bool
        DESCRIPTION.

    """
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                if diff_output(d1[k], d2[k], "%s -> %s" % (path, k) if path else k):
                    return True
            elif type(d1[k]) is tuple:
                for ii in range(len(d1[k])):
                    if (
                        type(d1[k][ii]) is pd.DataFrame
                        or type(d1[k][ii]) is gpd.GeoDataFrame
                    ):
                        if not d1[k][ii].equals(d2[k][ii]):
                            result = [
                                "%s: " % path,
                                " - %s : %s" % (k, d1[k]),
                                " + %s : %s" % (k, d2[k]),
                            ]
                            print("\n".join(result))
                    elif d1[k][ii] != d2[k][ii]:
                        result = [
                            "%s: " % path,
                            " - %s : %s" % (k, d1[k]),
                            " + %s : %s" % (k, d2[k]),
                        ]
                        print("\n".join(result))
                        return True
            elif (
                type(d1[k]) is pd.DataFrame
                or type(d1[k]) is gpd.GeoDataFrame
                or type(d1[k]) is xr.Dataset
            ):
                if not d1[k].equals(d2[k]):
                    result = [
                        "%s: " % path,
                        " - %s : %s" % (k, d1[k]),
                        " + %s : %s" % (k, d2[k]),
                    ]
                    print("\n".join(result))
                    return True
            elif type(d1[k]) is xr.DataArray:
                if not d1[k].broadcast_equals(d2[k]):
                    result = [
                        "%s: " % path,
                        " - %s : %s" % (k, d1[k]),
                        " + %s : %s" % (k, d2[k]),
                    ]
                    print("\n".join(result))
                    return True
            elif type(d1[k]) is CoCoMET.analysis.analysis_object.Analysis_Object:
                if diff_output(
                    d1[k].return_analysis_dictionary(),
                    d2[k].return_analysis_dictionary(),
                    "%s -> %s" % (path, k) if path else k,
                ):
                    return True
            elif d1[k] != d2[k]:
                result = [
                    "%s: " % path,
                    " - %s : %s" % (k, d1[k]),
                    " + %s : %s" % (k, d2[k]),
                ]
                print("\n".join(result))
                return True
        else:
            print("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))

    return False


data_out = Path(os.getcwd())
CONFIG = CoCoMET.CoCoMET_load(str(data_out) + "/CoCoMET/tests/config_for_tests.yml")


def test_load_data():
    # We need to store the data online somewhere
    # Then add a script to download data from online...probably zenodo

    data_out = Path(os.getcwd())
    data_files = glob.glob(str(data_out) + "/.cocomet_testing_datasets/WRF/*")

    if len(data_files) == 0:
        file_path = "https://zenodo.org/records/14611486/files/.cocomet_testing_datasets.zip?download=1"

        temp_zip_file = Path("temp.zip")
        print("=====Downloading and Extracing Data=====")

        urllib.request.urlretrieve(file_path, temp_zip_file)

        shutil.unpack_archive(temp_zip_file, data_out)
        temp_zip_file.unlink()

        data_files = glob.glob(str(data_out) + "/.cococomet_testing_datasets/WRF/*")

    assert True


# """
# TODO: Add full testing coverage
# """


def test_single_core_full_run():
    CONFIG["parallel_processing"] = False

    output = CoCoMET.CoCoMET_start(CONFIG=CONFIG)

    # Get expected output
    with shelve.open(str(data_out) + "/.cocomet_testing_datasets/true_out") as f:
        print(f)
        true_output = f["data"]

    assert diff_output(output, true_output) == False


def test_multi_core_full_run():
    CONFIG["parallel_processing"] = True

    output = CoCoMET.CoCoMET_start(CONFIG=CONFIG)

    # Get expected output
    with shelve.open(str(data_out) + "/.cocomet_testing_datasets/true_out") as f:
        true_output = f["data"]

    assert diff_output(output, true_output) == False
