#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Created on Tue Jul  9 13:02:43 2024

@author: thahn

ADAPTED FROM wrfcube package: 

BSD 3-Clause License

Copyright (c) 2019, Max Heikenfeld
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import vincenty
from iris import Constraint, coord_systems, coords
from iris.coords import AuxCoord
from iris.util import promote_aux_coord_to_dim_coord
from numpy import arange, array, transpose


def load(mesonh_xarray, variable, filename):

    array = mesonh_xarray[variable]
    variable_dimensions = array.dims
    attributes = mesonh_xarray.attrs
    cube = array.to_iris()
    coord_system = make_coord_system(attributes)

    dy, dx = guess_horizontal_spacing(mesonh_xarray, filename)

    for dim in range(len(variable_dimensions)):
        if variable_dimensions[dim] == "x":
            west_east = make_westeast_coord(dx, mesonh_xarray.x.size)
            cube.add_dim_coord(west_east, dim)
            projection_x_coord = make_x_coord(
                dx, mesonh_xarray.x.size, coord_system=coord_system
            )
            cube.add_aux_coord(projection_x_coord, dim)
            x_coord = AuxCoord(cube.coord("west_east").points, long_name="x", units=1)
            cube.add_aux_coord(x_coord, data_dims=cube.coord_dims("west_east"))
        elif variable_dimensions[dim] == "y":
            south_north = make_southnorth_coord(dy, mesonh_xarray.y.size)
            cube.add_dim_coord(south_north, dim)
            projection_y_coord = make_y_coord(
                dy, mesonh_xarray.y.size, coord_system=coord_system
            )
            cube.add_aux_coord(projection_y_coord, dim)
            y_coord = AuxCoord(cube.coord("south_north").points, long_name="y", units=1)
            cube.add_aux_coord(y_coord, data_dims=cube.coord_dims("south_north"))
        elif variable_dimensions[dim] == "z":
            bottom_top = make_bottom_top_coordinate(mesonh_xarray.z.size)
            cube.add_dim_coord(bottom_top, dim)
            model_level_number = make_model_level_number_coordinate(
                mesonh_xarray.z.size
            )
            cube.add_aux_coord(model_level_number, dim)

    if "time" in [coord.name() for coord in cube.coords()]:
        promote_aux_coord_to_dim_coord(cube, "time")
        cube.coord("time").attributes = {}

    # change latitude and longitude coordinates to  2D fields (fine andm ore consistent with other models for all static WRF Simulations)
    if "lat" in [coord.name() for coord in cube.coords()]:
        latitude_coord = cube.extract(
            Constraint(
                time=cube.coord("time").units.num2date(cube.coord("time").points[0])
            )
        ).coord("lat")
        latitude_coord.rename("latitude")
        xlat_dims = list(cube.coord_dims("lat"))
        time_dim = cube.coord_dims("time")[0]
        xlat_dims.remove(time_dim)
        data_dims = tuple(xlat_dims)
        cube.add_aux_coord(latitude_coord, data_dims=data_dims)
        cube.remove_coord("lat")
    if "lon" in [coord.name() for coord in cube.coords()]:
        longitude_coord = cube[0].coord("lon")
        longitude_coord.rename("longitude")
        xlong_dims = list(cube.coord_dims("lon"))
        time_dim = cube.coord_dims("time")[0]
        xlong_dims.remove(time_dim)
        data_dims = tuple(xlong_dims)
        cube.add_aux_coord(longitude_coord, data_dims=data_dims)
        cube.remove_coord("lon")

    return cube


def guess_horizontal_spacing(mesonh_xarray, filename):
    try:

        # Try to guess dimension from file name
        dis_value = filename.split("m")[0]

        # If is in kilometers, convert to meters
        if "k" in dis_value:

            dis_value = float(dis_value.replace("k", "")) * 1000

        else:

            dis_value = float(dis_value)

        return (dis_value, dis_value)

    except ValueError:
        print(
            "!=====Non-Default MesoNH Filename Found, Estimating Distance Instead=====!"
        )

        # Guess x dimension by finding distance between two points offset by one x value
        x_dis = (
            vincenty.vincenty(
                (mesonh_xarray.lat[0, 0, 0].values, mesonh_xarray.lon[0, 0, 0].values),
                (mesonh_xarray.lat[0, 0, 1].values, mesonh_xarray.lon[0, 0, 1].values),
            )
            * 1000
        )
        # Guess y dimension by finding distance between two points offset by one y value
        y_dis = (
            vincenty.vincenty(
                (mesonh_xarray.lat[0, 0, 0].values, mesonh_xarray.lon[0, 0, 0].values),
                (mesonh_xarray.lat[0, 1, 0].values, mesonh_xarray.lon[0, 1, 0].values),
            )
            * 1000
        )

        # Round to nearest tens place
        x_dis = 10 * round(x_dis / 10)
        y_dis = 10 * round(y_dis / 10)

        return (y_dis, x_dis)


def make_coord_system(attributes):
    return None

    #    :CEN_LAT = -3.212929f ;
    # 		:CEN_LON = -60.59799f ;
    # 		:TRUELAT1 = 0.f ;
    # 		:TRUELAT2 = -5.f ;
    # 		:MOAD_CEN_LAT = -3.212929f ;
    # 		:STAND_LON = -60.f ;
    # 		:POLE_LAT = 90.f ;
    # 		:POLE_LON = 0.f ;
    # 		:GMT = 0.f ;
    # 		:JULYR = 2014 ;
    # 		:JULDAY = 244 ;
    # 		:MAP_PROJ = 1 ;
    # 		:MAP_PROJ_CHAR = "Lambert Conformal" ;
    MAP_PROJ_CHAR = attributes["MAP_PROJ_CHAR"]
    MAP_PROJ = attributes["MAP_PROJ"]

    # cartesian coordinate system (idealized simulations):
    if MAP_PROJ_CHAR == "Cartesian" and MAP_PROJ == 0:
        coord_system = None

    # lambert Conformal system (idealized simulations):
    if MAP_PROJ_CHAR == "Lambert Conformal" and MAP_PROJ == 1:
        CEN_LON = attributes["CEN_LON"]
        TRUELAT1 = attributes["TRUELAT1"]
        TRUELAT2 = attributes["TRUELAT2"]
        MOAD_CEN_LAT = attributes["MOAD_CEN_LAT"]
        # STAND_LON = attributes["STAND_LON"]
        # POLE_LAT = attributes["POLE_LAT"]
        # POLE_LON = attributes["POLE_LON"]
        coord_system = coord_systems.LambertConformal(
            central_lat=MOAD_CEN_LAT,
            central_lon=CEN_LON,
            false_easting=0.0,
            false_northing=0.0,
            secant_latitudes=(TRUELAT1, TRUELAT2),
        )
    return coord_system


def make_westeast_coord(DX, WEST_EAST_PATCH_END_UNSTAG):

    WEST_EAST = arange(0, WEST_EAST_PATCH_END_UNSTAG)
    west_east = coords.DimCoord(
        WEST_EAST,
        standard_name=None,
        long_name="west_east",
        var_name="west_east",
        units="1",
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
    )
    return west_east


def make_westeast_stag_coord(DX, WEST_EAST_PATCH_END_STAG):

    WEST_EAST_U = arange(0, WEST_EAST_PATCH_END_STAG)
    west_east_stag = coords.DimCoord(
        WEST_EAST_U,
        standard_name=None,
        long_name="west_east_stag",
        var_name="west_east_stag",
        units="1",
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
    )
    return west_east_stag


def make_southnorth_coord(DY, SOUTH_NORTH_PATCH_END_UNSTAG):

    # SOUTH_NORTH_PATCH_END_UNSTAG=attributes["SOUTH-NORTH_PATCH_END_UNSTAG"]
    SOUTH_NORTH = arange(0, SOUTH_NORTH_PATCH_END_UNSTAG)
    south_north = coords.DimCoord(
        SOUTH_NORTH,
        standard_name=None,
        long_name="south_north",
        var_name="south_north",
        units="1",
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
    )
    return south_north


def make_southnorth_stag_coord(DY, SOUTH_NORTH_PATCH_END_STAG):

    SOUTH_NORTH_V = arange(0, SOUTH_NORTH_PATCH_END_STAG)
    south_north_stag = coords.DimCoord(
        SOUTH_NORTH_V,
        standard_name=None,
        long_name="south_north_stag",
        var_name="south_north_stag",
        units="1",
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
    )
    return south_north_stag


def make_bottom_top_coordinate(BOTTOM_TOP_PATCH_END_UNSTAG):

    BOTTOM_TOP = arange(0, BOTTOM_TOP_PATCH_END_UNSTAG)
    bottom_top = coords.DimCoord(
        BOTTOM_TOP,
        standard_name=None,
        long_name="altitude",
        var_name="altitude",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
    )
    return bottom_top


def make_bottom_top_stag_coordinate(BOTTOM_TOP_PATCH_END_STAG):

    BOTTOM_TOP_W = arange(0, BOTTOM_TOP_PATCH_END_STAG)
    bottom_top_stag = coords.DimCoord(
        BOTTOM_TOP_W,
        standard_name=None,
        long_name="altitude_stag",
        var_name="altitude_stag",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
    )
    return bottom_top_stag


def make_model_level_number_coordinate(BOTTOM_TOP_PATCH_END):

    MODEL_LEVEL_NUMBER = arange(0, BOTTOM_TOP_PATCH_END)
    model_level_number = coords.AuxCoord(
        MODEL_LEVEL_NUMBER, standard_name="model_level_number", units="1"
    )
    return model_level_number


def make_x_coord(DX, WEST_EAST_PATCH_END_UNSTAG, coord_system):

    X = DX * (arange(0, WEST_EAST_PATCH_END_UNSTAG) + 0.5)
    bounds = transpose(
        array(
            [
                DX * (arange(0, WEST_EAST_PATCH_END_UNSTAG)),
                DX * (arange(0, WEST_EAST_PATCH_END_UNSTAG) + 1),
            ]
        )
    )
    x_coord = coords.AuxCoord(
        X,
        standard_name="projection_x_coordinate",
        long_name="x",
        var_name="x",
        units="m",
        bounds=bounds,
        attributes=None,
        coord_system=coord_system,
    )
    # x_coord.add_dim_coord(west_east,0)
    return x_coord


def make_x_stag_coord(DX, WEST_EAST_PATCH_END_STAG, coord_system=None):

    X_U = DX * (arange(0, WEST_EAST_PATCH_END_STAG) - 1)
    x_stag_coord = coords.AuxCoord(
        X_U,
        standard_name="projection_x_coordinate",
        long_name="x",
        var_name="x",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=coord_system,
    )
    # x_stag_coord.add_dim_coord(west_east_stag,0)
    return x_stag_coord


def make_y_coord(DY, SOUTH_NORTH_PATCH_END_UNSTAG, coord_system=None):

    Y = DY * (arange(0, SOUTH_NORTH_PATCH_END_UNSTAG) + 0.5)
    bounds = transpose(
        array(
            [
                DY * (arange(0, SOUTH_NORTH_PATCH_END_UNSTAG)),
                DY * (arange(0, SOUTH_NORTH_PATCH_END_UNSTAG) + 1),
            ]
        )
    )
    y_coord = coords.AuxCoord(
        Y,
        standard_name="projection_y_coordinate",
        long_name="y",
        var_name="y",
        units="m",
        bounds=bounds,
        attributes=None,
        coord_system=coord_system,
    )
    # y_coord.add_dim_coord(south_north,0)
    return y_coord


def make_y_stag_coord(DY, SOUTH_NORTH_PATCH_END_STAG, coord_system=None):

    Y_V = DY * (arange(0, SOUTH_NORTH_PATCH_END_STAG) - 1)
    y_stag_coord = coords.AuxCoord(
        Y_V,
        standard_name="projection_y_coordinate",
        long_name="y",
        var_name="y",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=coord_system,
    )
    # y_stag_coord.add_dim_coord(south_north_stag,0)
    return y_stag_coord
