#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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


import warnings

warnings.filterwarnings("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=RuntimeWarning, append=True)
warnings.filterwarnings("ignore", category=FutureWarning, append=True)
from cf_units import Unit
from iris import Constraint
from iris.coords import AuxCoord
from iris.cube import CubeList
from iris.util import promote_aux_coord_to_dim_coord


def load(dataset, variable, mode="auto", **kwargs):
    return loadramscube_mult(dataset, variable)


def loadramscubelist(filenames, variable_list, **kwargs):

    cubelist_out = CubeList()
    for variable in variable_list:
        cubelist_out.append(loadramscube(filenames, variable, **kwargs))
    return cubelist_out


def loadramscube(dataset, variable, **kwargs):
    variable_cube = loadramscube_mult(dataset, variable)

    return variable_cube


def loadramscube_mult(dataset, variable, constraint=None, add_coordinates=None):

    array = dataset[variable]
    variable_dimensions = array.dims
    attributes = dataset.attrs
    cube = array.to_iris()
    coord_system = make_coord_system(attributes)

    for dim, dimension in enumerate(variable_dimensions):
        if variable_dimensions[dim] == "west_east":
            west_east = make_westeast_coord(
                attributes["DX"], attributes["WEST-EAST_PATCH_END_UNSTAG"]
            )
            cube.add_dim_coord(west_east, dim)
            projection_x_coord = make_x_coord(
                attributes["DX"],
                attributes["WEST-EAST_PATCH_END_UNSTAG"],
                coord_system=coord_system,
            )
            cube.add_aux_coord(projection_x_coord, dim)
            x_coord = AuxCoord(cube.coord("west_east").points, long_name="x", units=1)
            cube.add_aux_coord(x_coord, data_dims=cube.coord_dims("west_east"))
        elif variable_dimensions[dim] == "south_north":
            south_north = make_southnorth_coord(
                attributes["DY"], attributes["SOUTH-NORTH_PATCH_END_UNSTAG"]
            )
            cube.add_dim_coord(south_north, dim)
            projection_y_coord = make_y_coord(
                attributes["DY"],
                attributes["SOUTH-NORTH_PATCH_END_UNSTAG"],
                coord_system=coord_system,
            )
            cube.add_aux_coord(projection_y_coord, dim)
            y_coord = AuxCoord(cube.coord("south_north").points, long_name="y", units=1)
            cube.add_aux_coord(y_coord, data_dims=cube.coord_dims("south_north"))

        elif variable_dimensions[dim] == "west_east_stag":
            west_east_stag = make_westeast_stag_coord(
                attributes["DX"], attributes["WEST-EAST_PATCH_END_STAG"]
            )
            cube.add_dim_coord(west_east_stag, dim)
            projection_x_stag_coord = make_x_stag_coord(
                attributes["DX"],
                attributes["WEST-EAST_PATCH_END_STAG"],
                coord_system=coord_system,
            )
            cube.add_aux_coord(projection_x_stag_coord, dim)
            x_coord = AuxCoord(
                cube.coord("west_east_stag").points, long_name="x", units=1
            )
            cube.add_aux_coord(x_coord, data_dims=cube.coord_dims("west_east_stag"))

        elif variable_dimensions[dim] == "south_north_stag":
            south_north_stag = make_southnorth_stag_coord(
                attributes["DY"], attributes["SOUTH-NORTH_PATCH_END_STAG"]
            )
            cube.add_dim_coord(south_north_stag, dim)
            projection_y_stag_coord = make_y_stag_coord(
                attributes["DY"],
                attributes["SOUTH-NORTH_PATCH_END_STAG"],
                coord_system=coord_system,
            )
            cube.add_aux_coord(projection_y_stag_coord, dim)
            y_coord = AuxCoord(
                cube.coord("south_north_stag").points, long_name="y", units=1
            )
            cube.add_aux_coord(y_coord, data_dims=cube.coord_dims("south_north_stag"))

        elif variable_dimensions[dim] == "bottom_top":
            bottom_top = make_bottom_top_coordinate(
                attributes["BOTTOM-TOP_PATCH_END_UNSTAG"]
            )
            cube.add_dim_coord(bottom_top, dim)
            model_level_number = make_model_level_number_coordinate(
                attributes["BOTTOM-TOP_PATCH_END_UNSTAG"]
            )
            cube.add_aux_coord(model_level_number, dim)

        elif variable_dimensions[dim] == "bottom_top_stag":
            bottom_top_stag = make_bottom_top_stag_coordinate(
                attributes["BOTTOM-TOP_PATCH_END_STAG"]
            )
            cube.add_dim_coord(bottom_top_stag, dim)
            model_level_number = make_model_level_number_coordinate(
                attributes["BOTTOM-TOP_PATCH_END_STAG"]
            )
            cube.add_aux_coord(model_level_number, dim)

    if "Times" in [coord.name() for coord in cube.coords()]:
        cube.coord("Times").rename("time")
        promote_aux_coord_to_dim_coord(cube, "time")
        cube.coord("time").attributes = {}
        # only used to ensure that the time dimension has units
        cube.coord("time").units = Unit(dataset.date)

    # change latitude and longitude coordinates to 2D fields (fine andm ore consistent with other models for all static rams Simulations)
    if "GLAT" in [coord.name() for coord in cube.coords()]:
        latitude_coord = cube.extract(
            Constraint(
                time=cube.coord("time").units.num2date(cube.coord("time").points[0])
            )
        ).coord("GLAT")
        latitude_coord.rename("latitude")
        xlat_dims = list(cube.coord_dims("GLAT"))
        time_dim = cube.coord_dims("time")[0]
        xlat_dims.remove(time_dim)
        data_dims = tuple(xlat_dims)
        cube.add_aux_coord(latitude_coord, data_dims=data_dims)
        cube.remove_coord("GLAT")
    if "GLON" in [coord.name() for coord in cube.coords()]:
        longitude_coord = cube[0].coord("GLON")
        longitude_coord.rename("longitude")
        xlong_dims = list(cube.coord_dims("GLON"))
        time_dim = cube.coord_dims("time")[0]
        xlong_dims.remove(time_dim)
        data_dims = tuple(xlong_dims)
        cube.add_aux_coord(longitude_coord, data_dims=data_dims)
        cube.remove_coord("GLON")

    # bring time axis into range that can be understood by pandas (e.g. idealised simulation starting in year 0)
    # reference set to 2000-01-01_00:00:00 instead
    # date_0=cube.coord('time').units.num2date(cube.coord('time').points[0])
    # if (date_0 < datetime(1678,1,1) or date_0> datetime(2262,12,31)):
    #    cube.coord('time').units=Unit(cube.coord('time').units.name.split(' since ')[0]+' since 2000-01-01')

    cube = cube.extract(constraint)

    if add_coordinates != None:
        cube = add_aux_coordinates_multidim(
            dataset, cube, constraint=None, add_coordinates=add_coordinates
        )
    return cube


def deriveramscubelist(filenames, variable_list, **kwargs):

    cubelist_out = CubeList()
    for variable in variable_list:
        cubelist_out.append(deriveramscube(filenames, variable))
    return cubelist_out


#
# def deriveramscube(filenames,variable,**kwargs):
#    if type(filenames) is list:
#        variable_cube=deriveramscube_mult(filenames,variable,**kwargs)
#    elif type(filenames) is str:
#        variable_cube=deriveramscube_single(filenames,variable,**kwargs)
#    else:
#        raise ValueError('Type of input unknown: Must be str of list')
#    return variable_cube
#
# def deriveramscube_mult(filenames,variable,**kwargs):
#
#    cube_list=[]
#    for i in range(len(filenames)):
#        cube_list.append(deriveramscube_single(filenames[i],variable,**kwargs) )
#    for member in cube_list:
#        member.attributes={}
#    variable_cubes=CubeList(cube_list)
#    variable_cube=variable_cubes.concatenate_cube()
#    #variable_cube=variable_cube.extract(**kwargs.pop('constraint'))
#
#    return variable_cube

variable_dict_pseudonym = {}
variable_dict_pseudonym["radar_relfectivity"] = "REFL10CM"


import warnings

warnings.filterwarnings("ignore", category=UserWarning, append=True)


variable_list_derive = [
    "potential_temperature",
    "temperature",
    "air_temperature",
    "density",
    "air_density",
    "LWC",
    "IWC",
    "LWP",
    "IWP",
    "IWV",
    "airmass",
    "airmass_path",
    "layer_height",
    "volume",
    "area",
    "geopotential_height",
    "pressure",
    "air_pressure",
    "relative_humidity",
    "w_unstaggered",
    "u_unstaggered",
    "v_unstaggered",
    "maximum_reflectivity",
    "surface_precipitation",
    "surface_precipitation_average",
    "surface_precipitation_accumulated",
    "surface_precipitation_instantaneous",
    "QNCLOUD",
    "QNRAIN",
    "QNGRAUPEL",
    "QNICE",
    "QNSNOW",
]


# def deriveramscube_single(filenames,variable,**kwargs):
def deriveramscube(filenames, variable, **kwargs):
    if variable == "potential_temperature":
        variable_cube = calculate_rams_potential_temperature(filenames, **kwargs)
        # variable_cube_out=addcoordinates(filenames, 'T',variable_cube,add_coordinates)
    elif variable in ["temperature", "air_temperature"]:
        variable_cube = calculate_rams_temperature(filenames, **kwargs)
        # variable_cube_out=addcoordinates(filenames, 'T',variable_cube,add_coordinates)
    elif variable in ["density", "air_density"]:
        variable_cube = calculate_rams_density(filenames, **kwargs)
        # variable_cube_out=addcoordinates(filenames, 'T',variable_cube,add_coordinates)
    elif variable in ["pressure", "air_pressure"]:
        variable_cube = calculate_rams_pressure(filenames, **kwargs)
    elif variable == "LWC":
        variable_cube = calculate_rams_LWC(filenames, **kwargs)
        # variable_cube=addcoordinates(filenames, 'QCLOUD',variable_cube,add_coordinates)
    elif variable == "IWC":
        variable_cube = calculate_rams_IWC(filenames, **kwargs)
        # variable_cube=addcoordinates(filenames, 'QICE',variable_cube,add_coordinates)
    elif variable == "LWP":
        variable_cube = calculate_rams_LWP(filenames, **kwargs)
        # variable_cube=addcoordinates(filenames, 'OLR',variable_cube,add_coordinates)
    elif variable == "IWP":
        variable_cube = calculate_rams_IWP(filenames, **kwargs)
        # variable_cube=addcoordinates(filenames, 'OLR',variable_cube,add_coordinates)
    elif variable == "IWV":
        variable_cube = calculate_rams_IWV(filenames, **kwargs)
        # variable_cube=addcoordinates(filenames, 'OLR',variable_cube,add_coordinates)
    elif variable == "airmass":
        variable_cube = calculate_rams_airmass(filenames, **kwargs)
    elif variable == "airmass_path":
        variable_cube = calculate_rams_airmass_path(filenames, **kwargs)
    elif variable == "layer_height":
        variable_cube = calculate_rams_layerheight(filenames, **kwargs)
    elif variable == "area":
        variable_cube = calculate_rams_area(filenames, **kwargs)
    elif variable == "volume":
        variable_cube = calculate_rams_volume(filenames, **kwargs)
    elif variable == "geopotential_height":
        variable_cube = calculate_rams_geopotential_height(filenames, **kwargs)
        replace_cube = loadramscube(filenames, "T", **kwargs)
        variable_cube = replacecoordinates(variable_cube, replace_cube)

    elif variable == "geopotential_height_stag":
        variable_cube = calculate_rams_geopotential_height_stag(filenames, **kwargs)

    #    elif variable == 'geopotential_height_xstag':
    #        variable_cube=calculate_rams_geopotential_height_xstag(filenames,**kwargs)
    #        replace_cube=loadramscube(filenames,'U',**kwargs)
    #        variable_cube=replacecoordinates(variable_cube,replace_cube)
    #
    #    elif variable == 'geopotential_height_ystag':
    #        variable_cube=calculate_rams_geopotential_height_ystag(filenames,**kwargs)
    #        replace_cube=loadramscube(filenames,'V',**kwargs)
    #        variable_cube=replacecoordinates(variable_cube,replace_cube)

    elif variable == "geopotential":
        variable_cube = calculate_rams_geopotential(filenames, **kwargs)

    elif variable == "pressure_xstag":
        variable_cube = calculate_rams_pressure(filenames, **kwargs)
        replace_cube = loadramscube(filenames, "U", **kwargs)
        variable_cube = replacecoordinates(variable_cube, replace_cube)

    elif variable == "pressure_ystag":
        variable_cube = calculate_rams_pressure(filenames, **kwargs)
        replace_cube = loadramscube(filenames, "V", **kwargs)
        variable_cube = replacecoordinates(variable_cube, replace_cube)

    elif variable == "relative_humidity":
        variable_cube = calculate_rams_relativehumidity(filenames, **kwargs)
        # variable_cube_out=addcoordinates(filenames, 'T',variable_cube,add_coordinates)

    elif variable == "w_unstaggered":
        variable_cube = calculate_rams_w_unstaggered(filenames, **kwargs)
        replace_cube = loadramscube(filenames, "T", **kwargs)
        variable_cube = replacecoordinates(variable_cube, replace_cube)

    elif variable == "u_unstaggered":
        constraint = kwargs.pop("constraint", None)
        variable_cube = calculate_rams_u_unstaggered(filenames, **kwargs)
        replace_cube = loadramscube(filenames, "T", **kwargs)
        variable_cube = replacecoordinates(variable_cube, replace_cube)
        variable_cube = variable_cube.extract(constraint)

    elif variable == "v_unstaggered":
        constraint = kwargs.pop("constraint", None)
        variable_cube = calculate_rams_v_unstaggered(filenames, **kwargs)
        replace_cube = loadramscube(filenames, "T", **kwargs)
        variable_cube = replacecoordinates(variable_cube, replace_cube)
        variable_cube = variable_cube.extract(constraint)

    elif variable == "surface_precipitation_average":
        variable_cube = calculate_rams_surface_precipitation_average(
            filenames, **kwargs
        )

    elif variable == "surface_precipitation_accumulated":
        variable_cube = calculate_rams_surface_precipitation_accumulated(
            filenames, **kwargs
        )

    elif (variable == "surface_precipitation_instantaneous") or (
        variable == "surface_precipitation"
    ):
        variable_cube = calculate_rams_surface_precipitation_instantaneous(
            filenames, **kwargs
        )

        # variable_cube_out=addcoordinates(filenames, 'T',variable_cube,add_coordinates)
    elif variable == "maximum_reflectivity":
        variable_cube = calculate_rams_maximum_reflectivity(filenames, **kwargs)
    elif variable in ["QNCLOUD", "QNRAIN", "QNGRAUPEL", "QNICE", "QNSNOW"]:
        variable_cube = loadramscube(filenames, variable, **kwargs)
        variable_cube.units = "kg-1"
    else:
        raise NameError(variable, "is not a known variable")
    return variable_cube


def calculate_rams_surface_precipitation_average(filenames, **kwargs):
    from dask.array import concatenate

    surface_precip_accum = calculate_rams_surface_precipitation_accumulated(
        filenames, **kwargs
    )
    # caclulate timestep in hours
    time_coord = surface_precip_accum.coord("time")
    dt = (
        time_coord.units.num2date(time_coord.points[1])
        - time_coord.units.num2date(time_coord.points[0])
    ).total_seconds() / 3600.0
    # divide difference in precip between timesteps (in mm/h) by timestep (in h):
    surface_precip = surface_precip_accum
    surface_precip.data = (
        concatenate(
            (
                0 * surface_precip.core_data()[[0], :, :],
                surface_precip.core_data()[1:, :, :]
                - surface_precip.core_data()[:-1:, :, :],
            ),
            axis=0,
        )
        / dt
    )
    surface_precip.rename("surface_precipitation_average")
    surface_precip.units = "mm h-1"
    return surface_precip


def calculate_rams_surface_precipitation_accumulated(filenames, **kwargs):
    surface_precip_accum = loadramscube(filenames, "RAINNC", **kwargs)
    surface_precip_accum.rename("surface_precipitation_accumulated")
    return surface_precip_accum


def calculate_rams_surface_precipitation_instantaneous(filenames, **kwargs):
    from xarray import open_mfdataset

    dataset = open_mfdataset(filenames, coords="all")
    dt = dataset.attrs["DT"]
    RAINNCV = loadramscube(filenames, "RAINNCV", **kwargs)
    surface_precip = RAINNCV / dt
    surface_precip.units = "kg m-2 s-1"
    surface_precip.rename("surface_precipitation_instantaneous")
    return surface_precip


def variable_list(filenames):
    from netCDF4 import Dataset

    if type(filenames) == list:
        filenames = filenames[0]
    variable_list = list(Dataset(filenames).variables)
    return variable_list


def calculate_rams_potential_temperature(filenames, **kwargs):
    from iris import coords

    T = loadramscube(filenames, "T", **kwargs)
    T0 = coords.AuxCoord(300.0, long_name="reference_temperature", units="K")
    theta = T + T0
    theta.rename("potential_temperature")
    return theta


def calculate_rams_temperature(filenames, **kwargs):
    from iris import coords

    theta = deriveramscube(filenames, "potential_temperature", **kwargs)
    p = deriveramscube(filenames, "pressure", **kwargs)
    p0 = coords.AuxCoord(1000.0, long_name="reference_pressure", units="hPa")
    p0.convert_units(p.units)
    p1 = p / p0
    exp = 287.05 / 1005.0
    T = theta * (p1**exp)
    T.rename("air temperature")
    return T


def calculate_rams_relativehumidity(filenames, **kwargs):
    from iris import cube

    QVAPOR = loadramscube(filenames, "QVAPOR", **kwargs)
    T = deriveramscube(filenames, "temperature", **kwargs)
    p = deriveramscube(filenames, "pressure", **kwargs)
    p.convert_units("Pa")
    rh = calculate_RH(QVAPOR.core_data(), T.core_data(), p.core_data())
    RH = cube.Cube(rh, units="percent", long_name="realtive humidity")
    return RH


def calculate_RH(QVAPOR, T, p):
    from numpy import exp, maximum, minimum

    ES = 1e2 * 6.1094 * exp(17.625 * (T - 273.15) / (T - 273.15 + 243.04))
    QVS = 0.622 * ES / (p - (1.0 - 0.622) * ES)
    RH = 100 * maximum(minimum(QVAPOR / QVS, 1.0), 0.0)
    return RH


def calculate_rams_LWC(filenames, **kwargs):
    microphysics_scheme = kwargs.pop("microphysics_scheme", None)
    list_variables = ["QCLOUD", "QRAIN"]
    LWC = load_sum(filenames, list_variables, **kwargs)
    LWC.rename("liquid water content")
    # LWC.rename('mass_concentration_of_liquid_water_in_air')
    return LWC


#
def calculate_rams_IWC(filenames, **kwargs):
    microphysics_scheme = kwargs.pop("microphysics_scheme", None)
    if microphysics_scheme in [None, "morrison", "thompson"]:
        list_variables = ["QICE", "QSNOW", "QGRAUP"]
    elif microphysics_scheme in ["SBM_fast"]:
        list_variables = ["QICE", "QSNOW", "QGRAUP"]
    elif microphysics_scheme in ["SBM_full"]:
        list_variables = ["QICEC", "QICED", "QICEP", "QSNOW", "QGRAUP", "QHAIL"]
    IWC = load_sum(filenames, list_variables, **kwargs)
    IWC.rename("ice water content")
    # IWC.rename('mass_concentration_of_ice_water_in_air')

    return IWC


def calculate_rams_airmass_path(filenames, **kwargs):
    rho = deriveramscube(filenames, "density", **kwargs)
    layer_height = deriveramscube(filenames, "layer_height", **kwargs)
    Airmass = rho * layer_height
    Airmass.rename("airmass_path")
    return Airmass


def calculate_rams_airmass(filenames, **kwargs):
    rho = deriveramscube(filenames, "density", **kwargs)
    volume = deriveramscube(filenames, "volume", **kwargs)
    Airmass = rho * volume
    Airmass.rename("mass_of_air")
    return Airmass


def calculate_rams_volume(filenames, **kwargs):

    layer_height = deriveramscube(filenames, "layer_height", **kwargs)
    layer_height.add_aux_coord(
        AuxCoord(
            layer_height.coord("projection_x_coordinate").bounds[..., 1]
            - layer_height.coord("projection_x_coordinate").bounds[..., 0],
            long_name="x_diff",
            units=layer_height.coord("projection_y_coordinate").units,
        ),
        data_dims=layer_height.coord_dims("projection_x_coordinate"),
    )
    layer_height.add_aux_coord(
        AuxCoord(
            layer_height.coord("projection_y_coordinate").bounds[..., 1]
            - layer_height.coord("projection_y_coordinate").bounds[..., 0],
            long_name="y_diff",
            units=layer_height.coord("projection_y_coordinate").units,
        ),
        data_dims=layer_height.coord_dims("projection_y_coordinate"),
    )
    volume = layer_height * layer_height.coord("x_diff") * layer_height.coord("y_diff")
    volume.remove_coord("x_diff")
    volume.remove_coord("y_diff")
    volume.rename("cell_volume")
    return volume


def calculate_rams_area(filenames, **kwargs):
    from numpy import diff

    dummy = loadramscube(filenames, "OLR", **kwargs)
    dummy.data[:] = 1
    dummy.units = "1"
    dummy.add_aux_coord(
        AuxCoord(
            diff(dummy.coord("projection_x_coordinate").bounds).flatten(),
            long_name="x_diff",
            units=dummy.coord("projection_y_coordinate").units,
        ),
        data_dims=dummy.coord_dims("projection_x_coordinate"),
    )
    dummy.add_aux_coord(
        AuxCoord(
            diff(dummy.coord("projection_y_coordinate").bounds).flatten(),
            long_name="y_diff",
            units=dummy.coord("projection_y_coordinate").units,
        ),
        data_dims=dummy.coord_dims("projection_y_coordinate"),
    )
    area = dummy * dummy.coord("x_diff") * dummy.coord("y_diff")
    area.rename("cell_area")
    return area


def calculate_rams_layerheight(filenames, **kwargs):

    zH = deriveramscube(filenames, "geopotential_height_stag", **kwargs)
    bottom_top_stag = zH.coord("bottom_top_stag").points
    layer_height = (
        zH.extract(Constraint(bottom_top_stag=bottom_top_stag[1:]))
        - zH.extract(Constraint(bottom_top_stag=bottom_top_stag[:-1])).core_data()
    )
    layer_height.rename("layer_height")
    replace_cube = loadramscube(filenames, "T", **kwargs)
    layer_height = replacecoordinates(layer_height, replace_cube)
    return layer_height


def calculate_rams_LWP(filenames, **kwargs):
    from iris.analysis import SUM

    LWC = deriveramscube(filenames, "LWC", **kwargs)
    microphysics_scheme = kwargs.pop("microphysics_scheme", None)
    Airmass = deriveramscube(filenames, "airmass_path", **kwargs)
    LWP = (LWC * Airmass).collapsed(("model_level_number"), SUM)
    LWP.rename("liquid water path")
    # LWP.rename('atmosphere_mass_content_of_cloud_liquid_water')
    return LWP


def calculate_rams_IWP(filenames, **kwargs):
    from iris.analysis import SUM

    IWC = deriveramscube(filenames, "IWC", **kwargs)
    microphysics_scheme = kwargs.pop("microphysics_scheme", None)
    Airmass = deriveramscube(filenames, "airmass_path", **kwargs)
    IWP = (IWC * Airmass).collapsed(("model_level_number"), SUM)
    IWP.rename("ice water path")
    # IWP.rename('atmosphere_mass_content_of_cloud_ice_water')
    return IWP


def calculate_rams_IWV(filenames, **kwargs):
    from iris.analysis import SUM

    QVAPOR = loadramscube(filenames, "QVAPOR", **kwargs)
    Airmass = deriveramscube(filenames, "airmass_path", **kwargs)
    IWV = (QVAPOR * Airmass).collapsed(("model_level_number"), SUM)
    IWV.rename("integrated water vapour")
    # IWV.rename('atmosphere_mass_content_of_water_vapor')
    return IWV


def integrate_cube(variable, Airmass_or_dz, name=None):
    from iris.analysis import SUM

    if name is None:
        name = "integrated_" + variable.name()
    variable_integrated = variable * Airmass_or_dz
    if "geopotential_height" in [
        coord.name() for coord in variable_integrated.coords()
    ]:
        variable_integrated.remove_coord("geopotential_height")
    variable_integrated = variable_integrated.collapsed(("model_level_number"), SUM)
    variable_integrated.rename(name)
    return variable_integrated


def calculate_rams_LWP_fromcubes(LWC, Airmass):
    from iris.analysis import SUM

    LW = LWC * Airmass
    # LW.remove_coord('geopotential_height')
    # !LWP=LW.collapsed(('model_level_number'),SUM)
    LWP = collapse_removecoord(LW, "model_level_number", SUM)
    LWP.rename("liquid water path")
    # LWP.rename('atmosphere_mass_content_of_cloud_liquid_water')
    return LWP


def calculate_rams_IWP_fromcubes(IWC, Airmass):
    from iris.analysis import SUM

    IW = IWC * Airmass
    # IW.remove_coord('geopotential_height')
    # IWP=IW.collapsed(('model_level_number'),SUM)
    IWP = collapse_removecoord(IW, "model_level_number", SUM)

    IWP.rename("ice water path")
    return IWP


def calculate_rams_IWV_fromcubes(QVAPOR, Airmass):
    from iris.analysis import SUM

    VAPOR = QVAPOR * Airmass
    # VAPOR.remove_coord('geopotential_height')
    # IWV=VAPOR.collapsed(('model_level_number'),SUM)
    IWV = collapse_removecoord(VAPOR, "model_level_number", SUM)
    IWV.rename("integrated water vapor")
    return IWV


def calculate_rams_maximum_reflectivity(filenames, **kwargs):
    from iris.analysis import MAX

    REFL_10CM = loadramscube(filenames, "REFL_10CM", **kwargs)
    MAX_REFL_10CM = REFL_10CM.collapsed("model_level_number", MAX)
    MAX_REFL_10CM.rename("maximum reflectivity")
    return MAX_REFL_10CM


def calculate_rams_w_unstaggered(filenames, **kwargs):
    from iris import Constraint, cube

    w = loadramscube(filenames, "W", **kwargs)
    constraint_1 = Constraint(
        bottom_top_stag=lambda cell: cell > w.coord("bottom_top_stag").points[0]
    )
    constraint_2 = Constraint(
        bottom_top_stag=lambda cell: cell < w.coord("bottom_top_stag").points[-1]
    )
    w_unstaggered = cube.Cube(
        0.5
        * (w.extract(constraint_1).core_data() + w.extract(constraint_2).core_data()),
        standard_name="upward_air_velocity",
        units="m/s",
    )
    return w_unstaggered


def calculate_rams_u_unstaggered(filenames, **kwargs):
    from iris import Constraint, cube

    u = loadramscube(filenames, "U", **kwargs)
    constraint_1 = Constraint(
        west_east_stag=lambda cell: cell > u.coord("west_east_stag").points[0]
    )
    constraint_2 = Constraint(
        west_east_stag=lambda cell: cell < u.coord("west_east_stag").points[-1]
    )
    u_unstaggered = cube.Cube(
        0.5
        * (u.extract(constraint_1).core_data() + u.extract(constraint_2).core_data()),
        standard_name="x_wind",
        units="m/s",
    )
    return u_unstaggered


def calculate_rams_v_unstaggered(filenames, **kwargs):
    from iris import Constraint, cube

    v = loadramscube(filenames, "V", **kwargs)
    constraint_1 = Constraint(
        south_north_stag=lambda cell: cell > v.coord("south_north_stag").points[0]
    )
    constraint_2 = Constraint(
        south_north_stag=lambda cell: cell < v.coord("south_north_stag").points[-1]
    )
    v_unstaggered = cube.Cube(
        0.5
        * (v.extract(constraint_1).core_data() + v.extract(constraint_2).core_data()),
        standard_name="y_wind",
        units="m/s",
    )
    return v_unstaggered


def calculate_rams_density(filenames, **kwargs):
    from iris import coords

    if "ALT" in variable_list(filenames):
        alt = loadramscube(filenames, "ALT", **kwargs)
        rho = alt ** (-1)
    else:
        R = coords.AuxCoord(
            287.058, long_name="Specific gas constant for air", units="Joule kg^-1 K^-1"
        )
        p = deriveramscube(filenames, "pressure", **kwargs)
        T = deriveramscube(filenames, "temperature", **kwargs)
        rho = p * ((R * T) ** -1)
        rho.rename("air_density")
    return rho


def calculate_rams_pressure(filenames, **kwargs):
    P = loadramscube(filenames, "P", **kwargs)
    PB = loadramscube(filenames, "PB", **kwargs)
    p = P + PB
    p.rename("pressure")
    return p


def calculate_rams_pressure_stag(filenames, **kwargs):

    p = deriveramscube(filenames, "pressure", **kwargs)
    bottom_top = p.coord("bottom_top").points
    p_stag = 0.5 * (
        p.extract(Constraint(bottom_top=bottom_top[:-1]))
        + p.extract(Constraint(bottom_top=bottom_top[1:])).core_data()
    )
    return p_stag


def calculate_rams_pressure_xstag(filenames, **kwargs):

    p = deriveramscube(filenames, "pressure", **kwargs)
    west_east = p.coord("west_east").points
    p_xstag = 0.5 * (
        p.extract(Constraint(west_east=west_east[:-1]))
        + p.extract(Constraint(west_east=west_east[1:])).core_data()
    )
    p_xstag.rename("pressure")
    return p


def calculate_rams_pressure_ystag(filenames, **kwargs):

    p = deriveramscube(filenames, "pressure", **kwargs)
    south_north = p.coord("south_north").points
    p_ystag = 0.5 * (
        p.extract(Constraint(south_north=south_north[:-1]))
        + p.extract(Constraint(south_north=south_north[1:])).core_data()
    )
    p_ystag.rename("pressure")
    return p_ystag


def calculate_rams_geopotential(filenames, **kwargs):
    PH = loadramscube(filenames, "PH", **kwargs)
    PHB = loadramscube(filenames, "PHB", **kwargs)
    pH = PH + PHB
    pH.rename("geopotential")
    return pH


def calculate_rams_geopotential_height_stag(filenames, **kwargs):
    from iris import coords

    pH = deriveramscube(filenames, "geopotential", **kwargs)
    g = coords.AuxCoord(9.81, long_name="acceleration", units="m s^-2")
    zH = pH / g
    zH.rename("geopotential_height")
    return zH


def calculate_rams_geopotential_height(filenames, **kwargs):

    zH = deriveramscube(filenames, "geopotential_height_stag", **kwargs)
    bottom_top_stag = zH.coord("bottom_top_stag").points
    z = 0.5 * (
        zH.extract(Constraint(bottom_top_stag=bottom_top_stag[:-1]))
        + zH.extract(Constraint(bottom_top_stag=bottom_top_stag[1:])).core_data()
    )
    z.rename("geopotential_height")
    return z


def calculate_rams_geopotential_height_ystag(filenames, **kwargs):
    z = calculate_rams_geopotential_height(filenames, **kwargs)
    z_ystag = cube_interp_extendby1(z, "south_north")
    return z_ystag


def calculate_rams_geopotential_height_xstag(filenames, **kwargs):
    z = calculate_rams_geopotential_height(filenames, **kwargs)
    z_xstag = cube_interp_extendby1(z, "west_east")
    return z_xstag


def unstagger(cube_in, coord, filenames, **kwargs):
    cube_out = cube_interp_reduceby1(cube_in, coord)
    replace_cube = loadramscube(filenames, "T", **kwargs)
    cube_out = replacecoordinates(cube_out, replace_cube)


def array_interp_extendby1(array, dim):
    import numpy as np

    idx1 = [slice(None)] * (array.ndim)
    idx2 = [slice(None)] * (array.ndim)
    idx_start = [slice(None)] * (array.ndim)
    idx_end = [slice(None)] * (array.ndim)
    idx1[dim] = slice(1, None)
    idx2[dim] = slice(0, -1)
    idx_start[dim] = slice(1)
    idx_end[dim] = slice(-2, -1)
    array_out = np.concatenate(
        (array[idx_start], 0.5 * (array[idx1] + array[idx2]), array[idx_end]), axis=dim
    )
    return array_out


def array_interp_reduceby1(array, dim):
    idx = [slice(None)] * (array.ndim)
    idx1 = idx
    idx2 = idx
    idx1[dim] = slice(1, None)
    idx2[dim] = slice(None, -1)

    array_out = 0.5 * (array[idx1] + array[idx2])
    return array_out


def cube_interp_extendby1(cube_in, coord):
    import dask.array as da

    dim = cube_in.coord_dims(coord)[0]
    cube_data = cube_in.core_data()
    ndim = cube_in.ndim
    idx1 = [slice(None)] * (ndim)
    idx2 = [slice(None)] * (ndim)
    idx_start = [slice(None)] * (ndim)
    idx_end = [slice(None)] * (ndim)
    idx1[dim] = slice(1, None)
    idx2[dim] = slice(0, -1)
    idx_start[dim] = slice(1)
    idx_end[dim] = slice(-2, -1)
    cube_data[idx_start]
    cube_data[idx1]
    cube_data[idx2]
    cube_data[idx_end]
    array_out = da.concatenate(
        (
            cube_data[idx_start],
            0.5 * (cube_data[idx1] + cube_data[idx2]),
            cube_data[idx_end],
        ),
        axis=dim,
    )

    #    idx1[dim]=slice(1)
    #    array_out=np.concatenate((cube_data[idx1],cube_data),axis=dim)
    return array_out


def cube_interp_reduceby1(cube_in, coord):
    dim = cube_in.coord_dims(coord)
    ndim = cube_in.ndim
    idx1 = [slice(None)] * (ndim)
    idx2 = [slice(None)] * (ndim)
    idx1[dim] = slice(1, None)
    idx2[dim] = slice(None, -1)

    cube_out = 0.5 * (cube_in[idx1] + cube_in[idx2].core_data())
    return cube_out


def load_sum(filename, list_variables, **kwargs):
    cube_out = load(filename, list_variables[0], **kwargs)
    for variable in list_variables[1:]:
        cube_out = cube_out + load(filename, variable, **kwargs)
    return cube_out


def remove_all_coordinates(variable_cube):
    for coordinate in variable_cube.coords():
        variable_cube.remove_coord(coordinate.name())
    return variable_cube


def replacecoordinates(variable_cube, replace_cube):
    variable_cube_out = replace_cube
    variable_cube_out.data = variable_cube.core_data()
    variable_cube_out.rename(variable_cube.name())
    variable_cube_out.units = variable_cube.units
    variable_cube_out.attributes = {}
    return variable_cube_out


# def addcoordinates(filenames, variable,variable_cube,**kwargs):
##    if 'add_coordinates' in kwargs:
##        add_coordinates=kwargs['add_coordinates']
##    else:
##        add_coordinates=None
##
##    if add_coordinates==None:
##        variable_cube=add_dim_coordinates(filenames, variable,variable_cube,**kwargs)
##    else:
##        variable_cube=add_dim_coordinates(filenames, variable,variable_cube,**kwargs)
##        variable_cube=add_aux_coordinates_1dim(filenames, variable,variable_cube,**kwargs)
#    variable_cube=add_dim_coordinates(filenames, variable,variable_cube,**kwargs)
#    variable_cube=add_aux_coordinates_1dim(filenames, variable,variable_cube)
#    variable_cube=add_aux_coordinates_1dim(filenames, variable,variable_cube)
#    return variable_cube
#
# def add_time_coordinate(filenames, variable,variable_cube):
#    time=make_time_coord(filenames)
#    variable_cube.add_dim_coord(time,variable_cube.coord_dims('XTIME')[0])
#    return variable_cube


# def add_dim_coordinates(filenames, variable,variable_cube,add_coordinates=None):
#    from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
#    from iris import load_cube
#    variable_cube_dim= load_cube(filenames, variable)
#
#    attributes=variable_cube_dim.attributes
#    nc_id=Dataset(filenames)
#    nc_variable=nc_id.variables[variable]
#    variable_dimensions=nc_variable.dimensions
#    [str(line) for line in variable_dimensions]
#    DX=attributes['DX']
#    DY=attributes['DY']
#    WEST_EAST_PATCH_END_UNSTAG=attributes['WEST-EAST_PATCH_END_UNSTAG']
#    SOUTH_NORTH_PATCH_END_UNSTAG=attributes['SOUTH-NORTH_PATCH_END_UNSTAG']
#    BOTTOM_TOP_PATCH_END_UNSTAG=attributes['BOTTOM-TOP_PATCH_END_UNSTAG']
#    WEST_EAST_PATCH_END_STAG=attributes['WEST-EAST_PATCH_END_STAG']
#    SOUTH_NORTH_PATCH_END_STAG=attributes['SOUTH-NORTH_PATCH_END_STAG']
#    BOTTOM_TOP_PATCH_END_STAG=attributes['BOTTOM-TOP_PATCH_END_STAG']
#    for dim in range(len(variable_dimensions)):
##        if (variable_dimensions[dim]=='Time'):
##           time=make_time_coord(filenames)
##           variable_cube.add_dim_coord(time,dim)
#        if (variable_dimensions[dim]=='west_east'):
#            west_east=make_westeast_coord(DX,WEST_EAST_PATCH_END_UNSTAG)
#            variable_cube.add_dim_coord(west_east,dim)
#        elif (variable_dimensions[dim]=='south_north'):
#           south_north=make_southnorth_coord(DY, SOUTH_NORTH_PATCH_END_UNSTAG)
#           variable_cube.add_dim_coord(south_north,dim)
#        elif (variable_dimensions[dim]=='bottom_top'):
#           bottom_top=make_bottom_top_coordinate(BOTTOM_TOP_PATCH_END_UNSTAG)
#           variable_cube.add_dim_coord(bottom_top,dim)
#           model_level_number=make_model_level_number_coordinate(BOTTOM_TOP_PATCH_END_UNSTAG)
#           variable_cube.add_aux_coord(model_level_number,dim)
#        elif variable_dimensions[dim]=='west_east_stag':
#           west_east_stag=make_westeast_stag_coord(DX,WEST_EAST_PATCH_END_STAG)
#           variable_cube.add_dim_coord(west_east_stag,dim)
#        elif variable_dimensions[dim]=='south_north_stag':
#           south_north_stag=make_southnorth_stag_coord(DY, SOUTH_NORTH_PATCH_END_STAG)
#           variable_cube.add_dim_coord(south_north_stag,dim)
#        elif variable_dimensions[dim]=='bottom_top_stag':
#           bottom_top_stag=make_bottom_top_stag_coordinate(BOTTOM_TOP_PATCH_END_STAG)
#           variable_cube.add_dim_coord(bottom_top_stag,dim)
#           model_level_number=make_model_level_number_coordinate(BOTTOM_TOP_PATCH_END_STAG)
#           variable_cube.add_aux_coord(model_level_number,dim)
#
#    return variable_cube

# def add_aux_coordinates_1dim(filenames, variable,variable_cube):#,add_coordinates=None):
#    from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
#    from iris import load_cube
#
#    variable_cube_dim= load_cube(filenames, variable)
#    attributes=variable_cube_dim.attributes
#    nc_id=Dataset(filenames)
#
#    nc_variable=nc_id.variables[variable]
#    variable_dimensions=nc_variable.dimensions
#    [str(line) for line in variable_dimensions]
#    DX=attributes['DX']
#    DY=attributes['DY']
#    WEST_EAST_PATCH_END_UNSTAG=attributes['WEST-EAST_PATCH_END_UNSTAG']
#    SOUTH_NORTH_PATCH_END_UNSTAG=attributes['SOUTH-NORTH_PATCH_END_UNSTAG']
#    #BOTTOM_TOP_PATCH_END_UNSTAG=attributes['BOTTOM-TOP_PATCH_END_UNSTAG']
#    WEST_EAST_PATCH_END_STAG=attributes['WEST-EAST_PATCH_END_STAG']
#    SOUTH_NORTH_PATCH_END_STAG=attributes['SOUTH-NORTH_PATCH_END_STAG']
#    #BOTTOM_TOP_PATCH_END_STAG=attributes['BOTTOM-TOP_PATCH_END_STAG']
#    coord_system=make_coord_system(attributes)
#    coords=variable_cube.coords()
##    if type(add_coordinates)!=list:
##        add_coordinates1=add_coordinates
##        add_coordinates=[]
##        add_coordinates.append(add_coordinates1)
##    for coordinate in add_coordinates:
##        if coordinate=='xy':
#    for dim in range(len(coords)):
#        if (coords[dim].name()=='west_east'):
#            projection_x_coord=make_x_coord(DX,WEST_EAST_PATCH_END_UNSTAG,coord_system=coord_system)
#            variable_cube.add_aux_coord(projection_x_coord,dim)
#            x_coord=AuxCoord(variable_cube.coord('west_east').points,long_name='x',units=1)
#            variable_cube.add_aux_coord(x_coord,data_dims=variable_cube.coord_dims('west_east'))
#
#        elif (coords[dim].name()=='south_north'):
#            projection_y_coord=make_y_coord(DY, SOUTH_NORTH_PATCH_END_UNSTAG,coord_system=coord_system)
#            variable_cube.add_aux_coord(projection_y_coord,dim)
#            y_coord=AuxCoord(variable_cube.coord('south_north').points,long_name='y',units=1)
#            variable_cube.add_aux_coord(y_coord,data_dims=variable_cube.coord_dims('south_north'))
#
#        elif (coords[dim].name()=='west_east_stag'):
#            projection_x_stag_coord=make_x_stag_coord(DX,WEST_EAST_PATCH_END_STAG,coord_system=coord_system)
#            variable_cube.add_aux_coord(projection_x_stag_coord,dim)
#            x_coord=AuxCoord(variable_cube.coord('west_east_stag').points,long_name='x',units=1)
#            variable_cube.add_aux_coord(x_coord,data_dims=variable_cube.coord_dims('west_east_stag'))
#
#        elif coords[dim].name()=='south_north_stag':
#            projection_y_stag_coord=make_y_stag_coord(DY, SOUTH_NORTH_PATCH_END_STAG,coord_system=coord_system)
#            variable_cube.add_aux_coord(projection_y_stag_coord,dim)
#            y_coord=AuxCoord(variable_cube.coord('south_north_stag').points,long_name='y',units=1)
#            variable_cube.add_aux_coord(y_coord,data_dims=variable_cube.coord_dims('south_north_stag'))
#
#    return variable_cube
#


def add_aux_coordinates_multidim(
    filenames, variable_cube, add_coordinates=None, constraint=None, **kwargs
):
    from copy import deepcopy

    coords = variable_cube.coords()
    add_coordinates_new = deepcopy(add_coordinates)

    for entry in ["z", "p", "latlon", "zp", "pz"]:
        if entry in add_coordinates_new:
            add_coordinates_new.remove(entry)

    # if constraint in kwargs:
    #     constraint=kwargs.pop('constraint')

    if type(add_coordinates) != list:
        add_coordinates1 = add_coordinates
        add_coordinates = []
        add_coordinates.append(add_coordinates1)

    for coordinate in add_coordinates:

        if coordinate == "z":

            if (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east"
            ):
                z_coord = make_z_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "bottom_top"
                and coords[1].name() == "south_north"
                and coords[2].name() == "west_east"
            ):
                z_coord = make_z_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east_stag"
            ):
                z_coord = make_z_xstag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "bottom_top"
                and coords[1].name() == "south_north"
                and coords[2].name() == "west_east_stag"
            ):
                z_coord = make_z_xstag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north_stag"
                and coords[3].name() == "west_east"
            ):
                z_coord = make_z_ystag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "bottom_top"
                and coords[1].name() == "south_north_stag"
                and coords[2].name() == "west_east"
            ):
                z_coord = make_z_ystag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top_stag"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east"
            ):
                z_stag_coord = make_z_stag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_stag_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "bottom_top_stag"
                and coords[1].name() == "south_north"
                and coords[2].name() == "west_east"
            ):
                z_stag_coord = make_z_stag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_stag_coord, (0, 1, 2))
            else:
                raise ValueError("no z coordinates added")

        if coordinate == "pressure":
            if (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east"
            ):
                p_coord = make_p_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east_stag"
            ):
                p_coord = make_p_xstag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord)
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north_stag"
                and coords[3].name() == "west_east"
            ):
                p_coord = make_p_ystag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top_stag"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east"
            ):
                p_coord = make_p_stag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord, (0, 1, 2, 3))
            else:
                raise ValueError("no p coordinates added")

        if coordinate == "zp" or coordinate == "pz":
            if (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east"
            ):
                z_coord = make_z_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2, 3))
                p_coord = make_p_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east_stag"
            ):
                z_coord = make_z_xstag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2, 3))
                p_coord = make_p_xstag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top"
                and coords[2].name() == "south_north_stag"
                and coords[3].name() == "west_east"
            ):
                z_coord = make_z_ystag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_coord, (0, 1, 2, 3))
                p_coord = make_p_ystag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord, (0, 1, 2, 3))
            elif (
                coords[0].name() == "time"
                and coords[1].name() == "bottom_top_stag"
                and coords[2].name() == "south_north"
                and coords[3].name() == "west_east"
            ):
                z_stag_coord = make_z_stag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(z_stag_coord, (0, 1, 2, 3))
                p_coord = make_p_stag_coordinate(
                    filenames,
                    add_coordinates=add_coordinates_new,
                    constraint=None,
                    **kwargs,
                )
                variable_cube.add_aux_coord(p_coord, (0, 1, 2, 3))
            else:
                raise ValueError("no z and p and lat/lon coordinates added")
    return variable_cube


def make_time_coord(filenames):
    from datetime import datetime

    from cf_units import CALENDAR_STANDARD, date2num
    from iris import coords, load_cube

    Times = load_cube(filenames, "Times")
    filetimes = Times.data
    #    filetimelist = []   # Will contain list of times in seconds since model start time in file.
    timeobjlist = []  # Will contain list of corresponding datetime objects
    for i, filetime in enumerate(filetimes):
        timeobj = datetime.strptime(
            filetime.tostring().decode("UTF-8"), "%Y-%m-%d_%H:%M:%S"
        )
        #        if i == 0:
        #            timeobj0 = timeobj
        #        time_dt = timeobj-timeobj0 # timedelta object representing difference in time from start time
        #        filetimelist.append(time_dt.seconds)
        timeobjlist.append(timeobj)
    #    time_days=empty(len(timeobjlist))
    #    if timeobjlist[0]==datetime(1,1,1,0,0,0):
    #        for i,time in enumerate(timeobjlist):
    #            timeobjlist[i]=timeobjlist[i].replace(year=2000)
    #    #Include a different base_date for dates close to 0001-01-01 (idealised simulations)

    # if timeobjlist[0]<datetime(100,1,1):
    #     base_date=datetime(1,1,1)
    # else:
    base_date = datetime(1970, 1, 1)
    time_units = "days since " + base_date.strftime("%Y-%m-%d")
    #    for i in range(len(timeobjlist)):
    #        time_days[i]=(timeobjlist[i] - base_date).total_seconds() / timedelta(1).total_seconds()

    time_days = date2num(timeobjlist, time_units, CALENDAR_STANDARD)
    time_coord = coords.DimCoord(
        time_days,
        standard_name="time",
        long_name="time",
        var_name="time",
        units=time_units,
        bounds=None,
        attributes=None,
        coord_system=None,
        circular=False,
    )
    return time_coord


def make_westeast_coord(DX, WEST_EAST_PATCH_END_UNSTAG):
    from iris import coords
    from numpy import arange

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
    from iris import coords
    from numpy import arange

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
    from iris import coords
    from numpy import arange  # DY=attributes['DY']

    # SOUTH_NORTH_PATCH_END_UNSTAG=attributes['SOUTH-NORTH_PATCH_END_UNSTAG']
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
    from iris import coords
    from numpy import arange

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
    from iris import coords
    from numpy import arange

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
    from iris import coords
    from numpy import arange

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
    from iris import coords
    from numpy import arange

    MODEL_LEVEL_NUMBER = arange(0, BOTTOM_TOP_PATCH_END)
    model_level_number = coords.AuxCoord(
        MODEL_LEVEL_NUMBER, standard_name="model_level_number", units="1"
    )
    return model_level_number


def make_coord_system(attributes):
    from iris import coord_systems

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
    # MAP_PROJ = attributes["MAP_PROJ"] # not sure what the MAP_PROJ is for, it looks like an integer which also specifies the coordinate system

    # cartesian coordinate system (idealized simulations):
    if MAP_PROJ_CHAR == "Cartesian":  # and MAP_PROJ == 0:
        coord_system = None

    # lambert Conformal system (idealized simulations):
    if MAP_PROJ_CHAR == "Lambert Conformal":  # and MAP_PROJ == 1:
        CEN_LON = attributes["CEN_LON"]
        TRUELAT1 = attributes["TRUELAT1"]
        TRUELAT2 = attributes["TRUELAT2"]
        MOAD_CEN_LAT = attributes["MOAD_CEN_LAT"]
        STAND_LON = attributes["STAND_LON"]
        POLE_LAT = attributes["POLE_LAT"]
        POLE_LON = attributes["POLE_LON"]
        coord_system = coord_systems.LambertConformal(
            central_lat=MOAD_CEN_LAT,
            central_lon=CEN_LON,
            false_easting=0.0,
            false_northing=0.0,
            secant_latitudes=(TRUELAT1, TRUELAT2),
        )
    if MAP_PROJ_CHAR == "Polar-Stereographic":
        CEN_LAT = attributes["CEN_LAT"]
        CEN_LON = attributes["CEN_LON"]
        coord_system = coord_systems.PolarStereographic(
            central_lat=CEN_LAT,
            central_lon=CEN_LON,
            false_easting=0.0,  # I am not sure of any of these, but for now leave them as standard
            false_northing=0.0,
            true_scale_lat=None,
            scale_factor_at_projection_origin=None,
            ellipsoid=None,
        )
    return coord_system


def make_x_coord(DX, WEST_EAST_PATCH_END_UNSTAG, coord_system):
    from iris import coords
    from numpy import arange, array, transpose

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
    from iris import coords
    from numpy import arange

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
    from iris import coords
    from numpy import arange, array, transpose  # DY=attributes['DY']

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
    from iris import coords
    from numpy import arange

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


def make_z_coordinate(filenames, **kwargs):
    from iris import coords

    z = calculate_rams_geopotential_height(filenames, **kwargs)
    z_coord = coords.AuxCoord(
        z.core_data(),
        standard_name="geopotential_height",
        long_name="geopotential_height",
        var_name="z",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return z_coord


def make_z_xstag_coordinate(filenames, **kwargs):
    from iris import coords

    z = calculate_rams_geopotential_height_xstag(filenames, **kwargs)
    z_coord = coords.AuxCoord(
        z,
        standard_name="geopotential_height",
        long_name="geopotential_height",
        var_name="z",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return z_coord


def make_z_ystag_coordinate(filenames, **kwargs):
    from iris import coords

    z = calculate_rams_geopotential_height_ystag(filenames, **kwargs)
    z_coord = coords.AuxCoord(
        z,
        standard_name="geopotential_height",
        long_name="geopotential_height",
        var_name="z",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return z_coord


def make_z_stag_coordinate(filenames, **kwargs):
    from iris import coords

    z = calculate_rams_geopotential_height_stag(filenames, **kwargs)
    z_coord = coords.AuxCoord(
        z.core_data(),
        standard_name="geopotential_height",
        long_name="z",
        var_name="z",
        units="m",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return z_coord


def make_p_coordinate(filenames, **kwargs):
    from iris import coords

    p = calculate_rams_pressure(filenames, **kwargs)
    p_coord = coords.AuxCoord(
        p.core_data(),
        standard_name=None,
        long_name="pressure",
        var_name="pressure",
        units="Pa",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return p_coord


def make_p_xstag_coordinate(filenames, **kwargs):
    from iris import coords

    p = calculate_rams_pressure_xstag(filenames, **kwargs)
    p_coord = coords.AuxCoord(
        p.core_data(),
        standard_name=None,
        long_name="pressure",
        var_name="pressure",
        units="Pa",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return p_coord


def make_p_ystag_coordinate(filenames, **kwargs):
    from iris import coords

    p = calculate_rams_pressure_ystag(filenames, **kwargs)
    p_coord = coords.AuxCoord(
        p.core_data(),
        standard_name=None,
        long_name="pressure",
        var_name="pressure",
        units="Pa",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return p_coord


def make_p_stag_coordinate(filenames, **kwargs):
    from iris import coords

    p = calculate_rams_pressure_stag(filenames, **kwargs)
    p_coord = coords.AuxCoord(
        p.core_data(),
        standard_name=None,
        long_name="pressure",
        var_name="pressure",
        units="Pa",
        bounds=None,
        attributes=None,
        coord_system=None,
    )
    return p_coord


def collapse_removecoord(cube, coord, aggregator):
    for coordinate in cube.coords():
        if coordinate.ndim > 1 and cube.coord_dims(coord)[0] in cube.coord_dims(
            coordinate
        ):
            cube.remove_coord(coordinate.name())
    cube = cube.collapsed((coord), aggregator)
    return cube
