import glob
from datetime import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm


# TODO: Remove this and replace with RAMS-MAT package
def configure_rams(dataset0, path_to_header, CONFIG=None, configure_variables = []):
    """
    Serves to set up proper attributes, coordinates, and dimensions so that the RAMS data set is compatible with CoCoMET
    Inputs:
        dataset: an xarray of RAMS data

    Outputs:
        configured_dataset: an xarray of RAMS data which has the proper attributes, coordinates, and dimensions
    """

    # avoid errors with CONFIG file
    if CONFIG is None:
        CONFIG = {"verbose": False, "rams": {}}
    
    # since grid 3 of RAMS dataset is large, create a list of only the necessary variables and a new dataset to reduce required computational power
    ListOfNeededVariables = [
        "GLAT",
        "GLON",
        "TOPT",  # lat, lon, and topographical height
        "PI",
        "THETA",  # pressure and temperature var
        # "RV",
        # "RCP",
        # "RDP",  # mixing ratio (dbz calculation)
        # "RRP",
        # "RPP",
        # "RSP",
        # "RAP",
        # "RGP",
        # "RHP",
        # "CCP",
        # "CDP",
        # "CRP",  # concentration numbers (dbz calculation)
        # "CPP",
        # "CSP",
        # "CAP",
        # "CGP",
        # "CHP",
        # "PCPVR",
        # "PCPVS",
        # "PCPVA",  # 3D precipitation rates
        # "PCPVG",
        # "PCPVH",
        # "PCPVD",
        # "LWUP",  # longwave radiation (brightness temp calculation)
        # "DN0",  # dry air density (experimental for precipitation rate calculation)
    ]

    # Add in the tracking variable(s)
    ListOfNeededVariables.extend(configure_variables)

    dataset = xr.Dataset({})
    if CONFIG["verbose"]:
        for var in tqdm(
            ListOfNeededVariables,
            desc="=====Creating Truncated RAMS Dataset=====",
            total=len(ListOfNeededVariables),
        ):

            try:
                dims = dataset0[var].dims
                if len(dims) == 3:
                    dataset[var] = ([*dims], dataset0[var].data[:, 1:, 1:])
                    dataset[var].chunk(
                        dataset0[var][:, 1:, 1:].chunksizes
                    )  # rechunk the variables
                elif len(dims) == 4:
                    dataset[var] = ([*dims], dataset0[var].data[:, 1:, 1:, 1:])
                    dataset[var].chunk(
                        dataset0[var][:, 1:, 1:, 1:].chunksizes
                    )  # rechunk the variables
            
            except:
                continue

    else:
        for var in ListOfNeededVariables:

            try:
                dims = dataset0[var].dims
                if len(dims) == 3:
                    dataset[var] = ([*dims], dataset0[var].data[:, 1:, 1:])
                    dataset[var].chunk(
                        dataset0[var][:, 1:, 1:].chunksizes
                    )  # rechunk the variables
                elif len(dims) == 4:
                    dataset[var] = ([*dims], dataset0[var].data[:, 1:, 1:, 1:])
                    dataset[var].chunk(
                        dataset0[var][:, 1:, 1:, 1:].chunksizes
                    )  # rechunk the variables]
            
            except:
                continue

    # destagger winds
    pbar = tqdm(
        np.arange(6),
        desc="=====Destaggering Winds and Adding Coordinates/Attributes=====",
        total=6,
    )
    windDims = dataset0["WC"].dims
    WC = 0.5 * (dataset0["WC"][:, 1:, 1:, 1:] + dataset0["WC"][:, :-1, 1:, 1:])
    dataset["WC"] = ([*windDims], WC.data)
    dataset["WC"].chunk(dataset0["WC"][:, 1:, 1:, 1:].chunksizes)
    pbar.update()
    pbar.refresh()
    VC = 0.5 * (dataset0["VC"][:, 1:, 1:, 1:] + dataset0["VC"][:, :-1, 1:, 1:])
    dataset["VC"] = ([*windDims], VC.data)
    dataset["VC"].chunk(dataset0["VC"][:, 1:, 1:, 1:].chunksizes)
    pbar.update()
    pbar.refresh()
    UC = 0.5 * (dataset0["UC"][:, 1:, 1:, 1:] + dataset0["UC"][:, :-1, 1:, 1:])
    dataset["UC"] = ([*windDims], UC.data)
    dataset["UC"].chunk(dataset0["UC"][:, 1:, 1:, 1:].chunksizes)
    pbar.update()
    pbar.refresh()

    # rename phony dimensions to dimensions that make sense
    dataset = dataset.rename(
        {
            "phony_dim_0": "south_north",
            "phony_dim_1": "west_east",
            "phony_dim_2": "bottom_top",
        }
    )  # , 'phony_dim_3' : 'np', 'phony_dim_4' : 'nzs', 'phony_dim_5' : 'local_topography_height'})
    pbar.update()
    pbar.refresh()

    # define the coordinates
    dataset = dataset.assign_coords(
        GLAT=(["Time", "south_north", "west_east"], dataset["GLAT"].data)
    )
    dataset = dataset.assign_coords(
        GLON=(["Time", "south_north", "west_east"], dataset["GLON"].data)
    )
    pbar.update()
    pbar.refresh()

    # Since RAMS does not have any attributes, add them. These are necessary for the iris cube generation
    dataset.attrs["CEN_LAT"] = np.median(dataset["GLAT"].values)
    dataset.attrs["CEN_LON"] = np.median(dataset["GLON"].values)
    dataset.attrs["WEST-EAST_PATCH_END_UNSTAG"] = len(dataset["west_east"])
    dataset.attrs["SOUTH-NORTH_PATCH_END_UNSTAG"] = len(dataset["south_north"])
    dataset.attrs["BOTTOM-TOP_PATCH_END_UNSTAG"] = len(dataset["bottom_top"])
    dataset.attrs["BOTTOM-TOP_PATCH_START_UNSTAG"] = 0
    dataset.attrs["SOUTH-NORTH_PATCH_START_UNSTAG"] = 0
    dataset.attrs["WEST-EAST_PATCH_START_UNSTAG"] = 0
    dataset.attrs["MAP_PROJ_CHAR"] = "Polar-Stereographic"
    pbar.update()
    pbar.refresh()

    # read in header file metadata
    # use two header files next to each other to find the time difference
    # read in grid spacing constants and altitudes
    listing = glob.glob(path_to_header)
    firstHeaderStr = listing[0]
    if len(listing) > 1:
        times = np.zeros(len(listing))
        datetime_times = []
        for ind, i in tqdm(
            enumerate(listing),
            desc="=====Loading RAMS Header File Data=====",
            total=len(listing),
        ):
            f = np.loadtxt(i, skiprows=findWhereColsChange(i), dtype=str)
            time_now = float(f[np.where(f == "__time")[0] + 2][0])
            times[ind] = time_now

            # use the last loaded file for the initial date
            year = int(f[np.where(f == "__iyear1")[0] + 2][0])
            month = int(f[np.where(f == "__imonth1")[0] + 2][0])
            date = int(f[np.where(f == "__idate1")[0] + 2][0])
            itime = f[np.where(f == "__itime1")[0] + 2][0]

            hour = int(np.floor(time_now / 3600))
            time_now -= hour * 3600
            minute = int(np.floor(time_now / 60))
            time_now -= minute * 60
            second = int(time_now)
            date_str_unformatted = f"{year}-{month}-{date} {int(itime[:2]) + hour}:{int(itime[2:]) + minute}:{second}"
            datetime_date = datetime.strptime(
                date_str_unformatted, "%Y-%m-%d %H:%M:%S"
            )
            datetime_times.append(datetime_date)

            # declare the units from the initial time
            if ind == 0:
                date_str_formatted = datetime.strftime(
                    datetime_date, "%Y-%m-%d %H:%M:%S"
                    )                
                dataset.attrs["date"] = f"minutes since {date_str_formatted}"
        

        DT = times[1:] - times[:-1]
        if sum(~(DT == DT[0])) != 0:
            raise Exception("Header file time stamps are non-continuous")
        dataset.attrs["DT"] = DT[0]

    # Now that the times have been parsed, write the times coordinate
    dataset = dataset.assign_coords(Times=(["Time"], datetime_times))

    # use one header file to define the linear projection constants
    headerfile = np.loadtxt(
        firstHeaderStr, dtype=str, skiprows=findWhereColsChange(firstHeaderStr)
    )
    DX = float(headerfile[np.where(headerfile == "__xmn03")[0][0] + 3]) - float(
        headerfile[np.where(headerfile == "__xmn03")[0][0] + 2]
    )
    DY = float(headerfile[np.where(headerfile == "__ymn03")[0][0] + 3]) - float(
        headerfile[np.where(headerfile == "__ymn03")[0][0] + 2]
    )
    dataset.attrs["DY"] = DY
    dataset.attrs["DX"] = DX

    # calculate projection coordinates using linear projection constants
    proj_y_values = dataset.DY * (np.arange(0, dataset.south_north.shape[0]) + 0.5)
    proj_x_values = dataset.DX * (np.arange(0, dataset.west_east.shape[0]) + 0.5)

    dataset["PROJY"] = ("south_north", proj_y_values)
    dataset["PROJX"] = ("west_east", proj_x_values)

    # read in gamma shape parameters, used for reflectivity calculation
    IDgnu = np.where(headerfile == "__gnu")[0][0]
    numOfGnuPoints = int(headerfile[IDgnu + 1])
    startgnu = IDgnu + 2
    stopgnu = IDgnu + numOfGnuPoints + 2
    dataset.attrs["gnu"] = np.array(headerfile[startgnu:stopgnu], dtype=float)

    # calculate the altitude and geopotential
    unstagid = f"__zmn03"
    IDunstag = np.where(headerfile == unstagid)[0][0]

    numOfunstagPoints = int(headerfile[IDunstag + 1])
    startunstag = IDunstag + 2
    stopunstag = IDunstag + numOfunstagPoints + 2
    __zmn03 = np.array(headerfile[startunstag:stopunstag], dtype=float)[
        1:
    ]  # the first data point is unphysical

    # also add altitudes and geopotential to rams_xarray
    dataset["altitudes"] = (["bottom_top"], __zmn03)

    # if you are using moaap, calculate the geopotential
    if "moaap" in CONFIG["rams"]:
        TOPT = dataset["TOPT"].values
        geopt = np.zeros_like(dataset["THETA"])
        for t in tqdm(
            range(TOPT.shape[0]),
            desc="=====Calculating Geopotential for MOAAP Tracking=====",
            total=TOPT.shape[0],
        ):
            alt = np.array(
                np.expand_dims(dataset["altitudes"].values, (1, 2)), dtype=float
            )
            geopt[t, :, :, :] = (
                np.array(np.expand_dims(TOPT, 1)[t, :, :, :], dtype=float) + alt
            ) * 9.81
        geopt = xr.DataArray(
            geopt, dims=["Time", "bottom_top", "south_north", "west_east"]
        )
        geopt.chunk(dataset["THETA"].chunksizes)
        dataset["geopt"] = (
            ["Time", "bottom_top", "south_north", "west_east"],
            geopt.data,
        )

    return dataset


def findWhereColsChange(file):
    """
    Find where in a txt file the number of columns change for the first time,
    then return that index to be used as the "skiprows" argument of np.loadtxt

    Inputs:
        file: a txt file with varying column numbers

    Returns:
        displacer: an integer index of the first row in the txt file where the number of columns change
    """
    with open(file) as f:
        displacer = 1
        for line in f.readlines()[1:]:
            displacer += 1
            if len(line.split()) == 1:
                break
    return displacer
