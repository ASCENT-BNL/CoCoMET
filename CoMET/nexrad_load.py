#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:28:49 2024

@author: thahn
"""

# =============================================================================
# Loads in and grids the NEXRAD Arhcival Level 2 Data using ARM-Pyart then converts into iris cubes and xarray Datasets for use in trackers
# =============================================================================

"""
Inputs:
    path_to_files: glob type path to the NEXRAD level 2 archive files
    save_location: path to where the gridded NEXRAD files should be saved to, should be a directory and end with '/'
    tracking_var: ['dbz'], variable which is going to be used for tracking--reflectivity
    CONFIG: User configuration file
    parallel_processing: [True, False], bool determinig whether to use parallel processing when gridding files
    max_cores: Number of cores to use if parallel_processing == True
"""
def gen_and_save_nexrad_grid(path_to_files, save_location, tracking_var, CONFIG, parallel_processing = False, max_cores = None):
    import os
    import glob
    import pyart
    
    # If parallel processing is enabled, run that version and return
    if (parallel_processing):
        gen_and_save_nexrad_grid_multi(glob.glob(path_to_files), save_location, tracking_var, CONFIG, max_cores)
        return
    
    # Get all archive files and iterate over them
    files = glob.glob(path_to_files)
    # Extract just the filenames from the paths without the file extensions
    file_names = [os.path.basename(f).split('.')[0] for f in files]
    
    for idx, ff in enumerate(files):
        
        if (CONFIG['verbose']): print(f"=====GRIDDING NEXRAD, ({'%5.2f' % ((idx+1)/len(files)*100) + '%'})=====")
        
        if (tracking_var.lower() == 'dbz'):
            # Create radar object including only field of interest
            radar = pyart.io.read_nexrad_archive(ff,include_fields='reflectivity')
        
            # Create radar grid using user-defined params
            radar_grid = pyart.map.grid_from_radars(radar, grid_shape=CONFIG['grid_shape'], grid_limits=CONFIG['grid_limits'], gridding_algo='map_gates_to_grid',
                h_factor=CONFIG['h_factor'], nb=CONFIG['nb'], bsp=CONFIG['bsp'], min_radius=CONFIG['min_radius'])

            # Save radar grid to save_location as a netcdf file and delete radar and radar_grid objects to free memory
            pyart.io.write_grid(save_location + file_names[idx] + '_grid.nc', radar_grid, arm_alt_lat_lon_variables=True, write_point_x_y_z=True, write_point_lon_lat_alt=True)
        
            del radar
            del radar_grid
        
        else:
            raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
            return
        
    return
    



"""
Helper Function for Parallel Processing
"""
def create_and_save_grid_single(file, save_location, tracking_var, CONFIG):
    import os
    import pyart
    
    if (CONFIG['verbose']): print(f"=====PARALLEL GRIDDING NEXRAD, ({file})=====")
    
    if (tracking_var.lower() == 'dbz'):
        # Create radar object including only field of interest
        radar = pyart.io.read_nexrad_archive(file,include_fields='reflectivity')
    
        # Create radar grid using user-defined params
        radar_grid = pyart.map.grid_from_radars(radar, grid_shape=CONFIG['grid_shape'], grid_limits=CONFIG['grid_limits'], gridding_algo='map_gates_to_grid',
            h_factor=CONFIG['h_factor'], nb=CONFIG['nb'], bsp=CONFIG['bsp'], min_radius=CONFIG['min_radius'])

        # Save radar grid to save_location as a netcdf file and delete radar and radar_grid objects to free memory
        file_name = os.path.basename(file).split('.')[0]
        pyart.io.write_grid(save_location + file_name + '_grid.nc', radar_grid, arm_alt_lat_lon_variables=True, write_point_x_y_z=True, write_point_lon_lat_alt=True)
    
        del radar
        del radar_grid
    
    else:
        raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
        return
    
"""
Inputs:
    files: list containing all paths to NEXRAD level 2 archive files
    save_location: path to where the gridded NEXRAD files should be saved to
    tracking_var: ['dbz'], variable which is going to be used for tracking--reflectivity
    CONFIG: User configuration file
    max_cores: Number of cores to use for parallel processing
"""
def gen_and_save_nexrad_grid_multi(files, save_location, tracking_var, CONFIG, max_cores):
    import multiprocessing
    from functools import partial
    
    if (len(files) == 0):
        raise Exception(f'!=====No Files Present to Grid=====!')
        return
    
    # Start a pool with max_cores and run the grid function
    multi_pool = multiprocessing.Pool(max_cores)
    multi_pool.map(partial(create_and_save_grid_single, save_location=save_location, tracking_var=tracking_var, CONFIG=CONFIG), files)
    multi_pool.close()
    multi_pool.join()
    
    return



"""
Inputs:
    path_to_files: Glob path to input files, either archival or grided netcdf--i.e. "/data/usr/KVNX*_V06.ar2v"
    file_type: ['ar2v', 'nc'] type of input file--either archival or netcdf
    tracking_var: ['dbz'], variable which is going to be used for tracking--reflectivity.
    CONFIG: User configuration file
    save_location: Where to save gridded NEXRAD data to if file_type=='ar2v'
Outputs:
    nexrad_xarray: Xarray dataset containing gridded NEXRAD archival data
"""
def nexrad_load_netcdf_iris(path_to_files, file_type, tracking_var, CONFIG, save_location=None):
    import xarray as xr
    
    # If data is archival, perform gridding
    if (file_type.lower() == 'ar2v'):
        # Create grid
        gen_and_save_nexrad_grid(path_to_files, save_location, tracking_var, CONFIG, CONFIG['parallel_processing'], CONFIG['max_cores'])  
        
        # Convert to iris cube and return
        if (tracking_var.lower() == 'dbz'):
            # Open them as netcdf file and return
            nexrad_xarray = xr.open_mfdataset(save_location + "*", coords='all', concat_dim='time', combine='nested')
            
            # Add lat and lon coordinates to dataset
            nexrad_xarray = nexrad_xarray.reflectivity.assign_coords(lat=nexrad_xarray.point_latitude,lon=nexrad_xarray.point_longitude)
            
            nexrad_xarray['z'] = nexrad_xarray.z.assign_attrs({'standard_name': 'altitude'})
            nexrad_xarray['lat'] = nexrad_xarray.lat.assign_attrs({'standard_name': 'latitude'})
            nexrad_xarray['lon'] = nexrad_xarray.lon.assign_attrs({'standard_name': 'longitude'})
            
            return ((nexrad_xarray.to_iris(), nexrad_xarray))
        
        else:
            raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
            return
        
    # If data already grided, just return concated netcdf dataset
    elif (file_type.lower() == 'nc'):
        
        # Convert to iris cube and return
        if (tracking_var.lower() == 'dbz'):
            # Open them as netcdf file and return
            nexrad_xarray = xr.open_mfdataset(path_to_files, coords='all', concat_dim='time', combine='nested')
            
            # Add lat and lon coordinates to dataset
            nexrad_xarray = nexrad_xarray.reflectivity.assign_coords(lat=nexrad_xarray.point_latitude,lon=nexrad_xarray.point_longitude)
            
            nexrad_xarray['z'] = nexrad_xarray.z.assign_attrs({'standard_name': 'altitude'})
            nexrad_xarray['lat'] = nexrad_xarray.lat.assign_attrs({'standard_name': 'latitude'})
            nexrad_xarray['lon'] = nexrad_xarray.lon.assign_attrs({'standard_name': 'longitude'})
            
            return ((nexrad_xarray.to_iris(), nexrad_xarray))
        
        else:
            raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
            return
    
    else:
        raise Exception(f'!=====Invalid File Type. You Entered: {file_type.lower()}=====!')
        return

    
    
"""
Inputs:
    path_to_files: Glob path to input files, either archival or grided netcdf--i.e. "/data/usr/KVNX*_V06.ar2v"
    file_type: ['ar2v', 'nc'] type of input file--either archival or netcdf
    tracking_var: ['dbz'], variable which is going to be used for tracking--reflectivity.
    CONFIG: User configuration file
    save_location: Where to save gridded NEXRAD data to if file_type=='ar2v'
Outputs:
    nexrad_xarray: Xarray dataset containing gridded NEXRAD archival data
"""
def nexrad_load_netcdf(path_to_files, file_type, tracking_var, CONFIG, save_location=None):
    import xarray as xr
    
    # If data is archival, perform gridding
    if (file_type.lower() == 'ar2v'):
        # Create grid
        gen_and_save_nexrad_grid(path_to_files, save_location, tracking_var, CONFIG, CONFIG['parallel_processing'], CONFIG['max_cores'])
        
        # Open them as netcdf file and return
        if (tracking_var.lower() == 'dbz'):
            nexrad_xarray = xr.open_mfdataset(save_location + "*", coords='all', concat_dim='time', combine='nested')
        
            # Add lat and lon coordinates to dataset
            nexrad_xarray = nexrad_xarray.reflectivity.assign_coords(lat=nexrad_xarray.point_latitude,lon=nexrad_xarray.point_longitude)
            
            nexrad_xarray['z'] = nexrad_xarray.z.assign_attrs({'standard_name': 'altitude'})
            nexrad_xarray['lat'] = nexrad_xarray.lat.assign_attrs({'standard_name': 'latitude'})
            nexrad_xarray['lon'] = nexrad_xarray.lon.assign_attrs({'standard_name': 'longitude'})
            
            return(nexrad_xarray)
        
        else:
            raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
            return
        
    # If data already grided, just return concated netcdf dataset
    elif (file_type.lower() == 'nc'):
        
        if (tracking_var.lower() == 'dbz'):
            # Open the dataset
            nexrad_xarray = xr.open_mfdataset(path_to_files, coords='all', concat_dim='time', combine='nested')
        
            # Add lat and lon coordinates to dataset
            nexrad_xarray = nexrad_xarray.reflectivity.assign_coords(lat=nexrad_xarray.point_latitude,lon=nexrad_xarray.point_longitude)
            
            nexrad_xarray['z'] = nexrad_xarray.z.assign_attrs({'standard_name': 'altitude'})
            nexrad_xarray['lat'] = nexrad_xarray.lat.assign_attrs({'standard_name': 'latitude'})
            nexrad_xarray['lon'] = nexrad_xarray.lon.assign_attrs({'standard_name': 'longitude'})
            
            return(nexrad_xarray)
        
        else:
            raise Exception(f'!=====Invalid Tracking Variable. You Entered: {tracking_var.lower()}=====!')
            return
    
    else:
        raise Exception(f'!=====Invalid File Type. You Entered: {file_type.lower()}=====!')
        return
