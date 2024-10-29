'''
This file contains functions for dealing with the dynamic vertical scaling when working with 3d data
Some functions in here rely on others, so its best to import the entire file
'''



def calc_3d_perim(footprint_data, frame, feature_id):
    """
    Inputs:
        footprint_data: UDAF_segmentation_3d.Feature_Segmentation
        frame: frame number of the relevant cell
        feature_id: feature_id number of the relevant cell
    Outputs:
        perims: A dictionary with keys equal to what the surface area is touching (-1 for empty space, feature_id of other cells) and values equal to the amount of surface area shared with that object
    """

    from copy import deepcopy
    from dask.array import isin 
    import numpy as np

    final_mask = deepcopy(footprint_data[frame, :, :, :])
    feature_mask = deepcopy(final_mask)
    feature_mask.values[~isin(final_mask.values, feature_id)] = -1 
    # tilde inverts the output, so this creates a boolean array with True where feature_id is, then inverts that to 
    #set all non-feature_id values to -1


    # If cell has no area, skip
    if np.all(feature_mask.values == -1):
        return -1

    # see if this cell is touching any other cells
    feature_data = np.asarray(final_mask.values)

    z_dim_sizes = np.diff(footprint_data.altitude)
    x_dim_sizes = np.diff(footprint_data.projection_x_coordinate)
    y_dim_sizes = np.diff(footprint_data.projection_y_coordinate)


    # These are one cell too small due to how diff works, so infer last cell size using the same cell size as the previous cell
    x_dim_sizes = np.append(x_dim_sizes, x_dim_sizes[-1])
    y_dim_sizes = np.append(y_dim_sizes, y_dim_sizes[-1])
    z_dim_sizes = np.append(z_dim_sizes, z_dim_sizes[-1])

    x_dim_sizes = x_dim_sizes/1000 #convert to km
    y_dim_sizes = y_dim_sizes/1000
    z_dim_sizes = z_dim_sizes/1000

    perims = {} #this is to keep track of the amount of the perimeter that is shared with another cell
    #key is feature id of other cell, value is the shared perimeter

    for nz, ny, nx in list(zip(*np.where(feature_data == feature_id))):
        #print(nx, ny, nz)

        # Find if location is on edge of cell (i.e. any touching cells not)
        neighboring_cells = []

        for mx in (nx - 1, nx + 1):

            if mx in range(feature_data.shape[2]):
                t = feature_data[nz, ny, mx]
            else: 
                t = -1
            if t != feature_id:
                perims[t] = perims.get(t, 0) + y_dim_sizes[ny]*z_dim_sizes[nz]

        for my in (ny - 1, ny + 1):

            if my in range(feature_data.shape[1]):
                t = feature_data[nz, my, nx]
            else: 
                t = -1
            if t != feature_id:
                perims[t] = perims.get(t, 0) + x_dim_sizes[nx]*z_dim_sizes[nz]

        for mz in (nz - 1, nz + 1):

            if mz in range(feature_data.shape[0]):
                t = feature_data[mz, ny, nx]
            else: 
                t = -1
            if t != feature_id:
                perims[t] = perims.get(t, 0) + x_dim_sizes[nx]*y_dim_sizes[ny]
    return perims



def calculate_volume(mask, points):
    """
    Inputs:
        mask: UDAF_segmentation_3d.Feature_Segmentation
        points: list of (z, y, x) grid points that make up the cell
    Outputs:
        volume: volume of cell in km^3
    """

    import numpy as np

    # We first calculate the area of each individual cell
    # First get the size of each dimension
    x_dim_sizes = np.diff(mask.projection_x_coordinate)
    y_dim_sizes = np.diff(mask.projection_y_coordinate)
    z_dim_sizes = np.diff(mask.altitude)

    # These are one cell too small due to how diff works, so infer last cell size using the same cell size as the previous cell
    # x_dim_sizes.append(x_dim_sizes[-1])
    x_dim_sizes = np.append(x_dim_sizes, x_dim_sizes[-1])
    y_dim_sizes = np.append(y_dim_sizes, y_dim_sizes[-1])
    z_dim_sizes = np.append(z_dim_sizes, z_dim_sizes[-1])

    # use Einstein sum notation to get volume of cells
    #cell_volumes = np.einsum("i,j,k->ijk", z_dim_sizes, y_dim_sizes, x_dim_sizes)

    total_vol = 0
    for z, y, x in points:
        total_vol += x_dim_sizes[x] * y_dim_sizes[y] * z_dim_sizes[z]

    return total_vol/(10**9)



def create_gridded_bounds(footprint_data, center, radius):
    '''
    Inputs:
        footprint_data: UDAF_Segmentatio_3d. Must have altitude, x projection, and y projection as attributes
        center: center of the cube (usually feature position given by Tracks/Feature df) in projection coordinates (z, y, x)
        radius: half of side length of the desired cube in km
    Outputs:
        tuple of the minimum and maximum grid points of z, y, x as tuples (min,max)
    '''

    altitudes = footprint_data.altitude #get the projection values at every grid coordinate point
    yprojs = footprint_data.projection_y_coordinate
    xprojs = footprint_data.projection_x_coordinate

    altitude = center[0] 
    proj_y = center[1]
    proj_x = center[2]

    max_alt = altitude + (radius*1000) #get the bounds of the box in terms of projection coords
    min_alt = altitude - (radius*1000)
    max_y_proj = proj_y + (radius*1000)
    min_y_proj = proj_y - (radius*1000)
    max_x_proj = proj_x + (radius*1000)
    min_x_proj = proj_x - (radius*1000)

    #Convert to grid points
    max_z = find_nearest(altitudes, max_alt) #convert bounds back to grid coords
    min_z = find_nearest(altitudes, min_alt)
    max_y = find_nearest(yprojs, max_y_proj)
    min_y = find_nearest(yprojs, min_y_proj)
    max_x = find_nearest(xprojs, max_x_proj)
    min_x = find_nearest(xprojs, min_x_proj)

    return ((min_z, max_z), (min_y, max_y), (min_x, max_x))



def find_nearest(array, pivot):
    # Calculate nearest item in list to given pivot
    # This is used when converting projection coordinates into grid coordinate indices
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - pivot)).argmin()
    return idx



def create_scaled_3d_mesh(footprint_data, frame, feature_id, variable_field=None, step_size=1):
    '''
    Inputs:
        footprint_data: Xarray. UDAF_segmentation_3d.Feature_Segmentation
        variable_field: Xarray. Xarray of variable that was used for segmentation
        frame: Int. frame of the specified cell
        feature_id: Int. feature_id of the specified cell
        step_size: Int. Default is 1. Step size input to marching cubes algorithm. Larger number means coarser mesh. 
    Outputs:
        verts: (V, 3) Array. Vertices of the mesh in projection coordinates (altitude, projection_y, projection_x) meters
        faces: (F, 3) Array. List of tuples describing the faces of the mesh. Each entry of the tuple is the index of a vertex in verts, and 3 indices in a tuple make up one face. 
        values: (V,) Array. Maximum value of the data in the local region near each vertex. Usually used for visualization
    '''
    import numpy as np
    from skimage.measure import marching_cubes
    from copy import deepcopy

    final_mask = deepcopy(footprint_data[frame])
    data = final_mask.values
    mask = data == feature_id
    thresh = 0.5
    if type(variable_field) != type(None):
        mask = variable_field[frame, :, :, :].where(mask, -1).values #create a mask with the variable value where the segmentation is and -1 everywhere else
        thresh = 0            


    verts, faces, normals, values = marching_cubes(mask, thresh, step_size=step_size) 

    verts_proj = []
    for vertex in verts: #convert each vertex to projection coordinates
        vertex_projection = point_projection(footprint_data, vertex)
        verts_proj.append(vertex_projection/1000)
    verts_proj = np.array(verts_proj)
    return verts_proj, faces, values



def point_projection(footprint_data, point):
    '''
    Inputs:
        footprint_data: Xarray. UDAF_segmentation_3d.Feature_Segmentation
        point: Array-like. 3d point to be converted from grid coordinates to projection coordinates in form (z, y, x),
        Or 2d point in (x, y)
    Outputs:
        projection_point: List. Projection coordinates of input point in form [z, y, x]
    '''

    import numpy as np

    projection_x = footprint_data.projection_x_coordinate.values #get projection values for every grid coordinate
    projection_y = footprint_data.projection_y_coordinate.values
    altitudes = footprint_data.altitude.values

    if len(point) == 2:
        x = point[0]
        y = point[1]
        coord_list = [[x, projection_x], [y, projection_y]]
    elif len(point) == 3:
        x = point[2]
        y = point[1]
        z = point[0]
        coord_list = [[z, altitudes], [y, projection_y], [x, projection_x]]
    else:
        raise ValueError('Point should either be 2 or 3 dimensional')

    projection_point = [] #output point [z,y,x] or [x,y]

    for i, projection in coord_list:
        #for each coord z,y,x interpolate linearly between the projection bounds to find the projection value at the point
        upper_bound = projection[int(np.ceil(i))]
        lower_bound = projection[int(np.floor(i))]
        difference = upper_bound - lower_bound

        fraction = i - np.trunc(i) #fractional component of the input coord (i.e. 1.4 -> 0.4)

        fract_diff = fraction * difference 

        projection_i = lower_bound + fract_diff
        projection_point.append(projection_i)

    return np.array(projection_point)

def distance_projection(footprint_data, point1, point2):
    import numpy as np
    proj1 = point_projection(footprint_data, point1)
    proj2 = point_projection(footprint_data, point2)

    z1, y1, x1 = proj1
    z2, y2, x2 = proj2

    dist = np.sqrt((z2-z1)**2 + (y2-y1)**2 + (x2-x1)**2)

    return dist