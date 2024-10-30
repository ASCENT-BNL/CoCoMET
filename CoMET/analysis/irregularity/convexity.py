
#Need to add
#Allow user to choose whether to use 2d or 3d segmentation


import numpy as np
import scipy.spatial as sp
from copy import deepcopy
from skimage import measure    
from .projection_calc_3d import point_projection
from tqdm import tqdm
import pandas as pd

def convexity(
        analysis_dict : dict, 
        surface_area_df : pd.DataFrame
) -> pd.DataFrame:

    """
        Parameters
        ----------
        analysis_dict : DICT
            Dictionary output from analysis_object.return_analysis_dictionary(). Contains UDAF_tracks dataframe and segmentation xarray dataset.
        Raises
        ------
        ValueError
            No segmentation available in input dictionary.

        Returns
        -------
        convexity : pandas dataframe
            dataframe with columns of feature id and convexity, where convexity is a value between 0 and 1

    """

    #returns a value between 0 and 1 which represents the convexity of the cell. Lower convexity represents more irregular edges
    Tracks = analysis_dict['UDAF_tracks']
    TracksAndPerim = Tracks.join(surface_area_df)
    # If 3D segmentation is available, use that 
    if analysis_dict["UDAF_segmentation_3d"] is not None:

        footprint = analysis_dict['UDAF_segmentation_3d'].Feature_Segmentation
        convexities = []
        feature_ids = []
        frame_numbers = []
        cell_ids = []
        frame_groups = TracksAndPerim.groupby('frame')
        for frame_number, frame_df in tqdm(frame_groups, 
                                            desc='=====Calculating Cell Convexities=====', 
                                            total=len(frame_groups)):
            
            final_mask = deepcopy(footprint[frame_number])
            data = final_mask.values
            for row in frame_df.itertuples(): #using itertuples because internet told me it's faster
                id_num = row[5]
                cell_id = row[6]
                i = row[0]
                perim = row[-1]

                feature_ids.append(id_num)
                frame_numbers.append(frame_number)
                cell_ids.append(cell_id)

                points = np.asarray(list(zip(*np.where(data == id_num))))
                if len(points) == 0:
                    convexities.append(np.NaN)
                    continue

                points_proj = []
                for point in points: #convert to projection coordinates
                    point_proj = np.array(point_projection(footprint, point))
                    points_proj.append(point_proj/1000)
                points_proj = np.asarray(points_proj)

                try:
                    convex_hull = sp.ConvexHull(points=points_proj) #generate convex hull using segmentation coords
                except:
                    convexities.append(np.NaN)
                    continue

                convex_perim = convex_hull.area #extract surface area

                convexity = convex_perim/perim
                if convexity > 1:
                    convexities.append(1)
                    continue
                convexities.append(convexity)
        convexity_df = pd.DataFrame(
            data = {'frame': frame_numbers, 'feature_id': feature_ids, 'cell_id': cell_ids, 'convexity': convexities}
        )
        return convexity_df
        
    
    #handle 2D case
    elif analysis_dict["UDAF_segmentation_2d"] is not None:

        footprint_data = analysis_dict['UDAF_segmentation_2d'].Feature_Segmentation
        frame_groups = Tracks.groupby('frame')

        convexities = []
        feature_ids = []
        frame_numbers = []
        cell_ids = []
        for frame_number, frame_df in tqdm(frame_groups, 
                                           desc='=====Calculating Cell Convexities=====', 
                                           total=len(frame_groups)):
            final_mask = deepcopy(footprint_data[frame_number])
            data = final_mask.values
            for row in frame_df.itertuples():
                id_num = row[5]
                cell_id = row[6]
                perim = row[-1]

                feature_ids.append(id_num)
                frame_numbers.append(frame_number)
                cell_ids.append(cell_id)

                mask = data == id_num

                points = np.asarray(list(zip(*np.where(data == id_num)))) #coordinates to every point apart of the specified cell

                try:
                    convex_hull = sp.ConvexHull(points=points) #generate convex hull using segmentation coords
                    convex_perim = convex_hull.area #extract perimeter
                except:
                    convexities.append(np.NaN) #this is mainly for cells that are only along a plane, the convex hull doesn't work for them
                    continue


                contours = measure.find_contours(mask, 0.5) #create a contour around the 2d shape

                #this block is used to calculate 2d perimeter but you can replace it with another calculation if you want
                prev_vertex = [-1, -1]
                perim = 0
                for vertex in contours[0]: #loop through every contour coord to add up the total perimeter
                    if -1 in prev_vertex:
                        prev_vertex = vertex
                        continue
                    dist = np.linalg.norm(vertex - prev_vertex) #distance between current vertex and the previous
                    perim += dist
                    prev_vertex = vertex
                final_dist = np.linalg.norm(contours[0][0] - contours[0][-1]) #add final part of perimeter
                perim += final_dist


                convexity = convex_perim/perim
                convexities.append(convexity)

        convexity_df = pd.Dataframe(
            data = {'frame': frame_numbers, 'feature_id': feature_ids, 'cell_id': cell_ids, 'convexity': convexities}
        )
        return convexity_df
    else:
        raise ValueError("!=====Missing Segmentation Input=====!")

