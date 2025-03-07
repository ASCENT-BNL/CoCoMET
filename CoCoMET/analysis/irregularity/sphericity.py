import numpy as np
import pandas as pd
from tqdm import tqdm


def sphericity(
    surface_area_df: pd.DataFrame, volume_df: pd.DataFrame, **args: dict
) -> pd.DataFrame:
    """

    Parameters
    ----------
    surface_area_df : pd.DataFrame
        A pandas dataframe with the 2D surface area values of each cell.
    volume_df : pd.DataFrame
        A pandas dataframe with the volumes of each cell.
    **args : dict
        Throw away variables.

    Returns
    -------
    pandas.core.frame.DataFrame
        A pandas dataframe with the following rows: frame, feature_id, cell_id, sphericity.
    """

    sphericities = []
    frames = []
    feature_ids = []
    cell_ids = []
    for row in tqdm(
        surface_area_df.itertuples(),
        desc="=====Calculating Cell Sphericities=====",
        total=len(surface_area_df),
    ):

        i = row[0]
        frames.append(row[1])
        feature_ids.append(row[2])
        cell_ids.append(row[3])

        surface_area = row[
            4
        ]  # this is assuming both the surface area and volume are in units of km
        volume = volume_df.iloc[i]

        if surface_area == 0 or np.isnan(surface_area):  # check for 0 or NaN values
            sphericities.append(np.NaN)
            continue

        if volume == 0 or np.isnan(volume):  # check for 0 or NaN values
            sphericities.append(np.NaN)
            continue

        sphericity_val = ((np.pi) ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
        if sphericity_val > 1:
            sphericities.append(1)
            continue
        sphericities.append(sphericity_val)

    sphericity_df = pd.DataFrame(
        {
            "frame": frames,
            "feature_id": feature_ids,
            "cell_id": cell_ids,
            "sphericity": sphericities,
        }
    )
    return sphericity_df
