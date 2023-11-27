from arkitekt import register
import time
from mikro.api.schema import RepresentationFragment, from_xarray, from_df, TableFragment
import numpy as np
from skimage.filters import gaussian
import pandas as pd


@register
def gaussian(image: RepresentationFragment) -> RepresentationFragment:
    """A simple gaussian filter

    This function applies gaussian filter to the imags.

    Parameters
    ----------
    image : RepresentationFragment
        The input image

    sigma : Gaussian filter radius

    Returns
    -------
    RepresentationFragment
        The blurred image
    """
    image_gaussian = gaussian(np.array(image.data), sigma=sigma)
    image_gaussian = np.array(image_gaussian)
    return from_xarray(
        image_gaussian, name="Gaussian_" + image.name, origins=[image]
    )


@register
def get_features(mask: RepresentationFragment, image: RepresentationFragment) -> TableFragment:
    """Feature extractor    

    Parameters
    ----------
    image : RepresentationFragment
        The input image

    sigma : Gaussian filter radius

    Returns
    -------
    RepresentationFragment
        The blurred image
    """

    df_list = []
    unique_values, counts = np.unique(mask.data, return_counts=True)

    for value, count in zip(unique_values, counts):
        df_list.append([value, count])

    df = pd.DataFrame(df_list, columns=["unique_value", "count"])
    df_test = pd.DataFrame([[4,5], [4,5], [3, 2]], columns=["test1", "test2"])
    print(df_test)
    return from_df(
        df_test, name="Counts_" + mask.name
    )

