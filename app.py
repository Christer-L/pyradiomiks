from arkitekt import register
import time
from mikro.api.schema import RepresentationFragment, from_xarray
import numpy as np
from skimage.filters import gaussian

@register
def gaussian(image: RepresentationFragment, sigma: float) -> RepresentationFragment:
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



