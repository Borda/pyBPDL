"""
tools for registering images to reconstructed image using Atlas

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import time
import logging

import numpy as np
from scipy import interpolate
from scipy.ndimage import filters
from dipy.align import VerbosityLevels
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric

DEAMONS_PARAMS = dict(
    step_length=1.,
    level_iters=[5, 10],
    inv_iter=2,
    ss_sigma_factor=0.5,
    opt_tol=1.e-2
)

# todo, think about parallel registration per sets as for loading images


def register_demons_sym_diffeom(img_sense, img_ref, smooth_sigma=1.,
                                params=DEAMONS_PARAMS):
    """ Register the image and reconstruction from atlas
    on the end we smooth the final deformation by a gaussian filter

    :param ndarray img_sense:
    :param ndarray img_ref:
    :param float smooth_sigma:
    :param {} params:
    :return:

    >>> img_ref = np.zeros((10, 10), dtype=int)
    >>> img_ref[2:6, 1:7] = 1
    >>> img_ref[6:9, 5:10] = 1
    >>> img_ref
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> img_sense = np.zeros(img_ref.shape, dtype=float)
    >>> img_sense[4:9, 3:10] = 1
    >>> img_sense.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> img_warp, map = register_demons_sym_diffeom(img_sense, img_ref)
    >>> np.round(img_warp.astype(float), 1)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0.9,  1. ,  0.9,  0.7,  0.3,  0.1,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.9,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.7,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.2,  0.6,  0.6,  0.7,  0.8,  0.9,  0.9,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    """
    sdr = SymmetricDiffeomorphicRegistration(metric=SSDMetric(img_ref.ndim),
                                             step_length=params['step_length'],
                                             level_iters=params['level_iters'],
                                             inv_iter=params['inv_iter'],
                                             ss_sigma_factor=params['ss_sigma_factor'],
                                             opt_tol=params['opt_tol'])
    sdr.verbosity = VerbosityLevels.NONE

    t = time.time()
    sdr.optimize(img_ref, img_sense)
    logging.debug('demons took: %d s', time.time() - t)

    t = time.time()
    map = sdr.moving_to_ref
    # map.forward = filters.gaussian_filter(map.forward, sigma=smooth_sigma)
    map.backward = filters.gaussian_filter(map.backward, sigma=smooth_sigma)
    img_warped = map.transform_inverse(img_sense, 'linear')

    mapping_atlas = filters.gaussian_filter(sdr.static_to_ref.backward,
                                            sigma=smooth_sigma)
    logging.debug('smoothing and warping took: %d s', time.time() - t)

    return img_warped, mapping_atlas


def warp_atlas_domain_to_image(img, deform, method='linear'):
    """ warping reconstructed image using atlas and weight
    to the expected image image domain

    :param ndarray img:
    :param ndarray deform:
    :return ndarray:

    >>> img = np.zeros((8, 12), dtype=int)
    >>> img[2:6, 3:9] = 1
    >>> deform = np.ones(img.shape + (2,))
    >>> warp_atlas_domain_to_image(img, deform)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    """
    grid_x, grid_y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    deform_x = deform[..., 0]
    deform_y = deform[..., 1]

    grid_old = (grid_x, grid_y)
    # points_old = np.array([grid_old[0].ravel(), grid_old[1].ravel()]).T
    grid_new = (grid_x + deform_x, grid_y + deform_y)
    points_new = np.array([grid_new[0].ravel(), grid_new[1].ravel()]).T

    img_warped = interpolate.griddata(points_new, img.ravel(), grid_old,
                                      method=method, fill_value=0)
    img_warped.astype(img.dtype)
    return img_warped
