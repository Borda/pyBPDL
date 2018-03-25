"""
tools for registering images to reconstructed image using Atlas

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import time
import logging
import multiprocessing as mproc
from functools import partial

import numpy as np
from scipy import interpolate
from scipy.ndimage import filters
from dipy.align import VerbosityLevels
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric

import bpdl.pattern_atlas as ptn_atlas

NB_THREADS = int(mproc.cpu_count() * .8)
DEAMONS_PARAMS = dict(
    step_length=1.,
    level_iters=[5, 10],
    inv_iter=2,
    ss_sigma_factor=0.5,
    opt_tol=1.e-2
)


def register_demons_sym_diffeom(img_sense, img_ref, smooth_sigma=1.,
                                params=DEAMONS_PARAMS):
    """ Register the image and reconstruction from atlas
    on the end we smooth the final deformation by a gaussian filter

    :param ndarray img_sense:
    :param ndarray img_ref:
    :param float smooth_sigma:
    :param {} params:
    :return (ndarray, ndarray):

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
    if img_ref.max() == 0 or img_sense.max() == 0:
        logging.debug('one of the images is zeros')
        return img_sense.copy(), np.zeros(img_ref.shape + (2,))

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


def warp2d_apply_deform_field(img, deform, method='linear'):
    """ warping reconstructed image using atlas and weight
    to the expected image image domain

    :param ndarray img:
    :param ndarray deform:
    :return ndarray:

    >>> img = np.zeros((8, 12), dtype=int)
    >>> img[2:6, 3:9] = 1
    >>> deform = np.ones(img.shape + (2,))
    >>> deform[:, :, 1] *= -2
    >>> warp2d_apply_deform_field(img, deform)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    """
    assert img.ndim == 2, 'expected only 2D image'
    assert deform.ndim == 3, 'expected only 2D deformation'
    assert img.shape == deform.shape[:-1], \
        'image %s and deform %s size should match' \
        % (repr(img.shape), repr(deform.shape))
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


def wrapper_warp2d_image_deform(idx_img_deform, method='linear'):
    """ wrapper for registration of input images to reconstructed as demons

    :param (int, ndarray, ndarray) idx_img_deform:
    :return:
    """
    idx, img, deform = idx_img_deform
    img_warped = warp2d_apply_deform_field(img, deform, method=method)
    return idx, img_warped


def warp2d_images_deformations(list_images, list_deforms, nb_jobs=NB_THREADS):
    """ deforme whole set of images to expected image domain

    :param [ndarray] list_images:
    :param ndarray list_deforms:
    :param int nb_jobs:
    :return: [ndarray]

    >>> img = np.zeros((6, 9), dtype=int)
    >>> img[:3, 1:5] = 1
    >>> deform = np.ones(img.shape + (2,))
    >>> imgs = warp2d_images_deformations([img], [deform])
    >>> imgs  # doctest: +NORMALIZE_WHITESPACE
    [array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])]
    """
    assert len(list_images) == len(list_deforms), \
        'number of images (%i) and deformations (%i) have to match' \
        % (len(list_images), len(list_deforms))
    list_deforms = list(list_deforms)

    list_imgs_wrap = [None] * len(list_images)
    list_items = zip(range(len(list_images)), list_images, list_deforms)
    if nb_jobs < 1:
        nb_jobs = 1

    mproc_pool = mproc.Pool(nb_jobs)
    for idx, img_w in mproc_pool.map(wrapper_warp2d_image_deform, list_items):
        list_imgs_wrap[idx] = img_w
    mproc_pool.close()
    mproc_pool.join()

    return list_imgs_wrap


def wrapper_regist_demons_images_weights(idx_img_weights, atlas, coef,
                                         params=None):
    """ wrapper for registration of input images to reconstructed as demons

    :param (int, ndarray, ndarray) idx_img_weights:
    :param ndarray atlas:
    :param float coef:
    :param {str: } params:
    :return:
    """
    idx, img, w = idx_img_weights
    # extension for using zero as backround
    w_ext = np.asarray([0] + w.tolist())
    img_reconst = w_ext[atlas].astype(atlas.dtype)
    assert atlas.shape == img_reconst.shape, 'im. size of atlas and image'

    if params is None:
        params = DEAMONS_PARAMS
    img_warp, deform = register_demons_sym_diffeom(img, img_reconst, coef, params)

    return idx, img_warp, deform


def register_images_to_atlas_demons(list_images, atlas, list_weights, coef=1,
                                    params=None, nb_jobs=NB_THREADS):
    """ register whole set of images to estimated atlas and weights
    IDEA: think about parallel registration per sets as for loading images

    :param [ndarray] list_images:
    :param ndarray atlas:
    :param ndarray list_weights:
    :param float coef:
    :param {str:} params:
    :param int nb_jobs:
    :return: [ndarray], [ndarray]

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> w_bins = np.array([[0, 0], [0, 1], [1, 1]], dtype=bool)
    >>> imgs = ptn_atlas.reconstruct_samples(atlas, w_bins)
    >>> deform = np.ones(atlas.shape + (2,))
    >>> deforms = [deform * 0, deform * -2, deform * 0]
    >>> imgs = warp2d_images_deformations(imgs, deforms)
    >>> imgs[1].astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> _, _ = register_images_to_atlas_demons(imgs, atlas, w_bins, nb_jobs=1)
    >>> imgs_w, deforms = register_images_to_atlas_demons(imgs, atlas, w_bins,
    ...                                                   coef=0.1, nb_jobs=2)
    >>> np.sum(imgs_w[0])
    0.0
    >>> imgs_w[1].astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> np.mean(deforms[1][:, :, 0])  # doctest: +ELLIPSIS
    -0.9...
    >>> np.mean(deforms[1][:, :, 1])  # doctest: +ELLIPSIS
    -0.9...
    >>> np.sum(abs(imgs[2] - imgs_w[2]))
    0.0
    """
    assert len(list_images) == len(list_weights), \
        'number of images (%i) and weights (%i) have to match' \
        % (len(list_images), len(list_weights))
    atlas = np.asarray(atlas, dtype=int)
    list_weights = list(list_weights)

    list_imgs_wrap = [None] * len(list_images)
    list_deform = [None] * len(list_weights)
    list_items = zip(range(len(list_images)), list_images, list_weights)

    if nb_jobs > 1:
        wrapper_register = partial(wrapper_regist_demons_images_weights,
                                   atlas=atlas, coef=coef, params=params)
        mproc_pool = mproc.Pool(nb_jobs)
        for idx, img_w, deform in mproc_pool.map(wrapper_register, list_items):
            list_imgs_wrap[idx] = img_w
            list_deform[idx] = deform
        mproc_pool.close()
        mproc_pool.join()
    else:
        for item in list_items:
            idx, img_w, deform = wrapper_regist_demons_images_weights(
                item, atlas, coef, params)
            list_imgs_wrap[idx] = img_w
            list_deform[idx] = deform

    return list_imgs_wrap, list_deform
