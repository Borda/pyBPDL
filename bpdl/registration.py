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
import bpdl.dataset_utils as tl_data

NB_THREADS = int(mproc.cpu_count() * .8)

LIST_SDR_PARAMS = ['metric', 'level_iters', 'step_length', 'ss_sigma_factor',
                   'opt_tol', 'inv_iter', 'inv_tol', 'callback']
DEAMONS_PARAMS = dict(
    step_length=1.,
    level_iters=[5, 10],
    inv_iter=2,
    ss_sigma_factor=0.5,
    opt_tol=1.e-2
)


def register_demons_sym_diffeom(img_sense, img_ref, smooth_sigma=1.,
                                params=DEAMONS_PARAMS, verbose=False):
    """ Register the image and reconstruction from atlas
    on the end we smooth the final deformation by a gaussian filter

    :param ndarray img_sense:
    :param ndarray img_ref:
    :param float smooth_sigma:
    :param {} params:
    :param bool verbose: whether show debug time measurements
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
    >>> img_warp, m = register_demons_sym_diffeom(img_sense, img_ref, verbose=True)
    >>> np.round(img_warp.astype(float), 1)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0.3,  0.2,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0.2,  1. ,  1. ,  1. ,  0.8,  0.6,  0.7,  0.8],
           [ 0. ,  0. ,  0.2,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.2,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.2,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.1,  0.7,  0.8,  0.9,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    """
    if img_ref.max() == 0 or img_sense.max() == 0:
        logging.debug('one of the images is zeros')
        return img_sense.copy(), np.zeros(img_ref.shape + (2,))

    sdr_params = {k: params[k] for k in params if k in LIST_SDR_PARAMS}
    sdr = SmoothSymmetricDiffeomorphicRegistration(metric=SSDMetric(img_ref.ndim),
                                                   smooth_sigma=smooth_sigma,
                                                   **sdr_params)
    sdr.verbosity = VerbosityLevels.NONE

    t = time.time()
    sdr.optimize(img_ref, img_sense)
    if verbose:
        logging.debug('demons took: %d s', time.time() - t)

    t = time.time()
    mapping = sdr.moving_to_ref
    # map.forward = filters.gaussian_filter(map.forward, sigma=smooth_sigma)
    mapping.backward = filters.gaussian_filter(mapping.backward,
                                               sigma=smooth_sigma)
    img_warped = mapping.transform_inverse(img_sense, 'linear')

    mapping_atlas = filters.gaussian_filter(sdr.static_to_ref.backward,
                                            sigma=smooth_sigma)
    if verbose:
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
    :return (int, ndarray):
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
    for idx, img_w in tl_data.wrap_execute_parallel(
                wrapper_warp2d_image_deform, list_items, nb_jobs, desc=None):
        list_imgs_wrap[idx] = img_w

    return list_imgs_wrap


def wrapper_regist_demons_images_weights(idx_img_weights, atlas, coef,
                                         params=None):
    """ wrapper for registration of input images to reconstructed as demons

    :param (int, ndarray, ndarray) idx_img_weights:
    :param ndarray atlas:
    :param float coef:
    :param {str: ...} params:
    :return:
    """
    idx, img, w = idx_img_weights
    # extension for using zero as backround
    w_ext = np.asarray([0] + w.tolist())
    img_reconst = w_ext[atlas].astype(atlas.dtype)
    assert atlas.shape == img_reconst.shape, 'im. size of atlas and image'

    if params is None:
        params = DEAMONS_PARAMS
    img_warp, deform = register_demons_sym_diffeom(img, img_reconst,
                                                   coef, params)

    return idx, img_warp, deform


def register_images_to_atlas_demons(list_images, atlas, list_weights, coef=1.,
                                    params=None, nb_jobs=NB_THREADS):
    """ register whole set of images to estimated atlas and weights
    IDEA: think about parallel registration per sets as for loading images

    :param [ndarray] list_images:
    :param ndarray atlas:
    :param ndarray list_weights:
    :param float coef:
    :param {str: ...} params:
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

    _wrapper_register = partial(wrapper_regist_demons_images_weights,
                                atlas=atlas, coef=coef, params=params)
    for idx, img_w, deform in tl_data.wrap_execute_parallel(
                            _wrapper_register, list_items, nb_jobs, desc=None):
        list_imgs_wrap[idx] = img_w
        list_deform[idx] = deform

    return list_imgs_wrap, list_deform


class SmoothSymmetricDiffeomorphicRegistration(SymmetricDiffeomorphicRegistration):

    def __init__(self, metric, smooth_sigma=0.5, **kwargs):
        super(SmoothSymmetricDiffeomorphicRegistration, self).__init__(metric,
                                                                       **kwargs)
        self.smooth_sigma = smooth_sigma

    def update(self, current_displacement, new_displacement,
               disp_world2grid, time_scaling):
        """Composition of the current displacement field with the given field

        Interpolates new displacement at the locations defined by
        current_displacement. Equivalently, computes the composition C of the
        given displacement fields as C(x) = B(A(x)), where A is
        current_displacement and B is new_displacement. This function is
        intended to be used with deformation fields of the same sampling
        (e.g. to be called by a registration algorithm).

        Parameters
        ----------
        current_displacement : array, shape (R', C', 2) or (S', R', C', 3)
            the displacement field defining where to interpolate
            new_displacement
        new_displacement : array, shape (R, C, 2) or (S, R, C, 3)
            the displacement field to be warped by current_displacement
        disp_world2grid : array, shape (dim+1, dim+1)
            the space-to-grid transform associated with the displacements'
            grid (we assume that both displacements are discretized over the
            same grid)
        time_scaling : float
            scaling factor applied to d2. The effect may be interpreted as
            moving d1 displacements along a factor (`time_scaling`) of d2.

        Returns
        -------
        updated : array, shape (the same as new_displacement)
            the warped displacement field
        mean_norm : the mean norm of all vectors in current_displacement
        """
        sq_field = np.sum((np.array(current_displacement) ** 2), -1)
        mean_norm = np.sqrt(sq_field).mean()

        # smoothing the forward/backward step
        new_displacement = filters.gaussian_filter(new_displacement,
                                                   sigma=self.smooth_sigma)

        # We assume that both displacement fields have the same
        # grid2world transform, which implies premult_index=Identity
        # and premult_disp is the world2grid transform associated with
        # the displacements' grid
        self.compose(current_displacement, new_displacement, None,
                     disp_world2grid, time_scaling, current_displacement)

        return np.array(current_displacement), np.array(mean_norm)
