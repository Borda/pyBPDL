"""
tools for registering images to reconstructed image using Atlas

SEE:
* http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/
* https://bic-berkeley.github.io/psych-214-fall-2016/dipy_registration.html

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import time
import logging
import multiprocessing as mproc
from functools import partial

import numpy as np
from scipy import ndimage, interpolate
# from scipy.ndimage import filters
from dipy.align import VerbosityLevels
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration, DiffeomorphicMap
from dipy.align.metrics import SSDMetric

from bpdl.utilities import wrap_execute_sequence

NB_THREADS = int(mproc.cpu_count() * .8)

LIST_SDR_PARAMS = ('metric', 'level_iters', 'step_length', 'ss_sigma_factor',
                   'opt_tol', 'inv_iter', 'inv_tol', 'callback')
DIPY_DEAMONS_PARAMS = dict(
    step_length=0.1,
    level_iters=[30, 50],
    inv_iter=20,
    ss_sigma_factor=0.1,
    opt_tol=1.e-2,
)


def register_demons_sym_diffeom(img_sense, img_ref, smooth_sigma=1.,
                                params=DIPY_DEAMONS_PARAMS, inverse=False,
                                verbose=False):
    """ Register the image and reconstruction from atlas
    on the end we smooth the final deformation by a gaussian filter

    :param ndarray img_sense:
    :param ndarray img_ref:
    :param float smooth_sigma:
    :param {} params:
    :param bool verbose: whether show debug time measurements
    :return (ndarray, ndarray):

    >>> np.random.seed(0)
    >>> img_ref = np.zeros((10, 10), dtype=int)
    >>> img_ref[2:6, 1:7] = 1
    >>> img_ref[5:9, 4:10] = 1
    >>> img_ref
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> from skimage.morphology import erosion, dilation
    >>> img_ref_fuz = np.zeros((10, 10), dtype=float)
    >>> img_ref_fuz[dilation(img_ref, np.ones((3, 3))) == 1] = 0.1
    >>> img_ref_fuz[img_ref == 1] = 0.5
    >>> img_ref_fuz[erosion(img_ref, np.ones((3, 3))) == 1] = 1.0
    >>> img_ref_fuz
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0. ,  0. ],
           [ 0.1,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.1,  0. ,  0. ],
           [ 0.1,  0.5,  1. ,  1. ,  1. ,  1. ,  0.5,  0.1,  0. ,  0. ],
           [ 0.1,  0.5,  1. ,  1. ,  1. ,  1. ,  0.5,  0.1,  0.1,  0.1],
           [ 0.1,  0.5,  0.5,  0.5,  0.5,  1. ,  0.5,  0.5,  0.5,  0.5],
           [ 0.1,  0.1,  0.1,  0.1,  0.5,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0. ,  0.1,  0.5,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0. ,  0.1,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0. ,  0. ,  0. ,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1]])
    >>> d_deform = register_demons_sym_diffeom(img_ref_fuz, img_ref,
    ...                         smooth_sigma=1.5, inverse=True, verbose=True)
    >>> img_warp = warp2d_transform_image(img_ref, d_deform, method='nearest',
    ...                                   inverse=True)
    >>> np.round(img_warp.astype(float), 1)
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    >>> np.round(img_warp - img_sense, 1)  # doctest: +SKIP
    >>> img_sense = np.zeros(img_ref.shape, dtype=int)
    >>> img_sense[4:9, 3:10] = 1
    >>> img_sense
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
    >>> d_deform = register_demons_sym_diffeom(img_sense, img_ref, smooth_sigma=0.)
    >>> img_warp = warp2d_transform_image(img_sense, d_deform)
    >>> np.round(img_warp.astype(float), 1)  # doctest: +SKIP
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0.3,  0.5,  0.3,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  1. ,  1. ,  1. ,  0.8,  0.4,  0.1,  0. ,  0. ],
           [ 0. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ],
           [ 0. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0.2,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.6,  0.9,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0.2,  0.4,  0.5,  0.8,  1. ,  1. ,  1. ,  1. ],
           [ 0. ,  0. ,  0. ,  0.2,  0.2,  0.3,  0.4,  0.6,  0.7,  1. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> np.round(img_warp - img_sense, 1)  # doctest: +SKIP
    """
    if img_ref.max() == 0 or img_sense.max() == 0:
        logging.debug('skip image registration (demons): max values for '
                      'RECONST=%d and SENSE=%d', img_ref.max(), img_sense.max())
        return {'mapping': None, 'mapping-inv': None, 'package': 'dipy'}

    sdr_params = {k: params[k] for k in params if k in LIST_SDR_PARAMS}
    sdr = SmoothSymmetricDiffeomorphicRegistration(metric=SSDMetric(img_ref.ndim),
                                                   smooth_sigma=smooth_sigma,
                                                   **sdr_params)
    sdr.verbosity = VerbosityLevels.NONE

    t = time.time()
    mapping = sdr.optimize(img_ref.astype(float), img_sense.astype(float))
    if verbose:
        logging.debug('demons took: %d s', time.time() - t)

    mapping.forward = smooth_deform_field(mapping.forward, sigma=smooth_sigma)
    mapping.backward = smooth_deform_field(mapping.backward, sigma=smooth_sigma)

    # img_warped = mapping.transform(img_moving, 'linear')

    # mapping_inv = sdr.moving_to_ref
    if inverse:
        mapping_inv = DiffeomorphicMap(img_ref.ndim,
                                       img_ref.shape, None,
                                       img_ref.shape, None,
                                       img_ref.shape, None,
                                       None)
        mapping_inv.forward = smooth_deform_field(sdr.moving_to_ref.forward,
                                                  sigma=smooth_sigma)
        mapping_inv.backward = smooth_deform_field(sdr.moving_to_ref.backward,
                                                   sigma=smooth_sigma)
    else:
        mapping_inv = None

    if verbose:
        logging.debug('smoothing and warping took: %d s', time.time() - t)

    dict_deform = {
        'mapping': mapping,
        'mapping-inv': mapping_inv,
        'package': 'dipy'
    }

    return dict_deform


def smooth_deform_field(field, sigma):
    """

    :param field:
    :param sigma:
    :return:

    >>> np.random.seed(0)
    >>> field = np.random.random((10, 5, 1))
    >>> np.std(field)   # doctest: +ELLIPSIS
    0.27...
    >>> field_smooth = smooth_deform_field(field, 0.5)
    >>> np.std(field_smooth)   # doctest: +ELLIPSIS
    0.17...
    """
    if sigma <= 0:
        return np.array(field)
    field_smooth = np.empty(field.shape, dtype=field.dtype)

    # TODO: use different smoothing which would be fast also for large regul.

    for i in range(field.shape[-1]):
        field_smooth[..., i] = ndimage.gaussian_filter(field[..., i],
                                                       sigma=sigma,
                                                       order=0, mode='constant')
    return field_smooth


def warp2d_transform_image(img, dict_deform, method='linear', inverse=False):
    img_warped = img.copy()
    if dict_deform['package'] == 'dipy':
        use_mapping = 'mapping-inv' if inverse else 'mapping'
        if dict_deform[use_mapping] is None:
            logging.debug('missing (%s) transformation', use_mapping)
            return img_warped
        if inverse:
            img_warped = dict_deform['mapping-inv'].transform_inverse(img, method)
        else:
            img_warped = dict_deform['mapping'].transform(img, method)
    else:
        logging.error('missing warp interpreter')
    return img_warped


def warp2d_apply_deform_field(img, deform, method='linear'):
    """ warping reconstructed image using atlas and weight
    to the expected image image domain

    :param ndarray img:
    :param ndarray deform:
    :return ndarray:

    >>> img1 = np.zeros((8, 12), dtype=int)
    >>> img1[2:6, 3:9] = 1
    >>> deform = np.ones(img1.shape + (2,))
    >>> deform[:, :, 1] *= -2
    >>> warp2d_apply_deform_field(img1, deform)
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
        'image %r and deform %r size should match' % (img.shape, deform.shape)
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


def wrapper_warp2d_transform_image(idx_img_deform, method='linear', inverse=False):
    """ wrapper for registration of input images to reconstructed as demons

    :param (int, ndarray, ndarray) idx_img_deform:
    :return (int, ndarray):
    """
    idx, img, d_deform = idx_img_deform
    img_warped = warp2d_transform_image(img, d_deform, method=method,
                                        inverse=inverse)
    return idx, img_warped


def warp2d_images_deformations(list_images, list_deforms, method='linear',
                               inverse=False, nb_workers=NB_THREADS):
    """ deform whole set of images to expected image domain

    :param [ndarray] list_images:
    :param ndarray list_deforms:
    :param int nb_workers:
    :return: [ndarray]

    >>> img = np.zeros((5, 9), dtype=int)
    >>> img[:3, 1:5] = 1
    >>> deform = register_demons_sym_diffeom(img, img, smooth_sigma=10.)
    >>> imgs = warp2d_images_deformations([img], [deform], method='nearest')
    >>> imgs  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [array([[0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]]...)]
    """
    assert len(list_images) == len(list_deforms), \
        'number of images (%i) and deformations (%i) have to match' \
        % (len(list_images), len(list_deforms))
    list_deforms = list(list_deforms)

    _wrap_deform = partial(wrapper_warp2d_transform_image,
                           method=method, inverse=inverse)
    list_imgs_wrap = [None] * len(list_images)
    list_items = zip(range(len(list_images)), list_images, list_deforms)
    for idx, img_w in wrap_execute_sequence(_wrap_deform, list_items, nb_workers, desc=None):
        list_imgs_wrap[idx] = img_w

    return list_imgs_wrap


def wrapper_register_demons_image_weights(idx_img_weights, atlas, smooth_coef,
                                          params=None, interp_method='linear',
                                          inverse=False):
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
        params = DIPY_DEAMONS_PARAMS
    # set the maximal number of iteration by image size
    # params['level_iters'] = [max(img.shape)]
    # params['inv_iter'] = max(img.shape)

    # using multiply by 0.5 to set it as the threshold level for fuzzy inputs
    dict_deform = register_demons_sym_diffeom(img, img_reconst, params=params,
                                              smooth_sigma=smooth_coef,
                                              inverse=inverse, verbose=False)

    return idx, dict_deform


def register_images_to_atlas_demons(list_images, atlas, list_weights,
                                    smooth_coef=1., params=None,
                                    interp_method='linear', inverse=False,
                                    rm_mean=True, nb_workers=NB_THREADS):
    """ register whole set of images to estimated atlas and weights
    IDEA: think about parallel registration per sets as for loading images

    :param [ndarray] list_images:
    :param ndarray atlas:
    :param ndarray list_weights:
    :param float coef:
    :param {str: ...} params:
    :param int nb_workers:
    :return: [ndarray], [ndarray]

    >>> import bpdl.pattern_atlas as ptn_atlas
    >>> np.random.seed(42)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> w_bins = np.array([[0, 0], [0, 1], [1, 1]], dtype=bool)
    >>> imgs = ptn_atlas.reconstruct_samples(atlas, w_bins)
    >>> deform = np.ones(atlas.shape + (2,))
    >>> imgs[1] = warp2d_apply_deform_field(imgs[1], deform * -2)
    >>> np.round(imgs[1]).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> _, _ = register_images_to_atlas_demons(imgs, atlas, w_bins, nb_workers=1)
    >>> imgs_w, deforms = register_images_to_atlas_demons(imgs, atlas, w_bins,
    ...                     smooth_coef=20., interp_method='nearest', nb_workers=2)
    >>> np.sum(imgs_w[0])
    0
    >>> imgs_w[1]  # doctest: +SKIP
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> sorted(deforms[1].keys())
    ['mapping', 'mapping-inv', 'package']
    """
    assert len(list_images) == len(list_weights), \
        'number of images (%i) and weights (%i) have to match' \
        % (len(list_images), len(list_weights))
    atlas = np.asarray(atlas, dtype=int)
    list_weights = list(list_weights)

    list_imgs_wrap = [None] * len(list_images)
    list_deform = [None] * len(list_weights)
    iterations = zip(range(len(list_images)), list_images, list_weights)
    _wrapper_register = partial(wrapper_register_demons_image_weights,
                                atlas=atlas, smooth_coef=smooth_coef,
                                params=params, interp_method=interp_method,
                                inverse=inverse)
    for idx, deform in wrap_execute_sequence(_wrapper_register, iterations,
                                             nb_workers, desc=None):
        list_deform[idx] = deform

    # remove mean transform
    if rm_mean:
        for name in ['mapping', 'mapping-inv']:
            list_deform = subtract_mean_deform(list_deform, name)

    _wrapper_warp = partial(wrapper_warp2d_transform_image,
                            method='linear', inverse=False)
    iterations = zip(range(len(list_images)), list_images, list_deform)
    for idx, img_w in wrap_execute_sequence(_wrapper_warp, iterations, nb_workers,
                                            desc=None):
        list_imgs_wrap[idx] = img_w

    return list_imgs_wrap, list_deform


def subtract_mean_deform(list_deform, name):
    mean_field_bw = np.mean([d[name].backward
                             for d in list_deform
                             if d is not None and d[name] is not None], axis=0)
    mean_field_fw = np.mean([d[name].forward
                             for d in list_deform
                             if d is not None and d[name] is not None], axis=0)
    for i, deform in enumerate(list_deform):
        if deform is None or deform[name] is None:
            continue
        list_deform[i][name].backward = deform[name].backward - mean_field_bw
        list_deform[i][name].forward = deform[name].forward - mean_field_fw
    return list_deform


class SmoothSymmetricDiffeomorphicRegistration(SymmetricDiffeomorphicRegistration):

    def __init__(self, metric, smooth_sigma=0.5, **kwargs):
        super(SmoothSymmetricDiffeomorphicRegistration, self).__init__(metric, **kwargs)
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
        new_displacement = smooth_deform_field(new_displacement,
                                               sigma=self.smooth_sigma)

        # We assume that both displacement fields have the same
        # grid2world transform, which implies premult_index=Identity
        # and premult_disp is the world2grid transform associated with
        # the displacements' grid
        self.compose(current_displacement, new_displacement, None,
                     disp_world2grid, time_scaling, current_displacement)

        return np.array(current_displacement), np.array(mean_norm)

    def _get_energy_derivative(self):
        r"""Approximate derivative of the energy profile
        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration

        NOTE: this is just temporal fix until the bug fix is released in next version dipy>0.14.0
        """
        n_iter = len(self.energy_list)
        if n_iter < self.energy_window:
            raise ValueError('Not enough data to fit the energy profile')
        x = range(self.energy_window)
        y = self.energy_list[(n_iter - self.energy_window):n_iter]
        ss = sum(y)
        if not ss == 0:  # avoid division by zero
            ss = - ss if ss > 0 else ss
            y = [v / ss for v in y]
        der = self._approximate_derivative_direct(x, y)
        return der
