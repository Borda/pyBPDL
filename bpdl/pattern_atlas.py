"""
Estimating the pattern dictionary module

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
# from __future__ import absolute_import
import logging

# import numba
import numpy as np
from sklearn.decomposition import SparsePCA, FastICA, DictionaryLearning, NMF
from skimage import morphology, measure, segmentation, filters
from scipy import ndimage as ndi

from bpdl.data_utils import image_deform_elastic, extract_image_largest_element
from bpdl.pattern_weights import (weights_label_atlas_overlap_threshold,
                                  convert_weights_binary2indexes)

REINIT_PATTERN_COMPACT = True
UNARY_BACKGROUND = 1

# TRY: init: Otsu threshold on sum over all input images -> WaterShade on dist
# TRY: init: sum over all in images and use it negative as dist for WaterShade


def init_atlas_random(im_size, nb_patterns, rand_seed=None):
    """ initialise atlas with random labels

    :param tuple(int,int) im_size: size of image
    :param int nb_patterns: number of labels
    :param rand_seed: random initialization
    :return ndarray: np.array<height, width>

    >>> init_atlas_random((6, 12), 4, rand_seed=0)
    array([[1, 4, 2, 1, 4, 4, 4, 4, 2, 4, 2, 3],
           [1, 4, 3, 1, 1, 1, 3, 2, 3, 4, 4, 3],
           [1, 2, 2, 2, 2, 1, 2, 1, 4, 1, 4, 2],
           [3, 4, 4, 1, 3, 4, 1, 2, 4, 2, 4, 4],
           [3, 4, 1, 2, 2, 2, 4, 1, 4, 3, 1, 4],
           [4, 3, 4, 3, 4, 1, 3, 1, 1, 1, 2, 2]])
    """
    logging.debug('initialise atlas %r as random labeling', im_size)
    nb_labels = nb_patterns + 1
    # reinit seed to have random samples even in the same time
    np.random.seed(rand_seed)
    img_init = np.random.randint(1, nb_labels, im_size)
    return np.array(img_init, dtype=np.int)


def init_atlas_grid(im_size, nb_patterns, rand_seed=None):
    """ initialise atlas with a grid schema

    :param tuple(int,int) im_size: size of image
    :param int nb_patterns: number of pattern in the atlas to be set
    :param rand_seed: random initialisation
    :return ndarray: np.array<height, width>

    >>> init_atlas_grid((6, 12), 4, rand_seed=0)
    array([[3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
           [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
           [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]])
    >>> init_atlas_grid((6, 17), 5, rand_seed=0)
    array([[3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2],
           [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2],
           [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2],
           [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 0],
           [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 0],
           [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 0]])
    """
    # reinit seed to have random samples even in the same time
    np.random.seed(rand_seed)
    labels = np.random.permutation(range(1, nb_patterns + 1)).tolist()
    block_size = np.ceil(np.array(im_size) / np.sqrt(nb_patterns)).astype(int)
    img = np.zeros(im_size, dtype=int)
    for i in range(0, im_size[0], block_size[0]):
        for j in range(0, im_size[1], block_size[1]):
            label = labels.pop(0) if len(labels) > 0 else 0
            img[i:i + block_size[0], j:j + block_size[1]] = label
    return img


def init_atlas_mosaic(im_size, nb_patterns, coef=1., rand_seed=None):
    """ generate grids texture and into each rectangle plase a label,
    each row contains all labels (permutation)

    :param tuple(int,int) im_size: size of image
    :param int nb_patterns: number of pattern in the atlas to be set
    :param float coef:
    :param rand_seed: random initialization
    :return ndarray: np.array<height, width>

    >>> init_atlas_mosaic((8, 12), 3, rand_seed=0)
    array([[3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
           [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
    >>> init_atlas_mosaic((8, 12), 4, coef=2., rand_seed=0)
    array([[3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 1, 1],
           [3, 3, 2, 2, 4, 4, 1, 1, 3, 3, 4, 4],
           [4, 4, 4, 4, 1, 1, 3, 3, 1, 1, 3, 3],
           [1, 1, 3, 3, 2, 2, 2, 2, 3, 3, 1, 1],
           [3, 3, 1, 1, 3, 3, 4, 4, 2, 2, 2, 2],
           [2, 2, 3, 3, 4, 4, 1, 1, 3, 3, 1, 1],
           [4, 4, 3, 3, 1, 1, 1, 1, 3, 3, 4, 4],
           [4, 4, 3, 3, 1, 1, 2, 2, 4, 4, 2, 2]])
    """
    logging.debug('initialise atlas %r as grid labeling', im_size)
    max_label = int(np.ceil(nb_patterns * coef))
    assert max_label > 0, 'at least some labels should be reuested'
    # reinit seed to have random samples even in the same time
    np.random.seed(rand_seed)
    block_size = np.ceil(np.array(im_size) / float(max_label))
    block = np.ones(block_size.astype(np.int))
    vec = list(range(max_label))
    logging.debug('block size is %r', block.shape)
    rows = []
    for _ in range(0, max_label):
        vec = np.random.permutation(vec)
        row = np.hstack([block.copy() * vec[k] for k in range(max_label)])
        rows.append(row)
    mosaic = np.vstack(rows)
    logging.debug('generated mosaic %r with labeling %r', mosaic.shape,
                  np.unique(mosaic).tolist())
    img_init = mosaic[:im_size[0], :im_size[1]]
    img_init = np.remainder(img_init, nb_patterns) + 1
    return np.array(img_init, dtype=np.int)


def init_atlas_otsu_watershed_2d(imgs, nb_patterns=None, bg_threshold=0.5,
                                 bg_type='none'):
    """ do some simple operations to get better initialisation
    1] sum over all images, 2] Otsu thresholding, 3] watershed

    :param [ndarray] imgs: list of images np.array<height, width>
    :param int nb_patterns: number of pattern in the atlas to be set
    :param str bg_type: set weather the Otsu backround should be filled randomly
    :param float bg_threshold: threshold foe binarisation
    :return ndarray: np.array<height, width>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas].astype(float) for lut in luts]
    >>> init_atlas_otsu_watershed_2d(imgs, 5)
    array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> np.random.seed(0)
    >>> init_atlas_otsu_watershed_2d(imgs, 5, bg_type='rand')
    array([[5, 0, 1, 1, 0, 2, 4, 3, 5, 1, 1, 5],
           [3, 0, 1, 1, 0, 1, 2, 5, 4, 1, 4, 1],
           [3, 0, 0, 0, 0, 4, 4, 1, 2, 2, 2, 1],
           [3, 5, 4, 4, 3, 5, 0, 0, 0, 0, 0, 0],
           [2, 5, 2, 3, 3, 1, 0, 2, 2, 2, 2, 2],
           [3, 4, 1, 4, 5, 2, 0, 2, 2, 2, 2, 2],
           [4, 5, 5, 5, 1, 5, 0, 0, 0, 0, 0, 0],
           [1, 1, 2, 3, 5, 3, 1, 4, 3, 3, 1, 2]])
    """
    logging.debug('initialise atlas for %i labels from %i images of shape %r'
                  ' with Otsu-Watershed', nb_patterns, len(imgs), imgs[0].shape)
    img_sum = np.sum(np.asarray(imgs), axis=0) / float(len(imgs))
    img_gauss = filters.gaussian(img_sum.astype(np.float64), 1)
    # http://scikit-image.org/docs/dev/auto_examples/plot_otsu.html
    thresh = filters.threshold_otsu(img_gauss)
    img_otsu = (img_gauss >= thresh)
    # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
    img_dist = ndi.distance_transform_edt(img_otsu)
    seeds = detect_peaks(img_otsu)
    labels = morphology.watershed(-img_dist, seeds)
    labels[img_gauss < bg_threshold] = 0
    if nb_patterns is not None:
        labels = np.remainder(labels, nb_patterns + 1)
    if bg_type == 'rand':
        # add random labels on the potential backgound
        img_rand = np.random.randint(1, nb_patterns + 1, img_sum.shape)
        labels[img_otsu == 0] = img_rand[img_otsu == 0]
    return labels.astype(np.int)


def detect_peaks(image, struct=(2, 2)):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    :param ndarray image:
    :param tuple(int,int) struct:

    >>> img = [[0.46, 0.62, 0.62, 0.46, 0.2,  0.04, 0.01, 0.0,  0.0,  0.0],
    ...        [0.44, 0.59, 0.59, 0.44, 0.2,  0.06, 0.04, 0.04, 0.04, 0.04],
    ...        [0.33, 0.44, 0.44, 0.34, 0.2,  0.17, 0.19, 0.2,  0.2,  0.2],
    ...        [0.14, 0.19, 0.19, 0.17, 0.2,  0.34, 0.44, 0.46, 0.47, 0.47],
    ...        [0.03, 0.04, 0.04, 0.06, 0.2,  0.44, 0.59, 0.62, 0.62, 0.62],
    ...        [0.0,  0.0,  0.01, 0.04, 0.19, 0.44, 0.59, 0.62, 0.62, 0.62],
    ...        [0.0,  0.0,  0.0,  0.03, 0.14, 0.33, 0.44, 0.46, 0.47, 0.47],
    ...        [0.0,  0.0,  0.0,  0.01, 0.06, 0.14, 0.19, 0.2,  0.2,  0.2]]
    >>> detect_peaks(np.array(img))
    array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    # define an 8-connected neighborhood
    neighborhood = ndi.morphology.generate_binary_structure(*struct)
    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = ndi.filters.maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # we create the mask of the background
    background = (image == 0)
    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = ndi.morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    # label each peak by single label
    labeled_peaks = measure.label(detected_peaks)
    return labeled_peaks


def init_atlas_gauss_watershed_2d(imgs, nb_patterns=None,
                                  bg_threshhold=0.5):
    """ do some simple operations to get better initialisation
    1] sum over all images, 2]watershed

    :param [ndarray] imgs: list of input images np.array<height, width>
    :param int nb_patterns: number of pattern in the atlas to be set
    :param float bg_threshhold: threshold foe binarisation
    :return ndarray: np.array<height, width>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 9 + [[0, 0, 1]] * 9 + [[0, 1, 1]] * 9)
    >>> imgs = [lut[atlas].astype(float) for lut in luts]
    >>> init_atlas_gauss_watershed_2d(imgs, 5)
    array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('initialise atlas for %i labels from %i images of shape %r'
                  ' with Gauss-Watershed', nb_patterns, len(imgs), imgs[0].shape)
    img_sum = np.sum(np.asarray(imgs), axis=0) / float(len(imgs))
    img_gauss = filters.gaussian(img_sum.astype(np.float64), 1)
    seeds = detect_peaks(img_gauss)
    # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
    labels = morphology.watershed(-img_gauss, seeds)
    labels[img_gauss < bg_threshhold] = 0
    if nb_patterns is not None:
        labels = np.remainder(labels, nb_patterns + 1)
    return labels.astype(np.int)


def convert_lin_comb_patterns_2_atlas(atlas_components, used_components,
                                      bg_threshold=0.01):
    """ conver components rom linear decompostion into an atlass

    :param [ndarray] atlas_components:
    :param list(bool) used_components:
    :param float bg_threshold:
    :return ndarray:
    """
    atlas_components = np.abs(atlas_components)
    atlas_components = atlas_components[used_components, :]
    # take the maximal component
    atlas_mean = np.mean(atlas_components, axis=0)
    component_sum = [np.sum(atlas_components[i, ...])
                     for i in range(atlas_components.shape[0])]
    idxs = np.argsort(component_sum)[::-1]
    atlas = np.argmax(atlas_components[idxs, ...], axis=0) + 1
    # filter small values
    atlas[atlas_mean < bg_threshold] = 0
    # atlas = self._estim_atlas_as_unique_sum(atlas_ptns)
    atlas = segmentation.relabel_sequential(atlas)[0]
    atlas = np.remainder(atlas, len(used_components))
    return atlas


def init_atlas_nmf(imgs, nb_patterns, nb_iter=25, bg_threshold=0.1):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [ndarray] imgs: list of input images
    :param int nb_patterns: number of pattern in the atlas to be set
    :param int nb_iter: max number of iterations
    :param float bg_threshold:
    :return ndarray: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 99 + [[0, 0, 1]] * 99 + [[0, 1, 1]] * 99)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> init_atlas_nmf(imgs, 2)
    array([[0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    imgs_vec = np.array([np.ravel(im) for im in imgs])

    try:
        estimator = NMF(n_components=nb_patterns + 1,
                        max_iter=nb_iter,
                        init='random')
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_

        ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
        atlas_ptns = components.reshape((-1, ) + imgs[0].shape)

        atlas = convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used,
                                                  bg_threshold)
    except Exception:
        logging.exception('CRASH: %s' % init_atlas_nmf.__name__)
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def init_atlas_fast_ica(imgs, nb_patterns, nb_iter=25, bg_threshold=0.1):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [ndarray] imgs: list of input images
    :param int nb_patterns: number of pattern in the atlas to be set
    :param int nb_iter: max number of iterations
    :param float bg_threshold:
    :return ndarray: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 99 + [[0, 0, 1]] * 99 + [[0, 1, 1]] * 999)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> init_atlas_fast_ica(imgs, 2, bg_threshold=0.6)
    array([[0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    imgs_vec = np.array([np.ravel(im) for im in imgs])

    try:
        estimator = FastICA(n_components=nb_patterns + 1,
                            max_iter=nb_iter,
                            algorithm='deflation',
                            whiten=True)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.mixing_.T

        ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
        atlas_ptns = components.reshape((-1, ) + imgs[0].shape)

        atlas = convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used,
                                                  bg_threshold)
    except Exception:
        logging.exception('CRASH: %s' % init_atlas_fast_ica.__name__)
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def init_atlas_sparse_pca(imgs, nb_patterns, nb_iter=5, bg_threshold=0.1):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [ndarray] imgs: list of input images
    :param int nb_patterns: number of pattern in the atlas to be set
    :param int nb_iter: max number of iterations
    :param float bg_threshold:
    :return ndarray: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 99 + [[0, 0, 1]] * 99 + [[0, 1, 1]] * 99)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> init_atlas_sparse_pca(imgs, 2)
    array([[0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    imgs_vec = np.array([np.ravel(im) for im in imgs])

    try:
        estimator = SparsePCA(n_components=nb_patterns + 1,
                              max_iter=nb_iter)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_

        ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
        atlas_ptns = components.reshape((-1, ) + imgs[0].shape)

        atlas = convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used,
                                                  bg_threshold)
    except Exception:
        logging.exception('CRASH: %s' % init_atlas_sparse_pca.__name__)
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def init_atlas_dict_learn(imgs, nb_patterns, nb_iter=5, bg_threshold=0.1):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [ndarray] imgs: list of input images
    :param int nb_patterns: number of pattern in the atlas to be set
    :param int nb_iter: max number of iterations
    :param float bg_threshold:
    :return ndarray: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 99 + [[0, 0, 1]] * 99 + [[0, 1, 1]] * 99)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> init_atlas_dict_learn(imgs, 2)
    array([[0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    imgs_vec = np.array([np.ravel(im) for im in imgs])

    try:
        estimator = DictionaryLearning(n_components=nb_patterns + 1,
                                       fit_algorithm='lars',
                                       transform_algorithm='omp',
                                       split_sign=False,
                                       max_iter=nb_iter)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_

        ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
        atlas_ptns = components.reshape((-1, ) + imgs[0].shape)

        atlas = convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used,
                                                  bg_threshold)
    except Exception:
        logging.exception('CRASH: %s', init_atlas_dict_learn.__name__)
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def init_atlas_deform_original(atlas, coef=0.5, grid_size=(20, 20),
                               rand_seed=None):
    """ take the orginal atlas and use geometrical deformation
    to generate new deformed atlas

    :param ndarray atlas: np.array<height, width>
    :param float coef:
    :param tuple(int,int) grid_size:
    :param rand_seed: random initialization
    :return ndarray: np.array<height, width>

    >>> img = np.zeros((8, 12))
    >>> img[:3, 1:5] = 1
    >>> img[3:7, 6:12] = 2
    >>> init_atlas_deform_original(img, 0.1, (5, 5), rand_seed=0)
    array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('initialise atlas by deforming original one')
    res = image_deform_elastic(atlas, coef, grid_size, rand_seed=rand_seed)
    return np.array(res, dtype=np.int)


# @numba.jit
def reconstruct_samples(atlas, w_bins):
    """ create reconstruction of binary images according given atlas and weights

    :param ndarray atlas: input atlas np.array<height, width>
    :param ndarray w_bins: np.array<nb_imgs, nb_lbs>
    :return [ndarray]: [np.array<height, width>]

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> w_bins = np.array([[0, 0], [0, 1], [1, 1]], dtype=bool)
    >>> imgs = reconstruct_samples(atlas, w_bins)
    >>> np.sum(imgs[0])
    0
    >>> imgs[1]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> imgs[2]
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    # w_bins = np.array(w_bins)
    w_bin_ext = np.append(np.zeros((w_bins.shape[0], 1)), w_bins, axis=1)
    atlas = np.asarray(atlas, dtype=int)
    imgs = [None] * w_bins.shape[0]
    for i, w in enumerate(w_bin_ext):
        imgs[i] = np.asarray(w)[atlas].astype(atlas.dtype)
        assert atlas.shape == imgs[i].shape, 'im. size of atlas and image'
    return imgs


def prototype_new_pattern(imgs, imgs_reconst, diffs, atlas,
                          ptn_compact=REINIT_PATTERN_COMPACT, thr_fuzzy=0.5,
                          ptn_method='WaterShade'):
    """ estimate new pattern that occurs in input images and is not cover
    by any label in actual atlas, remove collision with actual atlas

    :param [ndarray] imgs: list of input images np.array<height, width>
    :param [ndarray] imgs_reconst: list of reconstructed images np.array<h, w>
    :param ndarray atlas: np.array<height, width>
    :param list(int) diffs: list of differences among input and reconstruct images
    :param bool ptn_compact: enforce compactness of patterns
    :param str ptn_method: pattern extraction method
    :param float thr_fuzzy:
    :return ndarray: np.array<height, width> binary single pattern

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> atlas[atlas == 1] = 0
    >>> imgs_reconst = [lut[atlas] for lut in luts]
    >>> diffs = [np.sum(np.abs(im - imR)) for im, imR in zip(imgs, imgs_reconst)]
    >>> img_ptn = prototype_new_pattern(imgs, imgs_reconst, diffs, atlas)
    >>> img_ptn.astype(int)
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> img_ptn = prototype_new_pattern(imgs, imgs_reconst, diffs, atlas,
    ...                                 ptn_compact=True, ptn_method='Morpho')
    >>> img_ptn.astype(int)
    array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> atlas[atlas == 2] = 0
    >>> imgs_reconst = [lut[atlas] for lut in luts]
    >>> diffs = [np.sum(np.abs(im - imR)) for im, imR in zip(imgs, imgs_reconst)]
    >>> img_ptn = prototype_new_pattern(imgs, imgs_reconst, diffs, atlas,
    ...                                 ptn_compact=False)
    >>> img_ptn.astype(int)
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    id_max = np.argmax(diffs)
    # take just positive differences
    assert thr_fuzzy >= 0, 'threshold has to be a positive number'
    im_diff = np.logical_and((imgs[id_max] - imgs_reconst[id_max]) > thr_fuzzy,
                             atlas == 0)
    if not ptn_compact:
        return im_diff
    if ptn_method == 'WaterShade':  # WaterShade
        logging.debug('.. reinit. pattern using WaterShade')
        # im_diff = morphology.opening(im_diff, morphology.disk(3))
        # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
        dist = ndi.distance_transform_edt(im_diff)
        try:
            peaks = detect_peaks(dist)
            labels = morphology.watershed(-dist, peaks, mask=im_diff)
        except Exception:
            logging.exception('morphology.watershed')
            labels = None
    else:
        logging.debug('.. reinit. pattern as major component')
        im_diff = morphology.closing(im_diff, morphology.disk(1))
        labels = None
    # find largest connected component
    img_ptn = extract_image_largest_element(im_diff, labels)
    # ptn_size = np.sum(ptn) / float(np.product(ptn.shape))
    # if ptn_size < 0.01:
    #     logging.debug('new patterns was too small %f', ptn_size)
    #     ptn = data.extract_image_largest_element(im_diff)
    img_ptn = (img_ptn == 1)
    # img_ptn = np.logical_and(img_ptn == True, atlas == 0)
    return img_ptn


def insert_new_pattern(imgs, imgs_reconst, atlas, label,
                       ptn_compact=REINIT_PATTERN_COMPACT):
    """ with respect to atlas empty spots inset new patterns

    :param [ndarray] imgs: list of input images np.array<height, width>
    :param [ndarray] imgs_reconst: list of reconstructed images np.array<h, w>
    :param ndarray atlas: np.array<height, width>
    :param int label:
    :param bool ptn_compact: enforce compactness of patterns
    :return ndarray: np.array<height, width> updated atlas

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> atlas[atlas == 1] = 0
    >>> imgs_reconst = [lut[atlas] for lut in luts]
    >>> atlas_new = insert_new_pattern(imgs, imgs_reconst, atlas, 3)
    >>> atlas_new
    array([[0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    # dimension indexes except the first one
    dims = tuple(range(1, 1 + imgs[0].ndim))
    # count just absolute difference
    diffs = np.sum((np.asarray(imgs) - np.asarray(imgs_reconst) > 0), axis=dims)
    im_ptn = prototype_new_pattern(imgs, imgs_reconst, diffs, atlas, ptn_compact)
    # plt.imshow(im_ptn), plt.title('im_ptn'), plt.show()
    atlas[im_ptn == 1] = label
    logging.debug('area of new pattern %i is %i', label, np.sum(atlas == label))
    return atlas


def reinit_atlas_likely_patterns(imgs, w_bins, atlas, label_max=None,
                                 ptn_compact=REINIT_PATTERN_COMPACT):
    """ walk and find all all free labels and try to reinit them by new patterns

    :param [ndarray] imgs: list of input images np.array<height, width>
    :param ndarray w_bins: binary weighs np.array<nb_imgs, nb_lbs>
    :param ndarray atlas: image np.array<height, width>
    :param int label_max: set max number of components
    :param bool ptn_compact: enforce compactness of patterns
    :return tuple(ndarray, ndarray): np.array<height, width>, np.array<nb_imgs, nb_lbs>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> atlas[atlas == 2] = 0
    >>> w_bins = luts[:, 1:]
    >>> w_bins[:, 1] = 0
    >>> atlas_new, w_bins = reinit_atlas_likely_patterns(imgs, w_bins, atlas)
    >>> atlas_new
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> w_bins
    array([[1, 0],
           [1, 0],
           [1, 0],
           [0, 1],
           [0, 1],
           [0, 1],
           [1, 1],
           [1, 1],
           [1, 1]])
    >>> _ = reinit_atlas_likely_patterns(imgs, w_bins, atlas, np.max(atlas) + 5)
    """
    if label_max is None:
        label_max = max(np.max(atlas), w_bins.shape[1])
    else:
        logging.debug('compare w_bin %r to max %i', w_bins.shape, label_max)
        for i in range(w_bins.shape[1], label_max):
            logging.debug('adding disappeared weigh column %i', i)
            w_bins = np.append(w_bins, np.zeros((w_bins.shape[0], 1)), axis=1)
    # w_bin_ext = np.append(np.zeros((w_bins.shape[0], 1)), w_bins, axis=1)
    # logging.debug('IN > sum over weights: %s', repr(np.sum(w_bin_ext, axis=0)))
    # add one while indexes does not cover 0 - bg
    logging.debug('total nb labels: %i', label_max)
    atlas_new = atlas.copy()
    labels_empty = [lb for lb in range(1, label_max + 1)
                    if np.sum(w_bins[:, lb - 1]) == 0]
    logging.debug('reinit. following labels: %r', labels_empty)
    for label in labels_empty:
        w_index = label - 1
        imgs_reconst = reconstruct_samples(atlas_new, w_bins)
        atlas_new = insert_new_pattern(imgs, imgs_reconst, atlas_new, label,
                                       ptn_compact)
        # logging.debug('w_bins before: %i', np.sum(w_bins[:, w_index]))
        # BE AWARE OF THIS CONSTANT, it can caused that there are weight even
        # they should not be which lead to have high unary for atlas estimation
        lim_repopulate = 1. / label_max
        w_bins[:, w_index] = weights_label_atlas_overlap_threshold(imgs, atlas_new, label,
                                                                   lim_repopulate)
        logging.debug('reinit. label: %i with w_bins after: %i',
                      label, np.sum(w_bins[:, w_index]))
    return atlas_new, w_bins


def atlas_split_indep_ptn(atlas, label_max):
    """ split  independent patterns labeled equally

    :param ndarray atlas: image np.array<height, width>
    :param int label_max:
    :return ndarray: np.array<height, width>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 1
    >>> atlas_split_indep_ptn(atlas, 2)
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    patterns = []
    for label in np.unique(atlas):
        labeled, nb_objects = ndi.label(atlas == label)
        logging.debug('for label %i detected #%i', label, nb_objects)
        ptn = [(labeled == j) for j in np.unique(labeled)]
        # skip the largest one assuming to be background
        patterns += sorted(ptn, key=lambda x: np.sum(x), reverse=True)[1:]
    patterns = sorted(patterns, key=lambda x: np.sum(x), reverse=True)
    logging.debug('list of all areas %r', [np.sum(p) for p in patterns])
    atlas_new = np.zeros(atlas.shape, dtype=np.int)
    # take just label_max largest elements
    for i, ptn in enumerate(patterns[:label_max]):
        label = i + 1
        # logging.debug('pattern #%i area %i', lb, np.sum(ptn))
        atlas_new[ptn] = label

    # plt.figure()
    # plt.subplot(121), plt.imshow(atlas), plt.colorbar()
    # plt.subplot(122), plt.imshow(atlas_new), plt.colorbar()
    # plt.show()
    logging.debug('atlas unique %r', np.unique(atlas_new))
    return atlas_new


def edges_in_image2d_plane(im_size, connect_diag=False):
    """ create list of edges for uniform image plane

    :param tuple(int,int) im_size: size of image
    :param bool connect_diag: used connecting diagonals, like use 8- instead 4-neighbour
    :return [[int, int], ]:

    >>> im_size = [2, 3]
    >>> np.reshape(range(np.product(im_size)), im_size)
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> e, w = edges_in_image2d_plane(im_size)
    >>> e.tolist()
    [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]]
    >>> w.tolist()
    [1, 1, 1, 1, 1, 1, 1]
    >>> e, w = edges_in_image2d_plane(im_size, connect_diag=True)
    >>> e.tolist()  #doctest: +NORMALIZE_WHITESPACE
    [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5],
     [0, 4], [1, 5], [1, 3], [2, 4]]
    >>> w.tolist()  #doctest: +ELLIPSIS
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.41..., 1.41..., 1.41..., 1.41...]
    """
    idx_grid = np.arange(np.product(im_size))
    idx_grid = idx_grid.reshape(im_size)
    # logging.debug(idxs)
    edges = list(zip(idx_grid[:, :-1].ravel(), idx_grid[:, 1:].ravel()))
    edges += list(zip(idx_grid[:-1, :].ravel(), idx_grid[1:, :].ravel()))
    weights = [1] * len(edges)
    if connect_diag:
        edges += list(zip(idx_grid[:-1, :-1].ravel(), idx_grid[1:, 1:].ravel()))
        edges += list(zip(idx_grid[:-1, 1:].ravel(), idx_grid[1:, :-1].ravel()))
        weights += [np.sqrt(2)] * np.product(np.array(im_size) - 1) * 2
    assert len(edges) == len(weights), 'the lengths must match'
    edges = np.array(edges)
    weights = np.array(weights)
    logging.debug('edges for image plane are shape %r', edges.shape)
    logging.debug('edges weights are shape %r', weights.shape)
    return edges, weights


def compute_relative_penalty_images_weights(imgs, weights):
    """ compute the relative penalty for all pixel and cjsing each label
    on that particular position

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> pott = compute_relative_penalty_images_weights(imgs, luts[:, 1:])
    >>> pott   # doctest: +ELLIPSIS
    array([[[ 0.        ,  0.666...,  0.666...],
            [ 0.666...,  0.        ,  0.666...],
            [ 0.666...,  0.        ,  0.666...],
            [ 0.666...,  0.        ,  0.666...],
            [ 0.666...,  0.        ,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...]],
    <BLANKLINE>
    ...
    <BLANKLINE>
           [[ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...],
            [ 0.        ,  0.666...,  0.666...]]])
    """
    logging.debug('compute unary cost from images and related weights')
    # weightsIdx = ptn_weight.convert_weights_binary2indexes(weights)
    nb_lbs = weights.shape[1] + 1
    assert len(imgs) == weights.shape[0], \
        'not matching nb images (%i) and nb weights (%i)' \
        % (len(imgs), weights.shape[0])
    pott_sum = np.zeros(imgs[0].shape + (nb_lbs,))
    # extenf the weights by background value 0
    weights_ext = np.append(np.zeros((weights.shape[0], 1)), weights, axis=1)
    # logging.debug(weights_ext)
    imgs = np.array(imgs)
    logging.debug('DIMS potts: %r, imgs %r, w_bin: %r',
                  pott_sum.shape, imgs.shape, weights_ext.shape)
    logging.debug('... walk over all pixels in each image')
    for i in range(pott_sum.shape[0]):
        for j in range(pott_sum.shape[1]):
            # make it as matrix ops
            img_vals = np.repeat(imgs[:, i, j, np.newaxis],
                                 weights_ext.shape[1], axis=1)
            pott_sum[i, j] = np.sum(np.abs(weights_ext - img_vals), axis=0)
    pott_sum_norm = pott_sum / float(len(imgs))
    return pott_sum_norm


def compute_positive_cost_images_weights(imgs, ptn_weights):
    """
    :param [ndarray] imgs: list of np.array<height, width> input images
    :param ndarray ptn_weights: matrix np.array<nb_imgs, nb_lbs> composed
        from wight vectors
    :return ndarray: np.array<height, width, nb_lbs>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> pott = compute_positive_cost_images_weights(imgs, luts[:, 1:])
    >>> pott   # doctest: +ELLIPSIS
    array([[[ 1.,  0.,  0.],
            [ 1.,  3.,  0.],
            [ 1.,  3.,  0.],
            [ 1.,  3.,  0.],
            [ 1.,  3.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.]],
    <BLANKLINE>
    ...
    <BLANKLINE>
           [[ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  0.]]])
    """
    # not using any more...
    logging.debug('compute unary cost from images and related ptn_weights')
    w_idx = convert_weights_binary2indexes(ptn_weights)
    nb_lbs = ptn_weights.shape[1] + 1
    assert len(imgs) == len(w_idx), 'nb of images (%i) and weights (%i) ' \
                                    'do not match' % (len(imgs), len(w_idx))
    pott_sum = np.zeros(imgs[0].shape + (nb_lbs,))
    # walk over all pixels in image
    logging.debug('... walk over all pixels in each image')
    for i in range(pott_sum.shape[0]):
        for j in range(pott_sum.shape[1]):
            # per all images in list
            for k in range(len(imgs)):
                # if pixel is active
                if imgs[k][i, j] == 1:
                    # increment all possible spots
                    for x in w_idx[k]:
                        pott_sum[i, j, x] += 1
                # else:
                #     graphSum[i,j,0] += 1e-9
            # set also the background values
            pott_sum[i, j, 0] = UNARY_BACKGROUND
    # graph = 1. / (graphSum +1)
    return pott_sum
