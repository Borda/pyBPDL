"""
Estimating the pattern dictionary module

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
import traceback

# to suppress all visual, has to be on the beginning
import matplotlib
matplotlib.use('Agg')

from sklearn.decomposition import SparsePCA, FastICA, DictionaryLearning, NMF
from skimage import morphology, measure, segmentation, filters
from scipy import ndimage as ndi
import numpy as np

import dataset_utils as data
import pattern_weights as ptn_weight

REINIT_PATTERN_COMPACT = True

# TODO: init: Otsu threshold on sum over all input images -> WaterShade on distance
# TODO: init: sum over all input images and use it negative as distance for WaterShade


def initialise_atlas_random(im_size, nb_patterns, rnd_seed=None):
    """ initialise atlas with random labels

    :param (int, int) im_size: size of image
    :param int label_max: number of labels
    :return: np.array<height, width>

    >>> initialise_atlas_random((6, 12), 4, rnd_seed=0)
    array([[1, 4, 2, 1, 4, 4, 4, 4, 2, 4, 2, 3],
           [1, 4, 3, 1, 1, 1, 3, 2, 3, 4, 4, 3],
           [1, 2, 2, 2, 2, 1, 2, 1, 4, 1, 4, 2],
           [3, 4, 4, 1, 3, 4, 1, 2, 4, 2, 4, 4],
           [3, 4, 1, 2, 2, 2, 4, 1, 4, 3, 1, 4],
           [4, 3, 4, 3, 4, 1, 3, 1, 1, 1, 2, 2]])
    """
    logging.debug('initialise atlas %s as random labeling', repr(im_size))
    nb_labels = nb_patterns + 1
    # reinit seed to have random samples even in the same time
    np.random.seed(rnd_seed)
    img_init = np.random.randint(1, nb_labels, im_size)
    return np.array(img_init, dtype=np.int)


def initialise_atlas_grid(im_size, nb_patterns, rnd_seed=None):
    """

    :param (int, int) im_size: size of image
    :param int nb_labels: number of labels
    :return: np.array<height, width>

    >>> initialise_atlas_grid((6, 12), 4, rnd_seed=0)
    array([[3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
           [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
           [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]])
    >>> initialise_atlas_grid((6, 17), 5, rnd_seed=0)
    array([[3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2],
           [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2],
           [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2],
           [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 0],
           [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 0],
           [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 0]])
    """
    # reinit seed to have random samples even in the same time
    np.random.seed(rnd_seed)
    labels = np.random.permutation(range(1, nb_patterns + 1)).tolist()
    block_size = np.ceil(np.array(im_size) / np.sqrt(nb_patterns)).astype(int)
    img = np.zeros((im_size), dtype=int)
    for i in range(0, im_size[0], block_size[0]):
        for j in range(0, im_size[1], block_size[1]):
            label = labels.pop(0) if len(labels) > 0 else 0
            img[i:i + block_size[0], j:j + block_size[1]] = label
    return img


def initialise_atlas_mosaic(im_size, nb_patterns, coef=1., rnd_seed=None):
    """ generate grids texture and into each rectangle plase a label,
    each row contains all labels (permutation)

    :param (int, int) im_size: size of image
    :param int nb_labels: number of labels
    :return: np.array<height, width>

    >>> initialise_atlas_mosaic((8, 12), 4, rnd_seed=0)
    array([[3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1],
           [3, 3, 3, 4, 4, 4, 2, 2, 2, 1, 1, 1],
           [1, 1, 1, 3, 3, 3, 2, 2, 2, 4, 4, 4],
           [1, 1, 1, 3, 3, 3, 2, 2, 2, 4, 4, 4],
           [4, 4, 4, 1, 1, 1, 3, 3, 3, 2, 2, 2],
           [4, 4, 4, 1, 1, 1, 3, 3, 3, 2, 2, 2],
           [2, 2, 2, 1, 1, 1, 3, 3, 3, 4, 4, 4],
           [2, 2, 2, 1, 1, 1, 3, 3, 3, 4, 4, 4]])
    >>> initialise_atlas_mosaic((8, 12), 4, coef=2., rnd_seed=0)
    array([[2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0],
           [3, 3, 2, 2, 0, 0, 2, 2, 1, 1, 4, 4],
           [1, 1, 1, 1, 3, 3, 0, 0, 3, 3, 2, 2],
           [4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
           [1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 4, 4],
           [2, 2, 3, 3, 0, 0, 3, 3, 1, 1, 3, 3],
           [3, 3, 1, 1, 4, 4, 0, 0, 2, 2, 3, 3],
           [3, 3, 2, 2, 1, 1, 0, 0, 3, 3, 2, 2]])
    """
    logging.debug('initialise atlas %s as grid labeling', repr(im_size))
    max_label = int(nb_patterns * coef)
    # reinit seed to have random samples even in the same time
    np.random.seed(rnd_seed)
    block_size = np.ceil(np.array(im_size) / float(max_label))
    block = np.ones(block_size.astype(np.int))
    vec = range(1, max_label + 1) * int(np.ceil(coef))
    logging.debug('block size is %s', repr(block.shape))
    for label in range(max_label):
        idx = np.random.permutation(vec)[:max_label]
        for k in range(max_label):
            b = block.copy() * idx[k]
            if k == 0:
                row = b
            else:
                row = np.hstack((row, b))
        if label == 0: mosaic = row
        else: mosaic = np.vstack((mosaic, row))
    logging.debug('generated mosaic %s with labeling %s',
                 repr(mosaic.shape), repr(np.unique(mosaic).tolist()))
    img_init = mosaic[:im_size[0], :im_size[1]]
    img_init = np.remainder(img_init, nb_patterns + 1)
    return np.array(img_init, dtype=np.int)


def initialise_atlas_otsu_watershed_2d(imgs, nb_patterns=None, bg_threshhold=0.5, bg='none'):
    """ do some simple operations to get better initialisation
    1] sum over all images, 2] Otsu thresholding, 3] watershed

    :param [np.array<height, width>] imgs:
    :param int nb_labels: number of labels
    :param str bg: set weather the Otsu backround sould be filled randomly
    :return: np.array<height, width>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas].astype(float) for lut in luts]
    >>> initialise_atlas_otsu_watershed_2d(imgs, 5)
    array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('initialise atlas for %i labels from %i images of shape %s '
                  'with Otsu-Watershed', nb_patterns, len(imgs), repr(imgs[0].shape))
    img_sum = np.sum(np.asarray(imgs), axis=0) / float(len(imgs))
    img_gauss = filters.gaussian_filter(img_sum, 1)
    # http://scikit-image.org/docs/dev/auto_examples/plot_otsu.html
    thresh = filters.threshold_otsu(img_gauss)
    img_otsu = (img_gauss >= thresh)
    # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
    img_dist = ndi.distance_transform_edt(img_otsu)
    seeds = detect_peaks(img_otsu)
    labels = morphology.watershed(-img_dist, seeds)
    labels[img_gauss < bg_threshhold] = 0
    if nb_patterns is not None:
        labels = np.remainder(labels, nb_patterns + 1)
    if bg == 'rand':
        # add random labels on the potential backgound
        img_rand = np.random.randint(1, nb_patterns + 1, img_sum.shape)
        labels[img_otsu == 0] = img_rand[img_otsu == 0]
    return labels.astype(np.int)


def detect_peaks(image, struct=(2, 2)):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

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
    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    local_max = ndi.filters.maximum_filter(image, footprint=neighborhood) == image
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.
    #we create the mask of the background
    background = (image == 0)
    # a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = ndi.morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    # label each peak by single label
    labeled_peaks = measure.label(detected_peaks)
    return labeled_peaks


def initialise_atlas_gauss_watershed_2d(imgs, nb_patterns=None,
                                        bg_threshhold=0.5):
    """ do some simple operations to get better initialisation
    1] sum over all images, 2]watershed

    :param [np.array<height, width>] imgs: list of input images
    :param int nb_labels:
    :return: np.array<height, width>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas].astype(float) for lut in luts]
    >>> initialise_atlas_gauss_watershed_2d(imgs, 5)
    array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('initialise atlas for %i labels from %i images of shape %s '
                  'with Gauss-Watershed', nb_patterns, len(imgs), repr(imgs[0].shape))
    img_sum = np.sum(np.asarray(imgs), axis=0) / float(len(imgs))
    img_gauss = filters.gaussian_filter(img_sum, 1)
    seeds = detect_peaks(img_gauss)
    # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
    labels = morphology.watershed(-img_gauss, seeds)
    labels[img_gauss < bg_threshhold] = 0
    if nb_patterns is not None:
        labels = np.remainder(labels, nb_patterns + 1)
    return labels.astype(np.int)


def convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used, bg_threshold=0.01):
    atlas_ptns = np.abs(atlas_ptns)
    atlas_ptns = atlas_ptns[ptn_used, :]
    # take the maximal component
    atlas = np.argmax(atlas_ptns, axis=0) + 1
    atlas_sum = np.sum(atlas_ptns, axis=0)
    # filter small values
    atlas[atlas_sum < bg_threshold] = 0
    # atlas = self.estim_atlas_as_unique_sum(atlas_ptns)
    atlas = segmentation.relabel_sequential(atlas)[0]
    atlas = np.remainder(atlas, len(atlas_ptns) + 1)
    return atlas


def initialise_atlas_nmf(imgs, nb_patterns, nb_iter=25, bg_threshold=0.01):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [np.array] imgs: list of input images
    :param int nb_patterns: max number of estimated atlases
    :param int nb_iter: max number of iterations
    :return np.array: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 9 + [[0, 0, 1]] * 9 + [[0, 1, 1]] * 9)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> initialise_atlas_nmf(imgs, 2)
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
        estimator = NMF(n_components=nb_patterns,
                        max_iter=nb_iter,
                        init='random')
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_

        ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
        atlas_ptns = components.reshape((-1, ) + imgs[0].shape)

        atlas = convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used,
                                                  bg_threshold)
    except:
        logging.warning('CRASH - initialise_atlas_nmf')
        logging.warning(traceback.format_exc())
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def initialise_atlas_fast_ica(imgs, nb_patterns, nb_iter=25, bg_threshold=0.01):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [np.array] imgs: list of input images
    :param int nb_patterns: max number of estimated atlases
    :param int nb_iter: max number of iterations
    :return np.array: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 9 + [[0, 0, 1]] * 9 + [[0, 1, 1]] * 9)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> initialise_atlas_fast_ica(imgs, 2)
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
        estimator = FastICA(n_components=nb_patterns,
                            max_iter=nb_iter,
                            algorithm='deflation',
                            whiten=True)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.mixing_.T

        ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
        atlas_ptns = components.reshape((-1, ) + imgs[0].shape)

        atlas = convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used,
                                                  bg_threshold)
    except:
        logging.warning('CRASH - initialise_atlas_fast_ica')
        logging.warning(traceback.format_exc())
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def initialise_atlas_sparse_pca(imgs, nb_patterns, nb_iter=5, bg_threshold=0.01):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [np.array] imgs: list of input images
    :param int nb_patterns: max number of estimated atlases
    :param int nb_iter: max number of iterations
    :return np.array: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 9 + [[0, 0, 1]] * 9 + [[0, 1, 1]] * 9)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> initialise_atlas_sparse_pca(imgs, 2)
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
        estimator = SparsePCA(n_components=nb_patterns,
                              max_iter=nb_iter)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_

        ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
        atlas_ptns = components.reshape((-1, ) + imgs[0].shape)

        atlas = convert_lin_comb_patterns_2_atlas(atlas_ptns, ptn_used,
                                                  bg_threshold)
    except:
        logging.warning('CRASH - initialise_atlas_sparse_pca')
        logging.warning(traceback.format_exc())
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def initialise_atlas_dict_learn(imgs, nb_patterns, nb_iter=5, bg_threshold=0.01):
    """ estimating initial atlas using SoA method based on linear combinations

    :param [np.array] imgs: list of input images
    :param int nb_patterns: max number of estimated atlases
    :param int nb_iter: max number of iterations
    :return np.array: estimated atlas

    >>> np.random.seed(0)
    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 9 + [[0, 0, 1]] * 9 + [[0, 1, 1]] * 9)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> initialise_atlas_sparse_pca(imgs, 2)
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
        estimator = DictionaryLearning(n_components=nb_patterns,
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
    except:
        logging.warning('CRASH - initialise_atlas_dict_learn')
        logging.warning(traceback.format_exc())
        atlas = np.zeros(imgs[0].shape, dtype=int)
    return atlas


def initialise_atlas_deform_original(atlas, coef=0.5, grid_size=(20, 20),
                                     rnd_seed=None):
    """ take the orginal atlas and use geometrical deformation
    to generate new deformed atlas

    :param np.array<height, width> atlas:
    :return: np.array<height, width>

    >>> img = np.zeros((8, 12))
    >>> img[:3, 1:5] = 1
    >>> img[3:7, 6:12] = 2
    >>> initialise_atlas_deform_original(img, 0.1, (5, 5), rnd_seed=0)
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
    res = data.image_deform_elastic(atlas, coef, grid_size, rand_seed=rnd_seed)
    return np.array(res, dtype=np.int)


def reconstruct_samples(atlas, w_bins):
    """ create reconstruction of binary images according given atlas and weights

    :param np.array<height, width> atlas: input atlas
    :param np.array<nb_imgs, nb_lbs> w_bins:
    :return: [np.array<height, width>]

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
    # w_bins = np.array(weights)
    w_bin_ext = np.append(np.zeros((w_bins.shape[0], 1)), w_bins, axis=1)
    imgs = [None] * w_bins.shape[0]
    for i, w in enumerate(w_bin_ext):
        imgs[i] = np.asarray(w)[np.asarray(atlas, dtype=int)]
        imgs[i] = imgs[i].astype(atlas.dtype)
        assert atlas.shape == imgs[i].shape
    return imgs


def prototype_new_pattern(imgs, imgs_reconst, diffs, atlas,
                          ptn_compact=REINIT_PATTERN_COMPACT, thr_prob=0.5):
    """ estimate new pattern that occurs in input images and is not cover
    by any label in actual atlas, remove collision with actual atlas

    :param [np.array<height, width>] imgs: list of input images
    :param np.array<height, width> atlas:
    :param diffs: [int] list of differences among input and reconstruct images
    :return: np.array<height, width> binary single pattern

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
    """
    id_max = np.argmax(diffs)
    # im_diff = np.logical_and(imgs[id_max] == True, imgs_reconst[id_max] == False)
    # take just positive differences
    im_diff = (imgs[id_max] - imgs_reconst[id_max]) > thr_prob
    if ptn_compact:  # WaterShade
        logging.debug('.. reinit pattern using WaterShade')
        # im_diff = morphology.opening(im_diff, morphology.disk(3))
        # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
        dist = ndi.distance_transform_edt(im_diff)
        try:
            peaks = detect_peaks(dist)
            labels = morphology.watershed(-dist, peaks, mask=im_diff)
        except:
            logging.warning(traceback.format_exc())
            labels = None
    else:
        logging.debug('.. reinit pattern as major component')
        im_diff = morphology.closing(im_diff, morphology.disk(1))
        labels = None
    # find largest connected component
    img_ptn = data.extract_image_largest_element(im_diff, labels)
    # ptn_size = np.sum(ptn) / float(np.product(ptn.shape))
    # if ptn_size < 0.01:
    #     logging.debug('new patterns was too small %f', ptn_size)
    #     ptn = data.extract_image_largest_element(im_diff)
    img_ptn = (img_ptn == True)
    # img_ptn = np.logical_and(img_ptn == True, atlas == 0)
    return img_ptn


def insert_new_pattern(imgs, imgs_reconst, atlas, label,
                       ptn_compact=REINIT_PATTERN_COMPACT):
    """ with respect to atlas empty spots inset new patterns

    :param [np.array<height, width>] imgs: list of input images
    :param [np.array<height, width>] imgs_reconst:
    :param np.array<height, width> atlas:
    :param int label:
    :return: np.array<height, width> updated atlas

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
    # count just positive difference
    diffs = [np.sum(np.abs(im - im_rc)) for im, im_rc in zip(imgs, imgs_reconst)]
    im_ptn = prototype_new_pattern(imgs, imgs_reconst, diffs, atlas, ptn_compact)
    # logging.debug('new im_ptn: {}'.format(np.sum(im_ptn) / np.prod(im_ptn.shape)))
    # plt.imshow(im_ptn), plt.title('im_ptn'), plt.show()
    atlas[im_ptn == True] = label
    logging.debug('area of new pattern %i is %i', label, np.sum(atlas == label))
    return atlas


def reinit_atlas_likely_patterns(imgs, w_bins, atlas, label_max=None,
                                 ptn_compact=REINIT_PATTERN_COMPACT):
    """ walk and find all all free labels and try to reinit them by new patterns

    :param int label_max:
    :param [np.array<height, width>] imgs: list of input images
    :param np.array<nb_imgs, nb_lbs> w_bins:
    :param np.array<height, width> atlas:
    :return: np.array<height, width>, np.array<nb_imgs, nb_lbs>

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
    """
    # find empty patterns
    # assert len(imgs) == w_bins.shape[1], \
    #     'images %i and weigts %i' % (len(imgs), w_bins.shape[1])
    if label_max is None:
        label_max = max(np.max(atlas), w_bins.shape[1])
    else:
        logging.debug('compare w_bin %s to max %i', repr(w_bins.shape), label_max)
        for i in range(w_bins.shape[1], label_max):
            logging.debug('adding disappeared weigh column %i', i)
            w_bins = np.append(w_bins, np.zeros((w_bins.shape[0], 1)), axis=1)
    # w_bin_ext = np.append(np.zeros((w_bins.shape[0], 1)), w_bins, axis=1)
    # logging.debug('IN > sum over weights: %s', repr(np.sum(w_bin_ext, axis=0)))
    # add one while indexes does not cover 0 - bg
    logging.debug('total nb labels: %i', label_max)
    atlas_new = atlas.copy()
    for label in range(1, label_max + 1):
        w_index = label - 1
        w_sum = np.sum(w_bins[:, w_index])
        logging.debug('reinit. label: %i with weight sum %i', label, w_sum)
        if w_sum > 0:
            continue
        imgs_reconst = reconstruct_samples(atlas_new, w_bins)
        atlas_new = insert_new_pattern(imgs, imgs_reconst, atlas_new, label,
                                       ptn_compact)
        # logging.debug('w_bins before: %i', np.sum(w_bins[:, w_index]))
        # BE AWARE OF THIS CONSTANT, it can caused that there are weight even
        # they should not be which lead to have high unary for atlas estimation
        lim_repopulate = 1. / label_max
        w_bins[:, w_index] = ptn_weight.weights_label_atlas_overlap_threshold(
                                        imgs, atlas_new, label, lim_repopulate)
        logging.debug('w_bins after: %i', np.sum(w_bins[:, w_index]))
    return atlas_new, w_bins


def atlas_split_indep_ptn(atlas, label_max):
    """ split  independent patterns labeled equally

    :param np.array<height, width> atlas:
    :param int label_max:
    :return: np.array<height, width>

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
    logging.debug('list of all areas %s', repr([np.sum(p) for p in patterns]))
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
    logging.debug('atlas unique %s', repr(np.unique(atlas_new)))
    return atlas_new
