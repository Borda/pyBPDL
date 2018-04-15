"""
Estimating pattern weight vector for each image

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

# from __future__ import absolute_import
import logging

import numpy as np


def initialise_weights_random(nb_imgs, nb_patterns, ratio_select=0.2, rand_seed=None):
    """
    :param int nb_imgs: number of all images
    :param int nb_patterns: number of all available labels
    :param float ratio_select: number <0, 1> defining how many should be set on,
        1 means all and 0 means none
    :param rand_seed: random initialization
    :return ndarray: np.array<nb_imgs, nb_labels>

    >>> initialise_weights_random(5, 3, rand_seed=0)
    array([[ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.],
           [ 1.,  0.,  0.,  0.]])
    """
    logging.debug('initialise weights for %i images and %i labels '
                  'as random selection', nb_imgs, nb_patterns)
    np.random.seed(rand_seed)
    fuzzy = np.random.random((nb_imgs, nb_patterns + 1))
    weights = np.zeros_like(fuzzy)
    weights[fuzzy <= ratio_select] = 1
    for i, w in enumerate(weights):
        if np.sum(w) == 0:
            weights[i, np.random.randint(0, nb_patterns + 1)] = 1
    return weights


def convert_weights_binary2indexes(weights):
    """ convert binary matrix oof weights to list of indexes o activated ptns

    :param np.array<nb_imgs, nb_lbs> weights:
    :return [[int, ]]:

    >>> weights = np.array([[ 0,  0,  1,  0],
    ...                     [ 0,  0,  0,  1],
    ...                     [ 1,  0,  0,  0],
    ...                     [ 0,  0,  1,  1],
    ...                     [ 1,  0,  0,  0]])
    >>> convert_weights_binary2indexes(weights)
    [array([2]), array([3]), array([0]), array([2, 3]), array([0])]
    """
    logging.debug('convert binary weights %s to list of indexes with True',
                  repr(weights.shape))
    # if type(weights) is np.ndarray:  weights = weights.tolist()
    w_index = [None] * weights.shape[0]
    for i in range(weights.shape[0]):
        # find postions equal 1
        # vec = [j for j in range(weights.shape[1]) if weights[i,j]==1]
        vec = np.where(weights[i, :] == 1)[0]
        w_index[i] = vec
    # idxs =  np.where(weights == 1)
    # for i in range(weights.shape[0]):
    #     w_idx[i] = idxs[1][idxs[0]==i] +1
    return w_index


def weights_image_atlas_overlap_major(img, atlas):
    """
    :param ndarray img: image np.array<height, width>
    :param ndarray atlas: image np.array<height, width>
    :return [int]: [int] * nb_lbs of values {0, 1}

    >>> atlas = np.zeros((8, 10), dtype=int)
    >>> atlas[:3, 0:4] = 1
    >>> atlas[3:7, 5:10] = 2
    >>> img = np.array([0, 1, 0])[atlas]
    >>> weights_image_atlas_overlap_major(img, atlas)
    [1, 0]
    >>> img = [[0.46, 0.62, 0.62, 0.46, 0.2,  0.04, 0.01, 0.0,  0.0,  0.0],
    ...        [0.44, 0.59, 0.59, 0.44, 0.2,  0.06, 0.04, 0.04, 0.04, 0.04],
    ...        [0.33, 0.44, 0.44, 0.34, 0.2,  0.17, 0.19, 0.2,  0.2,  0.2],
    ...        [0.14, 0.19, 0.19, 0.17, 0.2,  0.34, 0.44, 0.46, 0.47, 0.47],
    ...        [0.03, 0.04, 0.04, 0.06, 0.2,  0.44, 0.59, 0.62, 0.62, 0.62],
    ...        [0.0,  0.0,  0.01, 0.04, 0.19, 0.44, 0.59, 0.62, 0.62, 0.62],
    ...        [0.0,  0.0,  0.0,  0.03, 0.14, 0.33, 0.44, 0.46, 0.47, 0.47],
    ...        [0.0,  0.0,  0.0,  0.01, 0.06, 0.14, 0.19, 0.2,  0.2,  0.2]]
    >>> weights_image_atlas_overlap_major(np.array(img), atlas)
    [0, 1]
    """
    # logging.debug('weights input image according given atlas')
    weights = weights_image_atlas_overlap_threshold(img, atlas, 1.)
    return weights


def weights_image_atlas_overlap_partial(img, atlas):
    """
    :param ndarray img: image np.array<height, width>
    :param ndarray atlas: image np.array<height, width>
    :return [int]: [int] * nb_lbs of values {0, 1}

    >>> atlas = np.zeros((8, 10), dtype=int)
    >>> atlas[:3, 0:4] = 1
    >>> atlas[3:7, 5:10] = 2
    >>> img = np.array([0, 1, 0])[atlas]
    >>> weights_image_atlas_overlap_partial(img, atlas)
    [1, 0]
    >>> img = [[0.46, 0.62, 0.62, 0.46, 0.2,  0.04, 0.01, 0.0,  0.0,  0.0],
    ...        [0.44, 0.59, 0.59, 0.44, 0.2,  0.06, 0.04, 0.04, 0.04, 0.04],
    ...        [0.33, 0.44, 0.44, 0.34, 0.2,  0.17, 0.19, 0.2,  0.2,  0.2],
    ...        [0.14, 0.19, 0.19, 0.17, 0.2,  0.34, 0.44, 0.46, 0.47, 0.47],
    ...        [0.03, 0.04, 0.04, 0.06, 0.2,  0.44, 0.59, 0.62, 0.62, 0.62],
    ...        [0.0,  0.0,  0.01, 0.04, 0.19, 0.44, 0.59, 0.62, 0.62, 0.62],
    ...        [0.0,  0.0,  0.0,  0.03, 0.14, 0.33, 0.44, 0.46, 0.47, 0.47],
    ...        [0.0,  0.0,  0.0,  0.01, 0.06, 0.14, 0.19, 0.2,  0.2,  0.2]]
    >>> weights_image_atlas_overlap_partial(np.array(img), atlas)
    [1, 1]
    """
    # logging.debug('weights input image according given atlas')
    labels = np.unique(atlas).tolist()
    threshold = 1. / (np.max(labels) + 1)
    weights = weights_image_atlas_overlap_threshold(img, atlas, threshold)
    return weights


def weights_image_atlas_overlap_threshold(img, atlas, threshold=1.):
    """ estimate what patterns are activated  with given atlas and input image
    compute overlap matrix and eval nr of overlapping and non pixels and threshold

    :param ndarray img: image np.array<height, width>
    :param ndarray atlas: image np.array<height, width>
    :param float threshold: represent the ration between overlapping and non pixels
    :return [int]: [int] * nb_lbs of values {0, 1}
    """
    # logging.debug('weights input image according given atlas')
    # simple weight
    labels = np.unique(atlas).tolist()
    # logging.debug('weights image by atlas with labels: {}'.format(lbs))
    if 0 in labels:
        labels.remove(0)
    weight = [0] * int(np.max(atlas))
    for lb in labels:
        mask = (atlas == lb)
        total = np.sum(mask)
        nequal = np.sum(np.abs(1. - img[mask]))
        score = total / (nequal + 1e-9) - 1.
        # equal = np.sum(img[atlas == lb])
        # score = equal / float(total)
        if score >= threshold:
            weight[lb - 1] = 1
    return weight


def weights_label_atlas_overlap_threshold(imgs, atlas, label, threshold=1.):
    """ estimate what patterns are activated  with given atlas and input image
    compute overlap matrix and eval nr of overlapping and non pixels and threshold

    :param [ndarray] imgs: list of images np.array<height, width>
    :param ndarray atlas: image np.array<height, width>
    :param int label:
    :param float threshold: represent the ration between overlapping and non pixels
    :return ndarray: np.array<nb_imgs> of values {0, 1}

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> atlas[atlas == 2] = 0
    >>> weights_label_atlas_overlap_threshold(imgs, atlas, 2)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    """
    weight = [0] * len(imgs)
    for i, img in enumerate(imgs):
        mask = atlas == label
        equal = np.sum(img[mask])
        total = np.sum(mask)
        score = equal / (total + 1e-9)
        if score >= threshold:
            weight[i] = 1
    return np.array(weight)
