"""
Estimating pattern weight vector for each image

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np


def initialise_weights_random(nb_imgs, nb_labels, ratio_select=0.2):
    """
    :param int nb_imgs: number of all images
    :param int nb_labels: number of all availablelabels
    :param float ratio_select: number <0, 1> defining how many should be set on,
        1 means all and 0 means none
    :return: np.array<nb_imgs, nb_labels>
    """
    logging.debug('initialise weights for %i images and %i labels '
                 'as random selection', nb_imgs, nb_labels)
    prob = np.random.random((nb_imgs, nb_labels))
    weights = np.zeros_like(prob)
    weights[prob <= ratio_select] = 1
    return weights


def convert_weights_binary2indexes(weights):
    """ convert binary matrix oof weights to list of indexes o activated ptns

    :param np.array<nb_imgs, nb_lbs> weights:
    :return: [[int, ]]
    """
    logging.debug('convert binary weights %s to list of indexes with True',
                 repr(weights.shape))
    # if type(weights) is np.ndarray:  weights = weights.tolist()
    w_index = [None] * weights.shape[0]
    for i in range(weights.shape[0]):
        # find postions equal 1
        # vec = [j for j in range(weights.shape[1]) if weights[i,j]==1]
        vec = np.where(weights[i, :] == 1)[0]
        w_index[i] = vec + 1
    # idxs =  np.where(weights == 1)
    # for i in range(weights.shape[0]):
    #     w_idx[i] = idxs[1][idxs[0]==i] +1
    return w_index


def weights_image_atlas_overlap_major(img, atlas):
    """
    :param np.array<height, width> img:
    :param np.array<height, width> atlas:
    :return: [int] * nb_lbs of values {0, 1}
    """
    # logging.debug('weights input image according given atlas')
    weights = weights_image_atlas_overlap_threshold(img, atlas, 1.)
    return weights


def weights_image_atlas_overlap_partial(img, atlas):
    """
    :param np.array<height, width> img:
    :param np.array<height, width> atlas:
    :return: [int] * nb_lbs of values {0, 1}
    """
    # logging.debug('weights input image according given atlas')
    labels = np.unique(atlas).tolist()
    weights = weights_image_atlas_overlap_threshold(img, atlas,
                                                    (1. / np.max(labels)))
    return weights


def weights_image_atlas_overlap_threshold(img, atlas, threshold=1.):
    """ estimate what patterns are activated  with given atlas and input image
    compute overlap matrix and eval nr of overlapping and non pixels and threshold

    :param np.array<height, width> img:
    :param np.array<height, width> atlas:
    :param float threshold: represent the ration between overlapping and non pixels
    :return: [int] * nb_lbs of values {0, 1}
    """
    # logging.debug('weights input image according given atlas')
    # simple weight
    labels = np.unique(atlas).tolist()
    # logging.debug('weights image by atlas with labels: {}'.format(lbs))
    if 0 in labels:
        labels.remove(0)
    weight = [0] * np.max(atlas)
    for lb in labels:
        total = np.sum(atlas == lb)
        nequal = np.sum(np.abs(1 - img[atlas == lb]))
        score = total / float(nequal) - 1.
        # equal = np.sum(img[atlas == lb])
        # score = equal / float(total)
        if score >= threshold:
            weight[lb - 1] = 1
    return weight


def weights_label_atlas_overlap_threshold(imgs, atlas, label, threshold=1.):
    """ estimate what patterns are activated  with given atlas and input image
    compute overlap matrix and eval nr of overlapping and non pixels and threshold

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> atlas:
    :param int label:
    :param float threshold: represent the ration between overlapping and non pixels
    :return: np.array<nb_imgs> of values {0, 1}
    """
    weight = [0] * len(imgs)
    for i, img in enumerate(imgs):
        equal = np.sum(img[atlas == label])
        total = np.sum(atlas == label)
        score = equal / float(total)
        if score >= threshold:
            weight[i] = 1
    return np.array(weight)

