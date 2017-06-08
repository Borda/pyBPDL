"""
Estimating the pattern dictionary module

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

from scipy import ndimage
from skimage import morphology, feature, filters
from scipy import ndimage as ndi
import numpy as np

import dataset_utils as data
import pattern_weights as ptn_weight

REINIT_PATTERN_COMPACT = True


# TODO: init: Otsu threshold on sum over all input images -> WaterShade on distance
# TODO: init: sum over all input images and use it negative as distance for WaterShade


def initialise_atlas_random(im_size, label_max):
    """ initialise atlas with random labels

    :param (int, int) im_size: size of image
    :param int label_max: number of labels
    :return: np.array<height, width>
    """
    logging.debug('initialise atlas %s as random labeling', repr(im_size))
    nb_labels = label_max + 1
    np.random.seed()  # reinit seed to have random samples even in the same time
    img_init = np.random.randint(1, nb_labels, im_size)
    return np.array(img_init, dtype=np.int)


def initialise_atlas_mosaic(im_size, nb_labels, coef=1.):
    """ generate grids texture and into each rectangle plase a label,
    each row contains all labels (permutation)

    :param (int, int) im_size: size of image
    :param int nb_labels: number of labels
    :return: np.array<height, width>
    """
    logging.debug('initialise atlas %s as grid labeling', repr(im_size))
    max_label = int(nb_labels * coef)
    np.random.seed()  # reinit seed to have random samples even in the same time
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
    img_init = np.remainder(img_init, nb_labels)
    return np.array(img_init, dtype=np.int)


def initialise_atlas_otsu_watershed_2d(imgs, nb_labels, bg='none'):
    """ do some simple operations to get better initialisation
    1] sum over all images, 2] Otsu thresholding, 3] watershed

    :param [np.array<height, width>] imgs:
    :param int nb_labels: number of labels
    :param str bg: set weather the Otsu backround sould be filled randomly
    :return: np.array<height, width>
    """
    logging.debug('initialise atlas for %i labels from %i images of shape %s '
                  'with Otsu-Watershed', nb_labels, len(imgs), repr(imgs[0].shape))
    img_sum = np.sum(np.asarray(imgs), axis=0) / float(len(imgs))
    img_gauss = filters.gaussian_filter(img_sum, 1)
    # http://scikit-image.org/docs/dev/auto_examples/plot_otsu.html
    thresh = filters.threshold_otsu(img_gauss)
    img_otsu = (img_gauss >= thresh)
    # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
    img_dist = ndi.distance_transform_edt(img_otsu)
    local_maxi = feature.peak_local_max(img_dist, labels=img_otsu,
                                        footprint=np.ones((2, 2)))
    seeds = np.zeros_like(img_sum)
    seeds[local_maxi[:,0], local_maxi[:,1]] = range(1, len(local_maxi) + 1)
    labels = morphology.watershed(-img_dist, seeds)
    img_init = np.remainder(labels, nb_labels)
    if bg == 'rand':
        # add random labels on the potential backgound
        img_rand = np.random.randint(1, nb_labels, img_sum.shape)
        img_init[img_otsu == 0] = img_rand[img_otsu == 0]
    return img_init.astype(np.int)


def initialise_atlas_gauss_watershed_2d(imgs, nb_labels):
    """ do some simple operations to get better initialisation
    1] sum over all images, 2]watershed

    :param [np.array<height, width>] imgs: list of input images
    :param int nb_labels:
    :return: np.array<height, width>
    """
    logging.debug('initialise atlas for %i labels from %i images of shape %s '
                  'with Gauss-Watershed', nb_labels, len(imgs), repr(imgs[0].shape))
    img_sum = np.sum(np.asarray(imgs), axis=0) / float(len(imgs))
    img_gauss = filters.gaussian_filter(img_sum, 1)
    local_maxi = feature.peak_local_max(img_gauss, footprint=np.ones((2, 2)))
    seeds = np.zeros_like(img_sum)
    seeds[local_maxi[:,0], local_maxi[:,1]] = range(1, len(local_maxi) + 1)
    # http://scikit-image.org/docs/dev/auto_examples/plot_watershed.html
    labels = morphology.watershed(-img_gauss, seeds) # , mask=im_diff
    img_init = np.remainder(labels, nb_labels)
    return img_init.astype(np.int)


def initialise_atlas_deform_original(atlas):
    """take the orginal atlas and use geometrical deformation
    to generate new deformed atlas

    :param np.array<height, width> atlas:
    :return: np.array<height, width>
    """
    logging.debug('initialise atlas by deforming original one')
    res = data.image_deform_elastic(atlas)
    return np.array(res, dtype=np.int)


def reconstruct_samples(atlas, w_bins):
    """ create reconstruction of binary images according given atlas and weights

    :param np.array<height, width> atlas: input atlas
    :param np.array<nb_imgs, nb_lbs> w_bins:
    :return: [np.array<height, width>]
    """
    # w_bins = np.array(weights)
    w_bin_ext = np.append(np.zeros((w_bins.shape[0], 1)), w_bins, axis=1)
    imgs = [None] * w_bins.shape[0]
    for i, w in enumerate(w_bin_ext):
        imgs[i] = np.asarray(w)[np.asarray(atlas)]
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
        local_maxi = feature.peak_local_max(dist, indices=False, labels=im_diff,
                                            footprint=np.ones((3, 3)))
        labels = morphology.watershed(-dist, ndi.label(local_maxi)[0], mask=im_diff)
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
    """
    # count just positive difference
    diffs = [np.sum((im - im_rc) > 0) for im, im_rc in zip(imgs, imgs_reconst)]
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
    """
    # find empty patterns
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
        lim_repopulate = 1e3 / np.prod(atlas_new.shape)
        w_bins[:, w_index] = ptn_weight.weights_label_atlas_overlap_threshold(
                                        imgs, atlas_new, label, lim_repopulate)
        logging.debug('w_bins after: %i', np.sum(w_bins[:, w_index]))
    return atlas_new, w_bins


def atlas_split_indep_ptn(atlas, label_max):
    """ split  independent patterns labeled equally

    :param np.array<height, width> atlas:
    :param int label_max:
    :return: np.array<height, width>
    """
    patterns = []
    for label in np.unique(atlas):
        labeled, nb_objects = ndimage.label(atlas == label)
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
