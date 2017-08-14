"""
The main module for Atomic pattern dictionary, jjoiningthe atlas estimation
and computing the encoding / weights

Copyright (C) 2015-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import
import os
import time
import logging
import traceback

# to suppress all visual, has to be on the beginning
import matplotlib
if os.environ.get('DISPLAY','') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation as sk_image
# using https://github.com/Borda/pyGCO
from gco import cut_general_graph, cut_grid_graph_simple

import bpdl.pattern_atlas as ptn_dict
import bpdl.pattern_weights as ptn_weight
import bpdl.metric_similarity as sim_metric
import bpdl.dataset_utils as gen_data

UNARY_BACKGROUND = 1
NB_GRAPH_CUT_ITER = 5
TEMPLATE_NAME_ATLAS = 'APDL_{}_{}_iter_{:04d}'

# TRY: init: spatial clustering
# TRY: init: use ICA
# TRY: init: greedy


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
    assert len(imgs) == weights.shape[0]
    pott_sum = np.zeros(imgs[0].shape + (nb_lbs,))
    # extenf the weights by background value 0
    weights_ext = np.append(np.zeros((weights.shape[0], 1)), weights, axis=1)
    # logging.debug(weights_ext)
    imgs = np.array(imgs)
    logging.debug('DIMS potts: %s, imgs %s, w_bin: %s',
                  repr(pott_sum.shape), repr(imgs.shape), repr(weights_ext.shape))
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
    :param [np.array<height, width>] imgs: list of input images
    :param np.array<nb_imgs, nb_lbs> weights: matrix composed from wight vectors
    :return: np.array<height, width, nb_lbs>

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
    w_idx = ptn_weight.convert_weights_binary2indexes(ptn_weights)
    nb_lbs = ptn_weights.shape[1] + 1
    assert len(imgs) == len(w_idx)
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


def edges_in_image_plane(im_size):
    """ create list of edges for uniform image plane

    :param (int, int) im_size: size of image
    :return [[int, int], ]:

    >>> edges_in_image_plane([2, 3])
    array([[0, 1],
           [1, 2],
           [3, 4],
           [4, 5],
           [0, 3],
           [1, 4],
           [2, 5]])
    """
    idxs = np.arange(np.product(im_size))
    idxs = idxs.reshape(im_size)
    # logging.debug(idxs)
    eA = idxs[:, :-1].ravel().tolist() + idxs[:-1, :].ravel().tolist()
    eB = idxs[:, 1:].ravel().tolist() + idxs[1:, :].ravel().tolist()
    edges = np.array([eA, eB]).transpose()
    logging.debug('edges for image plane are shape {}'.format(edges.shape))
    return edges


def estimate_atlas_graphcut_simple(imgs, ptn_weights, coef=1.):
    """ run the graphcut to estimate atlas from computed unary terms
    source: https://github.com/yujiali/pyGCO

    :param [np.array<height, width>] imgs: list of input binary images
    :param np.array<nb_imgs, nb_lbs> ptn_weights: binary ptn selection
    :param float coef: coefficient for graphcut
    :return:

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> estimate_atlas_graphcut_simple(imgs, luts[:, 1:]).astype(int)
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('estimate atlas via GraphCut from Potts model')
    if ptn_weights.shape[1] <= 1:
        logging.warning('nothing to do for single label')
        labels = np.zeros(imgs[0].shape)
        return labels

    labeling_sum = compute_positive_cost_images_weights(imgs, ptn_weights)
    unary_cost = np.array(-1 * labeling_sum , dtype=np.int32)
    logging.debug('graph unaries potentials %s: \n %s', repr(unary_cost.shape),
                                        repr(np.histogram(unary_cost, bins=10)))
    # original and the right way..
    pairwise = (1 - np.eye(labeling_sum.shape[-1])) * coef
    pairwise_cost = np.array(pairwise , dtype=np.int32)
    logging.debug('graph pairwise coefs %s', repr(pairwise_cost.shape))
    # run GraphCut
    try:
        labels = cut_grid_graph_simple(unary_cost, pairwise_cost,
                                       algorithm='expansion')
    except:
        logging.warning(traceback.format_exc())
        labels = np.argmin(unary_cost, axis=1)
    # reshape labels
    labels = labels.reshape(labeling_sum.shape[:2])
    logging.debug('resulting labelling %s: \n %s', repr(labels.shape), repr(labels))
    return labels


def estimate_atlas_graphcut_general(imgs, ptn_weights, coef=0., init_atlas=None):
    """ run the graphcut on the unary costs with specific pairwise cost
    source: https://github.com/yujiali/pyGCO

    :param [np.array<height, width>] imgs: list of input binary images
    :param np.array<nb_imgs, nb_lbs> encoding: binary ptn selection
    :param float coef: coefficient for graphcut
    :param np.array<nb_seg, 1> init_labels: init labeling
        while None it take the arg ming of the unary costs
    :return np.array<nb_seg, 1>:

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> estimate_atlas_graphcut_general(imgs, luts[:, 1:]).astype(int)
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('estimate atlas via GraphCut from Potts model')
    if ptn_weights.shape[1] <= 1:
        logging.warning('nothing to do for single label')
        labels = np.zeros(imgs[0].shape)
        return labels

    u_cost = compute_relative_penalty_images_weights(imgs, ptn_weights)
    # u_cost = 1. / (labelingSum +1)
    unary_cost = np.array(u_cost , dtype=np.float64)
    unary_cost = unary_cost.reshape(-1, u_cost.shape[-1])
    logging.debug('graph unaries potentials %s: \n %s', repr(unary_cost.shape),
                  repr(np.histogram(unary_cost, bins=10)))

    edges = edges_in_image_plane(u_cost.shape[:2])
    logging.debug('edges for image plane are shape %s', format(edges.shape))
    edge_weights = np.ones(edges.shape[0])
    logging.debug('edges weights are shape %s', repr(edge_weights.shape))

    # original and the right way...
    pairwise = (1 - np.eye(u_cost.shape[-1])) * coef
    pairwise_cost = np.array(pairwise , dtype=np.float64)
    logging.debug('graph pairwise coefs %s', repr(pairwise_cost.shape))

    if init_atlas is None:
        init_labels = np.argmin(unary_cost, axis=1)
    else:
        init_labels = init_atlas.ravel()
    logging.debug('graph initial labels of shape %s', repr(init_labels.shape))

    # run GraphCut
    try:
        labels = cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost,
                                   algorithm='expansion', init_labels=init_labels,
                                   n_iter=NB_GRAPH_CUT_ITER)
    except:
        logging.warning(traceback.format_exc())
        labels = np.argmin(unary_cost, axis=1)
    # reshape labels
    labels = labels.reshape(u_cost.shape[:2])
    logging.debug('resulting labelling %s of %s', repr(labels.shape),
                  np.unique(labels).tolist())
    return labels


def export_visualization_image(img, idx, out_dir, prefix='debug', name='',
                               ration=None, labels=('', '')):
    """ export visualisation as an image with some special desc.

    :param np.array<height, width> img:
    :param int idx: iteration to be shown in the img name
    :param str out_dir: path to the resulting folder
    :param str prefix:
    :param str name: name of this particular visual
    :param str ration: mainly for  weights to ne stretched
    :param [str, str] labels: labels for axis

    # CRASH: TclError: no display name and no $DISPLAY environment variable
    # >>> img = np.random.random((50, 50))
    # >>> path_fig = export_visualization_image(img, 0, '.')
    # >>> os.path.exists(path_fig)
    # True
    # >>> os.remove(path_fig)
    """
    plt.ioff()
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='none', aspect=ration)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    name_fig = TEMPLATE_NAME_ATLAS.format(prefix, name, idx)
    path_fig = os.path.join(out_dir, name_fig + '.png')
    logging.debug('.. export Visualization as "%s"', path_fig)
    fig.savefig(path_fig, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return path_fig


def export_visual_atlas(i, out_dir, atlas=None, prefix='debug'):
    """ export the atlas and/or weights to results directory

    :param int i: iteration to be shown in the img name
    :param str out_dir: path to the resulting folder
    :param np.array<height, width> atlas:
    :param np.array<nb_imgs, nb_lbs> weights:
    :param str prefix:

    """
    if logging.getLogger().getEffectiveLevel() < logging.DEBUG:
        return
    if out_dir is None or not os.path.exists(out_dir):
        logging.debug('results path "%s" does not exist', out_dir)
        return None
    if atlas is not None:
        # export_visualization_image(atlas, i, out_dir, prefix, 'atlas',
        #                            labels=['X', 'Y'])
        n_img = 'APDL_{}_atlas_iter_{:04d}'.format(prefix, i)
        gen_data.export_image(out_dir, atlas, n_img)
    # if weights is not None:
    #     export_visualization_image(weights, i, out_dir, prefix, 'weights',
    #                                'auto', ['patterns', 'images'])


def bpdl_initialisation(imgs, init_atlas, init_weights, out_dir, out_prefix,
                        rnd_seed=None):
    """ more complex initialisation depending on inputs

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> init_atlas:
    :param np.array<nb_imgs, nb_lbs> init_weights:
    :param str out_prefix:
    :param str out_dir: path to the results directory
    :return: np.array<height, width>, np.array<nb_imgs, nb_lbs>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> w_bins = luts[:, 1:]
    >>> init_atlas, init_w_bins = bpdl_initialisation(imgs, init_atlas=None,
    ...        init_weights=None, out_dir=None, out_prefix='', rnd_seed=0)
    >>> init_atlas
    array([[3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2],
           [3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2],
           [3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2]])
    """
    if init_weights is not None and init_atlas is None:
        logging.debug('... initialise Atlas from w_bins')
        init_atlas = estimate_atlas_graphcut_general(imgs, init_weights, 0.)
        export_visual_atlas(0, out_dir, init_atlas, out_prefix)
    if init_atlas is None:
        nb_patterns = int(np.sqrt(len(imgs)))
        logging.debug('... initialise Atlas with ')
        # IDEA: find better way of initialisation
        init_atlas = ptn_dict.initialise_atlas_mosaic(
            imgs[0].shape, nb_patterns, rnd_seed=rnd_seed)
        export_visual_atlas(0, out_dir, init_atlas, out_prefix)

    atlas = init_atlas
    w_bins = init_weights
    if len(np.unique(atlas)) == 1:
        logging.error('the init. atlas does not contain any label... %s',
                      repr(np.unique(atlas)))
    export_visual_atlas(0, out_dir, atlas, out_prefix)
    return atlas, w_bins


def bpdl_update_weights(imgs, atlas, overlap_major=False):
    """ single iteration of the block coordinate descent algo

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> atlas:
    :return: np.array<nb_imgs, nb_lbs>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> bpdl_update_weights(imgs, atlas)
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
    # update w_bins
    logging.debug('... perform pattern weights')
    if overlap_major:
        w_bins = [ptn_weight.weights_image_atlas_overlap_major(img, atlas)
                  for img in imgs]
    else:
        w_bins = [ptn_weight.weights_image_atlas_overlap_partial(img, atlas)
                  for img in imgs]
    # add once for patterns that are not used at all
    # w_bins = ptn_weight.fill_empty_patterns(np.array(w_bins))
    return np.array(w_bins)


def bpdl_update_atlas(imgs, atlas, w_bins, label_max, gc_coef, gc_reinit, ptn_split):
    """ single iteration of the block coordinate descent algo

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> atlas:
    :param np.array<nb_imgs, nb_lbs> w_bins:
    :param int label_max:
    :param float gc_coef: graph cut regularisation
    :param bool gc_reinit: weather use atlas from previous step as init for act.
    :param bool ptn_split:
    :return: np.array<height, width>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> bpdl_update_atlas(imgs, atlas, luts[:, 1:], 2, gc_coef=0.,
    ...                   gc_reinit=False, ptn_split=False)
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    """
    if np.sum(w_bins) == 0:
        logging.warning('the w_bins is empty... %s', repr(np.unique(atlas)))
    w_bins = np.array(w_bins)

    logging.debug('... perform Atlas estimation')
    if gc_reinit:
        atlas_new = estimate_atlas_graphcut_general(imgs, w_bins, gc_coef, atlas)
    else:
        atlas_new = estimate_atlas_graphcut_general(imgs, w_bins, gc_coef)

    if ptn_split:
        atlas_new = ptn_dict.atlas_split_indep_ptn(atlas_new, label_max)

    atlas_new = np.remainder(atlas_new, label_max + 1)
    return atlas_new


def bpdl_pipe_atlas_learning_ptn_weights(imgs, init_atlas=None, init_weights=None,
                                         gc_coef=0.0, tol=1e-3, max_iter=25,
                                         gc_reinit=True, ptn_split=True,
                                         overlap_major=False, ptn_compact=True,
                                         out_prefix='debug', out_dir=''):
    """ the experiments_synthetic pipeline for block coordinate descent
    algo with graphcut...

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> init_atlas:
    :param np.array<nb_imgs, nb_lbs> init_weights:
    :param float gc_coef: graph cut regularisation
    :param float tol: stop if the diff between two conseq steps
        is less then this given threshold. eg for -1 never until max nb iters
    :param int max_iter: max namber of iteration
    :param bool gc_reinit: wether use atlas from previous step as init for act.
    :param str out_prefix:
    :param str out_dir: path to the results directory
    :return: np.array<height, width>, np.array<nb_imgs, nb_lbs>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> w_bins = luts[:, 1:]
    >>> init_atlas = ptn_dict.initialise_atlas_mosaic(atlas.shape,
    ...                                               nb_patterns=2, rnd_seed=0)
    >>> init_atlas
    array([[2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]])
    >>> bpdl_atlas, bpdl_w_bins = bpdl_pipe_atlas_learning_ptn_weights(imgs, init_atlas)
    >>> bpdl_atlas
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> bpdl_w_bins
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
    logging.debug('compute an Atlas and weights for %i images...', len(imgs))
    assert len(imgs) >= 0
    if logging.getLogger().getEffectiveLevel()==logging.DEBUG:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    # assert initAtlas is not None or type(max_nb_lbs)==int
    # initialise
    label_max = np.max(init_atlas)
    logging.debug('max nb labels set: %i', label_max)
    atlas, w_bins = bpdl_initialisation(imgs, init_atlas, init_weights,
                                        out_dir, out_prefix)
    list_crit = []

    for iter in range(max_iter):
        if len(np.unique(atlas)) == 1:
            logging.warning('.. iter: %i, no labels in the atlas %s', iter,
                            repr(np.unique(atlas).tolist()))
        w_bins = bpdl_update_weights(imgs, atlas, overlap_major)
        atlas_reinit, w_bins = ptn_dict.reinit_atlas_likely_patterns(
                                    imgs, w_bins, atlas, label_max, ptn_compact)
        atlas_new = bpdl_update_atlas(imgs, atlas_reinit, w_bins, label_max,
                                      gc_coef, gc_reinit, ptn_split)

        step_diff = sim_metric.compare_atlas_adjusted_rand(atlas, atlas_new)
        # step_diff = np.sum(abs(atlas - atlas_new)) / float(np.product(atlas.shape))
        list_crit.append(step_diff)
        atlas = sk_image.relabel_sequential(atlas_new)[0]

        logging.debug('-> iter. #%i with Atlas diff %f', (iter + 1), step_diff)
        export_visual_atlas(iter + 1, out_dir, atlas, out_prefix)

        # stopping criterion
        if step_diff <= tol and len(np.unique(atlas)) > 1:
            logging.debug('>> exit while the atlas diff %f is smaller then %f',
                          step_diff, tol)
            break
    logging.info('APDL: terminated with iter %i / %i and step diff %f <? %f',
                 iter, max_iter, step_diff, tol)
    logging.debug('criterion evolved:\n %s', repr(list_crit))
    # atlas = sk_image.relabel_sequential(atlas)[0]
    w_bins = [ptn_weight.weights_image_atlas_overlap_major(img, atlas) for img in imgs]
    return atlas, np.array(w_bins)


def test_simple_show_case(atlas, imgs, ws):
    """ simple experiment

    >>> atlas = gen_data.create_simple_atlas()
    >>> imgs = gen_data.create_sample_images(atlas)
    >>> ws=([1, 0, 0], [0, 1, 1], [0, 0, 1])
    >>> res, fig = test_simple_show_case(atlas, imgs, ws)
    >>> res.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> fig  # doctest: +ELLIPSIS
    <matplotlib.figure.Figure object at ...>
    """
    # implement simple case just with 2 images and 2/3 classes in atlas
    fig, axarr = plt.subplots(2, len(imgs) + 2)

    plt.title('w: {}'.format(repr(ws)))
    axarr[0, 0].set_title('atlas')
    cm = plt.cm.get_cmap('jet', len(np.unique(atlas)))
    im = axarr[0, 0].imshow(atlas, cmap=cm, interpolation='nearest')
    fig.colorbar(im, ax=axarr[0, 0])

    for i, (img, w) in enumerate(zip(imgs, ws)):
        axarr[0, i + 1].set_title('w:{}'.format(w))
        axarr[0, i + 1].imshow(img, cmap='gray', interpolation='nearest')

    t = time.time()
    uc = compute_relative_penalty_images_weights(imgs, np.array(ws))
    logging.info('elapsed TIME: %s', repr(time.time() - t))
    res = estimate_atlas_graphcut_general(imgs, np.array(ws), 0.)

    axarr[0, -1].set_title('result')
    im = axarr[0, -1].imshow(res, cmap=cm, interpolation='nearest')
    fig.colorbar(im, ax=axarr[0, -1])
    uc = uc.reshape(atlas.shape + uc.shape[2:])

    # logging.debug(ws)
    for i in range(uc.shape[2]):
        axarr[1, i].set_title('cost lb #{}'.format(i))
        im = axarr[1, i].imshow(uc[:, :, i], vmin=0, vmax=1,
                                interpolation='nearest')
        fig.colorbar(im, ax=axarr[1, i])
    # logging.debug(uc)
    return res, fig
