"""
The main module for Atomic pattern dictionary, jjoiningthe atlas estimation
and computing the encoding / weights

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import skimage.segmentation as sk_image
from pygco import cut_general_graph, cut_grid_graph_simple

import pattern_disctionary as ptn_dict
import pattern_weights as ptn_weight
import metric_similarity as sim_metric
import dataset_utils as gen_data

UNARY_BACKGROUND = 1
NB_GRAPH_CUT_ITER = 5

# TRY: init: spatial clustering
# TRY: init: use ICA
# TRY: init: greedy


def compute_relative_penalty_images_weights(imgs, weights):
    """ compute the relative penalty for all pixel and cjsing each label
    on that particular position

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


def compute_positive_cost_images_weights(imgs, weights):
    """
    :param [np.array<height, width>] imgs: list of input images
    :param np.array<nb_imgs, nb_lbs> weights: matrix composed from wight vectors
    :return: np.array<height, width, nb_lbs>
    """
    # not using any more...
    logging.debug('compute unary cost from images and related weights')
    w_idx = ptn_weight.convert_weights_binary2indexes(weights)
    nb_lbs = weights.shape[1] + 1
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
    """
    idxs = np.arange(np.product(im_size))
    idxs = idxs.reshape(im_size)
    # logging.debug(idxs)
    eA = idxs[:, :-1].ravel().tolist() + idxs[:-1, :].ravel().tolist()
    eB = idxs[:, 1:].ravel().tolist() + idxs[1:, :].ravel().tolist()
    edges = np.array([eA, eB]).transpose()
    logging.debug('edges for image plane are shape {}'.format(edges.shape))
    return edges


def estimate_atlas_graphcut_simple(imgs, encoding, coef=1.):
    """ run the graphcut to estimate atlas from computed unary terms
    source: https://github.com/yujiali/pyGCO

    :param [np.array<height, width>] imgs: list of input binary images
    :param np.array<nb_imgs, nb_lbs> encoding: binary ptn selection
    :param float coef: coefficient for graphcut
    :return:
    """
    logging.debug('estimate atlas via GraphCut from Potts model')
    # source: https://github.com/Borda/pyGCO
    # sys.path.append(os.path.abspath(os.path.join('..', 'libs')))  # Add path to libs
    # from pyGCO.pygco import cut_grid_graph_simple

    labeling_sum = compute_positive_cost_images_weights(imgs, encoding)
    unary_cost = np.array(-1 * labeling_sum , dtype=np.int32)
    logging.debug('graph unaries potentials %s: \n %s', repr(unary_cost.shape),
                                        repr(np.histogram(unary_cost, bins=10)))
    # original and the right way..
    pairwise = (1 - np.eye(labeling_sum.shape[-1])) * coef
    pairwise_cost = np.array(pairwise , dtype=np.int32)
    logging.debug('graph pairwise coefs %s', repr(pairwise_cost.shape))
    # run GraphCut
    labels = cut_grid_graph_simple(unary_cost, pairwise_cost,
                                   algorithm='expansion')
    # reshape labels
    labels = labels.reshape(labeling_sum.shape[:2])
    logging.debug('resulting labelling %s: \n %s', repr(labels.shape), repr(labels))
    return labels


def estimate_atlas_graphcut_general(imgs, encoding, coef=0., init_atlas=None):
    """ run the graphcut on the unary costs with specific pairwise cost
    source: https://github.com/yujiali/pyGCO

    :param [np.array<height, width>] imgs: list of input binary images
    :param np.array<nb_imgs, nb_lbs> encoding: binary ptn selection
    :param float coef: coefficient for graphcut
    :param np.array<nb_seg, 1> init_labels: init labeling
        while None it take the arg ming of the unary costs
    :return np.array<nb_seg, 1>:
    """
    logging.debug('estimate atlas via GraphCut from Potts model')
    # source: https://github.com/Borda/pyGCO
    # sys.path.append(os.path.abspath(os.path.join('..', 'libs')))  # Add path to libs
    # from pyGCO.pygco import cut_general_graph

    u_cost = compute_relative_penalty_images_weights(imgs, encoding)
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
    logging.debug('graph initial labels %s', repr(init_labels.shape))

    # run GraphCut
    labels = cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost,
                               algorithm='expansion', init_labels=init_labels,
                               n_iter=NB_GRAPH_CUT_ITER)
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
    """
    fig = plt.figure()
    plt.imshow(img, interpolation='none', aspect=ration)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    n_fig = 'APDL_{}_{}_iter_{:04d}'.format(prefix, name, idx)
    p_fig = os.path.join(out_dir, n_fig + '.png')
    logging.debug('.. export Visualization as "%s...%s"', p_fig[:19], p_fig[-19:])
    fig.savefig(p_fig, bbox_inches='tight', pad_inches=0.05)
    plt.close()


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
    if not os.path.exists(out_dir):
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


def apdl_initialisation(imgs, init_atlas, init_weights, out_dir, out_prefix):
    """ more complex initialisation depending on inputs

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> init_atlas:
    :param np.array<nb_imgs, nb_lbs> init_weights:
    :param str out_prefix:
    :param str out_dir: path to the results directory
    :return: np.array<height, width>, np.array<nb_imgs, nb_lbs>
    """
    if init_weights is not None and init_atlas is None:
        logging.debug('... initialise Atlas from w_bins')
        init_atlas = estimate_atlas_graphcut_general(imgs, init_weights, 0.)
        export_visual_atlas(0, out_dir, init_atlas, out_prefix)
    if init_atlas is None:
        max_nb_lbs = int(np.sqrt(len(imgs)))
        logging.debug('... initialise Atlas with ')
        # IDEA: find better way of initialisation
        init_atlas = ptn_dict.initialise_atlas_mosaic(imgs[0].shape, max_nb_lbs)
        export_visual_atlas(0, out_dir, init_atlas, out_prefix)

    atlas = init_atlas
    w_bins = init_weights
    if len(np.unique(atlas)) == 1:
        logging.error('the init. atlas does not contain any label... %s',
                      repr(np.unique(atlas)))
    export_visual_atlas(0, out_dir, atlas, out_prefix)
    return atlas, w_bins


def apdl_update_weights(imgs, atlas, overlap_major=False):
    """ single iteration of the block coordinate descent algo

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> atlas:
    :return: np.array<nb_imgs, nb_lbs>
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


def apdl_update_atlas(imgs, atlas, w_bins, label_max, gc_coef, gc_reinit, ptn_split):
    """ single iteration of the block coordinate descent algo

    :param [np.array<height, width>] imgs:
    :param np.array<height, width> atlas:
    :param np.array<nb_imgs, nb_lbs> w_bins:
    :param int label_max:
    :param float gc_coef: graph cut regularisation
    :param bool gc_reinit: weather use atlas from previous step as init for act.
    :param bool ptn_split:
    :return: np.array<height, width>
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
    return atlas_new


def apdl_pipe_atlas_learning_ptn_weights(imgs, init_atlas=None, init_weights=None,
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
    atlas, w_bins = apdl_initialisation(imgs, init_atlas, init_weights,
                                        out_dir, out_prefix)
    list_crit = []

    for iter in range(max_iter):
        if len(np.unique(atlas)) == 1:
            logging.warning('.. iter: %i, no labels in the atlas %s', iter,
                            repr(np.unique(atlas).tolist()))
        w_bins = apdl_update_weights(imgs, atlas, overlap_major)
        atlas_reinit, w_bins = ptn_dict.reinit_atlas_likely_patterns(
                                    imgs, w_bins, atlas, label_max, ptn_compact)
        atlas_new = apdl_update_atlas(imgs, atlas_reinit, w_bins, label_max,
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
