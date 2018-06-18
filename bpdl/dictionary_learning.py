"""
The main module for Atomic pattern dictionary, jjoiningthe atlas estimation
and computing the encoding / weights

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

# from __future__ import absolute_import
import os
import time
import logging
import traceback

# to suppress all visual, has to be on the beginning
import matplotlib
if os.environ.get('DISPLAY', '') == '' \
        and matplotlib.rcParams['backend'] != 'agg':
    # logging.warning('No display found. Using non-interactive Agg backend.')
    # https://matplotlib.org/faq/usage_faq.html
    matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.segmentation as sk_image
# using https://github.com/Borda/pyGCO
from gco import cut_general_graph, cut_grid_graph_simple

import bpdl.pattern_atlas as ptn_atlas
import bpdl.pattern_weights as ptn_weight
import bpdl.metric_similarity as sim_metric
import bpdl.data_utils as tl_data
import bpdl.registration as regist

NB_GRAPH_CUT_ITER = 5
TEMPLATE_NAME_ATLAS = 'BPDL_{}_{}_iter_{:04d}'
LIST_BPDL_STEPS = [
    'weights update',
    'reinit. atlas',
    'atlas update',
    'deform images'
]

# TRY: init: spatial clustering
# TRY: init: use ICA
# TRY: init: greedy


def estimate_atlas_graphcut_simple(imgs, ptn_weights, coef=1.):
    """ run the graphcut to estimate atlas from computed unary terms
    source: https://github.com/yujiali/pyGCO

    :param [np.array<height, width>] imgs: list of input binary images
    :param np.array<nb_imgs, nb_lbs> ptn_weights: binary ptn selection
    :param float coef: coefficient for graphcut
    :return [int]:

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
    >>> np.sum(abs(estimate_atlas_graphcut_simple(imgs, luts[:, :1]).astype(int)))
    0
    """
    logging.debug('estimate atlas via GraphCut from Potts model')
    if ptn_weights.shape[1] <= 1:
        logging.warning('nothing to do for single label')
        labels = np.zeros(imgs[0].shape)
        return labels

    labeling_sum = ptn_atlas.compute_positive_cost_images_weights(imgs, ptn_weights)
    unary_cost = np.array(-1 * labeling_sum, dtype=np.int32)
    logging.debug('graph unaries potentials %s: \n %s', repr(unary_cost.shape),
                  repr(zip(np.histogram(unary_cost, bins=10))))
    # original and the right way..
    pairwise = (1 - np.eye(labeling_sum.shape[-1])) * coef
    pairwise_cost = np.array(pairwise, dtype=np.int32)
    logging.debug('graph pairwise coefs %s', repr(pairwise_cost.shape))
    # run GraphCut
    try:
        labels = cut_grid_graph_simple(unary_cost, pairwise_cost,
                                       algorithm='expansion')
    except Exception:
        logging.warning(traceback.format_exc())
        labels = np.argmin(unary_cost, axis=1)
    # reshape labels
    labels = labels.reshape(labeling_sum.shape[:2])
    logging.debug('resulting labelling %s: \n %s', repr(labels.shape), repr(labels))
    return labels


def estimate_atlas_graphcut_general(imgs, ptn_weights, coef=0., init_atlas=None,
                                    connect_diag=False):
    """ run the graphcut on the unary costs with specific pairwise cost
    source: https://github.com/yujiali/pyGCO

    :param [ndarray] imgs: list of np.array<height, width> input binary images
    :param ndarray ptn_weights: np.array<nb_imgs, nb_lbs> binary ptn selection
    :param float coef: coefficient for graphcut
    :param ndarray init_atlas: init labeling  np.array<nb_seg, 1>
        while None it take the arg ming of the unary costs
    :param bool connect_diag: used connecting diagonals, like use 8- instead 4-neighbour
    :return ndarray: np.array<nb_seg, 1>

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
    >>> np.sum(abs(estimate_atlas_graphcut_general(imgs, luts[:, :1]).astype(int)))
    0
    """
    logging.debug('estimate atlas via GraphCut from Potts model')
    if ptn_weights.shape[1] <= 1:
        logging.warning('nothing to do for single label')
        labels = np.zeros(imgs[0].shape)
        return labels

    u_cost = ptn_atlas.compute_relative_penalty_images_weights(imgs, ptn_weights)
    # u_cost = 1. / (labelingSum +1)
    unary_cost = np.array(u_cost, dtype=np.float64)
    unary_cost = unary_cost.reshape(-1, u_cost.shape[-1])
    logging.debug('graph unaries potentials %s: \n %s', repr(unary_cost.shape),
                  repr(zip(np.histogram(unary_cost, bins=10))))

    edges, edge_weights = ptn_atlas.edges_in_image2d_plane(u_cost.shape[:-1], connect_diag)

    # original and the right way...
    pairwise = (1 - np.eye(u_cost.shape[-1])) * coef
    pairwise_cost = np.array(pairwise, dtype=np.float64)
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
    except Exception:
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

    CRASH: TclError: no display name and no $DISPLAY environment variable
    >>> img = np.random.random((50, 50))
    >>> path_fig = export_visualization_image(img, 0, '.')
    >>> os.path.exists(path_fig)
    True
    >>> os.remove(path_fig)
    """
    # plt.ioff()
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
    :param ndarray atlas: np.array<height, width>
    :param str prefix:

    >>> import shutil
    >>> dir_name = 'sample_dir'
    >>> os.mkdir(dir_name)
    >>> export_visual_atlas(0, dir_name, np.random.randint(0, 5, (10, 5)))
    >>> shutil.rmtree(dir_name, ignore_errors=True)
    """
    if logging.getLogger().getEffectiveLevel() < logging.DEBUG:
        return
    if out_dir is None or not os.path.exists(out_dir):
        logging.debug('results path "%s" does not exist', out_dir)
        return None
    if atlas is not None:
        # export_visualization_image(atlas, i, out_dir, prefix, 'atlas',
        #                            labels=['X', 'Y'])
        n_img = TEMPLATE_NAME_ATLAS.format(prefix, 'atlas', i)
        tl_data.export_image(out_dir, atlas, n_img)
    # if weights is not None:
    #     export_visualization_image(weights, i, out_dir, prefix, 'weights',
    #                                'auto', ['patterns', 'images'])


def bpdl_initialisation(imgs, init_atlas, init_weights, out_dir, out_prefix,
                        rand_seed=None):
    """ more complex initialisation depending on inputs

    :param [ndarray] imgs: list of np.array<height, width>
    :param ndarray init_atlas: np.array<height, width>
    :param ndarray init_weights: np.array<nb_imgs, nb_lbs>
    :param str out_prefix:
    :param str out_dir: path to the results directory
    :param rand_seed: random initialization
    :return (ndarray, ndarray): np.array<height, width>, np.array<nb_imgs, nb_lbs>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> imgs = [lut[atlas] for lut in luts]
    >>> w_bins = luts[:, 1:]
    >>> init_atlas, init_w_bins = bpdl_initialisation(imgs, init_atlas=None,
    ...        init_weights=w_bins, out_dir=None, out_prefix='', rand_seed=0)
    >>> init_atlas.astype(int)
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> init_w_bins
    array([[1, 0],
           [1, 0],
           [1, 0],
           [0, 1],
           [0, 1],
           [0, 1],
           [1, 1],
           [1, 1],
           [1, 1]])
    >>> init_atlas, init_w_bins = bpdl_initialisation(imgs, init_atlas=None,
    ...        init_weights=None, out_dir=None, out_prefix='', rand_seed=0)
    >>> init_atlas
    array([[3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
           [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
           [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
    >>> init_w_bins
    """
    if init_weights is not None and init_atlas is None:
        logging.debug('... initialise Atlas from w_bins')
        init_atlas = estimate_atlas_graphcut_general(imgs, init_weights, 0.)
        # export_visual_atlas(0, out_dir, init_atlas, out_prefix)
    if init_atlas is None:
        nb_patterns = int(np.sqrt(len(imgs)))
        logging.debug('... initialise Atlas with ')
        # IDEA: find better way of initialisation
        init_atlas = ptn_atlas.init_atlas_mosaic(
            imgs[0].shape, nb_patterns, rand_seed=rand_seed)
        # export_visual_atlas(0, out_dir, init_atlas, out_prefix)

    atlas = init_atlas
    w_bins = init_weights
    if len(np.unique(atlas)) == 1:
        logging.error('the init. atlas does not contain any label... %s',
                      repr(np.unique(atlas)))
    export_visual_atlas(0, out_dir, atlas, out_prefix)
    return atlas, w_bins


def bpdl_update_weights(imgs, atlas, overlap_major=False):
    """ single iteration of the block coordinate descent algo

    :param [ndarray] imgs: list of images np.array<height, width>
    :param ndarray atlas: used atlas of np.array<height, width>
    :param bool overlap_major: whether it has majority overlap the pattern
    :return ndarray: np.array<nb_imgs, nb_lbs>

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
    >>> bpdl_update_weights(imgs, atlas, overlap_major=True)
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


def bpdl_update_atlas(imgs, atlas, w_bins, label_max, gc_coef, gc_reinit,
                      ptn_split, connect_diag=False):
    """ single iteration of the block coordinate descent algo

    :param [ndarray] imgs: list of images np.array<height, width>
    :param ndarray atlas: used atlas of np.array<height, width>
    :param ndarray w_bins: weights np.array<nb_imgs, nb_lbs>
    :param int label_max: max number of used labels
    :param float gc_coef: graph cut regularisation
    :param bool gc_reinit: weather use atlas from previous step as init for act.
    :param bool ptn_split: split individial patterns
    :param bool connect_diag: used connecting diagonals, like use 8- instead 4-neighbour
    :return ndarray: np.array<height, width>

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
        atlas_new = estimate_atlas_graphcut_general(imgs, w_bins, gc_coef, atlas,
                                                    connect_diag=connect_diag)
    else:
        atlas_new = estimate_atlas_graphcut_general(imgs, w_bins, gc_coef,
                                                    connect_diag=connect_diag)

    if ptn_split:
        atlas_new = ptn_atlas.atlas_split_indep_ptn(atlas_new, label_max)

    atlas_new = np.remainder(atlas_new, label_max + 1)
    return atlas_new


def bpdl_deform_images(images, atlas, weights, deform_coef, inverse=False):
    if deform_coef is None or deform_coef < 0:
        return images, None
    # coef = deform_coef * np.sqrt(np.product(images.shape))
    smooth_coef = deform_coef * min(images[0].shape)
    logging.debug('... perform register images onto atlas with smooth_coef: %f',
                  smooth_coef)
    images_warped, deforms = regist.register_images_to_atlas_demons(
                            images, atlas, weights, smooth_coef, inverse=inverse)
    return images_warped, deforms


def bpdl_pipeline(images, init_atlas=None, init_weights=None,
                  gc_regul=0.0, tol=1e-3, max_iter=25,
                  gc_reinit=True, ptn_split=True,
                  overlap_major=False, ptn_compact=True,
                  connect_diag=False, deform_coef=None,
                  out_prefix='debug', out_dir=''):
    """ the experiments_synthetic pipeline for block coordinate descent
    algo with graphcut...

    :param float deform_coef: regularise the deformation
    :param [ndarray] images: list of images np.array<height, width>
    :param ndarray init_atlas: used atlas of np.array<height, width>
    :param ndarray init_weights: weights np.array<nb_imgs, nb_lbs>
    :param float gc_regul: graph cut regularisation
    :param float tol: stop if the diff between two conseq steps
        is less then this given threshold. eg for -1 never until max nb iters
    :param int max_iter: max namber of iteration
    :param bool gc_reinit: wether use atlas from previous step as init for act.
    :param bool ptn_compact: enforce compactness of patterns
    :param bool ptn_split: split the connected components
    :param bool overlap_major: whether it has majority overlap the pattern
    :param bool connect_diag: used connecting diagonals, like use 8- instead 4-neighbour
    :param str out_prefix:
    :param str out_dir: path to the results directory
    :return (ndarray, ndarray): np.array<height, width>, np.array<nb_imgs, nb_lbs>

    >>> atlas = np.zeros((8, 12), dtype=int)
    >>> atlas[:3, 1:5] = 1
    >>> atlas[3:7, 6:12] = 2
    >>> luts = np.array([[0, 1, 0]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 3)
    >>> images = [lut[atlas] for lut in luts]
    >>> w_bins = luts[:, 1:]
    >>> init_atlas = ptn_atlas.init_atlas_mosaic(atlas.shape, nb_patterns=2,
    ...                                         coef=1.5, rand_seed=0)
    >>> init_atlas
    array([[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
           [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
           [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
           [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
           [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1]])
    >>> bpdl_atlas, bpdl_w_bins, deforms = bpdl_pipeline(images, init_atlas)
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
    >>> bpdl_atlas, bpdl_w_bins, deforms = bpdl_pipeline(images, init_atlas,
    ...                                                  deform_coef=1)
    >>> bpdl_atlas
    array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('compute an Atlas and weights for %i images...', len(images))
    assert len(images) >= 0, 'missing input images'
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    # initialise
    label_max = np.max(init_atlas)
    assert label_max > 0, 'at least some patterns should be searched'
    logging.debug('max nb labels set: %i', label_max)
    atlas, w_bins = bpdl_initialisation(images, init_atlas, init_weights,
                                        out_dir, out_prefix)
    list_diff = []
    list_times = []
    imgs_warped = images
    deforms = None
    max_iter = max(1, max_iter)  # set at least single iteration

    for it in range(max_iter):
        if len(np.unique(atlas)) == 1:
            logging.warning('.. iter: %i, no labels in the atlas %s', it,
                            repr(np.unique(atlas).tolist()))
        times = [time.time()]
        # 1: update WEIGHTS
        w_bins = bpdl_update_weights(imgs_warped, atlas, overlap_major)
        times.append(time.time())
        # 2: reinitialise empty patterns
        atlas_reinit, w_bins = ptn_atlas.reinit_atlas_likely_patterns(
            imgs_warped, w_bins, atlas, label_max, ptn_compact)
        times.append(time.time())
        # 3: update the ATLAS
        atlas_new = bpdl_update_atlas(imgs_warped, atlas_reinit, w_bins,
                                      label_max, gc_regul, gc_reinit, ptn_split,
                                      connect_diag)
        times.append(time.time())
        # 4: optional deformations
        if it > 0:
            imgs_warped, deforms = bpdl_deform_images(images, atlas_new,
                                                      w_bins, deform_coef)
            times.append(time.time())

        times = [times[i] - times[i-1] for i in range(1, len(times))]
        d_times = dict(zip(LIST_BPDL_STEPS[:len(times)], times))
        step_diff = sim_metric.compare_atlas_adjusted_rand(atlas, atlas_new)
        # step_diff = np.sum(abs(atlas - atlas_new)) / float(np.product(atlas.shape))
        list_diff.append(step_diff)
        list_times.append(d_times)
        atlas = sk_image.relabel_sequential(atlas_new)[0]

        logging.debug('-> iter. #%i with Atlas diff %f', (it + 1), step_diff)
        export_visual_atlas(it + 1, out_dir, atlas, out_prefix)

        # STOPPING criterion
        if step_diff <= tol and len(np.unique(atlas)) > 1:
            logging.debug('>> exit while the atlas diff %f is smaller then %f',
                          step_diff, tol)
            break

    imgs_warped, deforms = bpdl_deform_images(images, atlas, w_bins, deform_coef)
    logging.info('BPDL: terminated with iter %i / %i and step diff %f <? %f',
                 len(list_diff), max_iter, list_diff[-1], tol)
    logging.debug('criterion evolved:\n %s', repr(list_diff))
    df_time = pd.DataFrame(list_times)
    logging.debug('measured time: \n%s', repr(df_time))
    logging.info('times: \n%s', df_time.describe())
    # atlas = sk_image.relabel_sequential(atlas)[0]
    w_bins = [ptn_weight.weights_image_atlas_overlap_major(img, atlas)
              for img in imgs_warped]
    return atlas, np.array(w_bins), deforms
