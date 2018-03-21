"""
Introducing some used similarity measures fro atlases and etc.

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

# from __future__ import absolute_import
import logging
import traceback

import numpy as np
from sklearn import metrics

METRIC_AVERAGES = ['macro', 'weighted']


def compare_atlas_rnd_pairs(a1, a2, rand_seed=None):
    """ compare two atlases as taking random pixels pairs from both
    and evaluate that the are labeled equally of differently

    :param a1: np.array<height, width>
    :param a2: np.array<height, width>
    :param rand_seed: random initialization
    :return float: with 0 means no difference

    >>> atlas1 = np.zeros((7, 15), dtype=int)
    >>> atlas1[1:4, 5:10] = 1
    >>> atlas1[5:7, 6:13] = 2
    >>> atlas2 = np.zeros((7, 15), dtype=int)
    >>> atlas2[2:5, 7:12] = 1
    >>> atlas2[4:7, 7:14] = 2
    >>> compare_atlas_rnd_pairs(atlas1, atlas1)
    0.0
    >>> round(compare_atlas_rnd_pairs(atlas1, atlas2, rand_seed=0), 5)
    0.37143
    """
    logging.debug('comparing two atlases of shapes %s <-> %s',
                  repr(a1.shape), repr(a2.shape))
    assert a1.shape == a2.shape, \
        'shapes: %s and %s' % (repr(a1.shape), repr(a2.shape))
    # assert A1.shape[0]==A2.shape[0] and A1.shape[1]==A2.shape[1]
    np.random.seed(rand_seed)
    logging.debug('unique labels are %s and %s', repr(np.unique(a1).tolist()),
                  repr(np.unique(a2).tolist()))
    matrix_x, matrix_y = np.meshgrid(range(a1.shape[0]), range(a1.shape[1]))
    vec_x, vec_y = matrix_x.flatten(), matrix_y.flatten()
    vec_x_perm = np.random.permutation(vec_x)
    vec_y_perm = np.random.permutation(vec_y)
    diffs = 0
    for x1, y1, x2, y2 in zip(vec_x, vec_y, vec_x_perm, vec_y_perm):
        b1 = a1[x1, y1] == a1[x2, y2]
        b2 = a2[x1, y1] == a2[x2, y2]
        if not b1 == b2:  # T&F or F&T
            diffs += 1
    res = diffs / float(len(vec_x))
    return res


def compare_atlas_adjusted_rand(a1, a2):
    """ using adjusted rand and transform original values from (-1, 1) to (0, 1)
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    :param a1: np.array<height, width>
    :param a2: np.array<height, width>
    :return float: with 0 means no difference

    >>> atlas1 = np.zeros((7, 15), dtype=int)
    >>> atlas1[1:4, 5:10] = 1
    >>> atlas1[5:7, 6:13] = 2
    >>> atlas2 = np.zeros((7, 15), dtype=int)
    >>> atlas2[2:5, 7:12] = 1
    >>> atlas2[4:7, 7:14] = 2
    >>> compare_atlas_adjusted_rand(atlas1, atlas1)
    0.0
    >>> compare_atlas_adjusted_rand(atlas1, atlas2) #doctest: +ELLIPSIS
    0.656...
    """
    assert a1.shape == a2.shape, \
        'shapes: %s and %s' % (repr(a1.shape), repr(a2.shape))
    ars = metrics.adjusted_rand_score(a1.ravel(), a2.ravel())
    res = 1. - abs(ars)
    return res


def compute_labels_overlap_matrix(seg1, seg2):
    """ compute overlap between tho segmentation atlasess) with same sizes

    :param seg1: np.array<height, width>
    :param seg2: np.array<height, width>
    :return: np.array<height, width>

    >>> seg1 = np.zeros((7, 15), dtype=int)
    >>> seg1[1:4, 5:10] = 3
    >>> seg1[5:7, 6:13] = 2
    >>> seg2 = np.zeros((7, 15), dtype=int)
    >>> seg2[2:5, 7:12] = 1
    >>> seg2[4:7, 7:14] = 3
    >>> compute_labels_overlap_matrix(seg1, seg1)
    array([[76,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, 14,  0],
           [ 0,  0,  0, 15]])
    >>> compute_labels_overlap_matrix(seg1, seg2)
    array([[63,  4,  0,  9],
           [ 0,  0,  0,  0],
           [ 2,  0,  0, 12],
           [ 9,  6,  0,  0]])
    """
    logging.debug('computing overlap of two seg_pipe of shapes %s <-> %s',
                  repr(seg1.shape), repr(seg2.shape))
    assert seg1.shape == seg2.shape, \
        'shapes: %s and %s' % (repr(seg1.shape), repr(seg2.shape))
    maxims = [np.max(seg1) + 1, np.max(seg2) + 1]
    overlap = np.zeros(maxims, dtype=int)
    for i in range(seg1.shape[0]):
        for j in range(seg1.shape[1]):
            lb1, lb2 = seg1[i, j], seg2[i, j]
            overlap[lb1, lb2] += 1
    # logging.debug(res)
    return overlap


def compare_matrices(m1, m2):
    """ sum all element differences and divide it by number of elements

    :param m1: np.array<height, width>
    :param m2: np.array<height, width>
    :return float:

    >>> seg1 = np.zeros((7, 15), dtype=int)
    >>> seg1[1:4, 5:10] = 3
    >>> seg1[5:7, 6:13] = 2
    >>> seg2 = np.zeros((7, 15), dtype=int)
    >>> seg2[2:5, 7:12] = 1
    >>> seg2[4:7, 7:14] = 3
    >>> compare_matrices(seg1, seg1)
    0.0
    >>> compare_matrices(seg1, seg2) # doctest: +ELLIPSIS
    0.819...
    """
    assert m1.shape == m2.shape, \
        'shapes: %s and %s' % (repr(m1.shape), repr(m2.shape))
    diff = np.sum(abs(m1 - m2))
    return diff / float(np.product(m1.shape))


def compare_weights(c1, c2):
    """
    :param c1: np.array<height, width>
    :param c2: np.array<height, width>
    :return float:

    >>> np.random.seed(0)
    >>> compare_weights(np.random.randint(0, 2, (10, 5)),
    ...                 np.random.randint(0, 2, (10, 5)))
    0.44
    """
    return compare_matrices(c1, c2)


def relabel_max_overlap_unique(seg_ref, seg_relabel, keep_bg=True):
    """ relabel the second segmentation cu that maximise relative overlap
    for each pattern (object), the relation among patterns is 1-1
    NOTE: it skips background class - 0

    :param ndarray seg_ref: segmentation
    :param ndarray seg_relabel: segmentation
    :param bool keep_bg:
    :return ndarray:

    >>> atlas1 = np.zeros((7, 15), dtype=int)
    >>> atlas1[1:4, 5:10] = 1
    >>> atlas1[5:7, 3:13] = 2
    >>> atlas2 = np.zeros((7, 15), dtype=int)
    >>> atlas2[0:3, 7:12] = 1
    >>> atlas2[3:7, 1:7] = 2
    >>> atlas2[4:7, 7:14] = 3
    >>> atlas2[:2, :3] = 5
    >>> relabel_max_overlap_unique(atlas1, atlas2, keep_bg=True)
    array([[5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0]])
    >>> relabel_max_overlap_unique(atlas2, atlas1, keep_bg=True)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0]])
    >>> relabel_max_overlap_unique(atlas1, atlas2, keep_bg=False)
    array([[5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0]])
    """
    assert seg_ref.shape == seg_relabel.shape, \
        'shapes: %s and %s' % (repr(seg_ref.shape), repr(seg_relabel.shape))
    overlap = compute_labels_overlap_matrix(seg_ref, seg_relabel)

    lut = [-1] * (np.max(seg_relabel) + 1)
    if keep_bg:  # keep the background label
        lut[0] = 0
        overlap[0, :] = 0
        overlap[:, 0] = 0
    for _ in range(max(overlap.shape) + 1):
        if np.sum(overlap) == 0:
            break
        lb_ref, lb_est = np.argwhere(overlap.max() == overlap)[0]
        lut[lb_est] = lb_ref
        overlap[lb_ref, :] = 0
        overlap[:, lb_est] = 0

    for i, lb in enumerate(lut):
        if lb == -1 and i not in lut:
            lut[i] = i
    for i, lb in enumerate(lut):
        if lb > -1:
            continue
        for j in range(len(lut)):
            if j not in lut:
                lut[i] = j

    seg_new = np.array(lut)[seg_relabel]
    return seg_new


def relabel_max_overlap_merge(seg_ref, seg_relabel, keep_bg=True):
    """ relabel the second segmentation cu that maximise relative overlap
    for each pattern (object), if one pattern in reference atlas is likely
    composed from multiple patterns in relabel atlas, it merge them
    NOTE: it skips background class - 0

    :param ndarray seg_ref: segmentation
    :param ndarray seg_relabel: segmentation
    :param bool keep_bg:
    :return ndarray:

    >>> atlas1 = np.zeros((7, 15), dtype=int)
    >>> atlas1[1:4, 5:10] = 1
    >>> atlas1[5:7, 3:13] = 2
    >>> atlas2 = np.zeros((7, 15), dtype=int)
    >>> atlas2[0:3, 7:12] = 1
    >>> atlas2[3:7, 1:7] = 2
    >>> atlas2[4:7, 7:14] = 3
    >>> atlas2[:2, :3] = 5
    >>> relabel_max_overlap_merge(atlas1, atlas2, keep_bg=True)
    array([[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]])
    >>> relabel_max_overlap_merge(atlas2, atlas1, keep_bg=True)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]])
    >>> relabel_max_overlap_merge(atlas1, atlas2, keep_bg=False)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0]])
    """
    assert seg_ref.shape == seg_relabel.shape, \
        'shapes: %s and %s' % (repr(seg_ref.shape), repr(seg_relabel.shape))
    overlap = compute_labels_overlap_matrix(seg_ref, seg_relabel)
    # ref_ptn_size = np.bincount(seg_ref.ravel())
    # overlap = overlap.astype(float) / np.tile(ref_ptn_size, (overlap.shape[1], 1)).T
    # overlap = np.nan_to_num(overlap)
    max_axis = 1 if overlap.shape[0] > overlap.shape[1] else 0
    if keep_bg:
        id_max = np.argmax(overlap[1:, 1:], axis=max_axis) + 1
        lut = np.array([0] + id_max.tolist())
    else:
        lut = np.argmax(overlap, axis=max_axis)
    # in case there is no overlap
    ptn_sum = np.sum(overlap, axis=0)
    if 0 in ptn_sum:
        lut[ptn_sum == 0] = np.arange(len(lut))[ptn_sum == 0]
    seg_new = lut[seg_relabel]
    return seg_new


def compute_classif_metrics(y_true, y_pred, metric_averages=METRIC_AVERAGES):
    """ compute standard metrics for multi-class classification

    :param [str] metric_averages:
    :param [int] y_true:
    :param [int] y_pred:
    :return: {str: float}

    >>> y_true = np.array([0] * 3 + [1] * 5)
    >>> y_pred = np.array([0] * 5 + [1] * 3)
    >>> dist_sm = compute_classif_metrics(y_true, y_pred)
    >>> pair_sm = [(n, dist_sm[n]) for n in sorted(dist_sm.keys())]
    >>> pair_sm #doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [('ARS', 0.138...),
     ('accuracy', 0.75),
     ('confusion', [[3, 0], [2, 3]]),
     ('f1_macro', 0.800...), ('f1_weighted', 0.849...),
     ('precision_macro', 0.800...), ('precision_weighted', 0.75),
     ('recall_macro', 0.749...), ('recall_weighted', 0.749...),
     ('support_macro', None), ('support_weighted', None)]
    >>> y_true = np.array([0] * 5 + [1] * 5 + [2] * 5)
    >>> y_pred = np.array([0] * 5 + [1] * 3 + [2] * 7)
    >>> dist_sm = compute_classif_metrics(y_true, y_pred)
    >>> pair_sm = [(n, dist_sm[n]) for n in sorted(dist_sm.keys())]
    >>> pair_sm #doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [('ARS', 0.641...),
     ('accuracy', 0.866...),
     ('confusion', [[5, 0, 0], [0, 3, 2], [0, 0, 5]]),
     ('f1_macro', 0.904...), ('f1_weighted', 0.904...),
     ('precision_macro', 0.866...), ('precision_weighted', 0.866...),
     ('recall_macro', 0.861...), ('recall_weighted', 0.861...),
     ('support_macro', None), ('support_weighted', None)]
    """
    y_pred = np.array(y_pred)
    assert y_true.shape == y_pred.shape, \
        'shapes: %s and %s' % (repr(y_true.shape), repr(y_pred.shape))
    uq_y_true = np.unique(y_true)
    logging.debug('unique lbs true: %s, predict %s',
                  repr(uq_y_true), repr(np.unique(y_pred)))

    # in case the are just two classes relabel them as [0, 1] only
    # solving sklearn error:
    #  "ValueError: pos_label=1 is not a valid label: array([  0, 255])"
    if np.array_equal(sorted(uq_y_true), sorted(np.unique(y_pred))) \
            and len(uq_y_true) <= 2:
        logging.debug('relabeling original %s to [0, 1]', repr(uq_y_true))
        lut = np.zeros(uq_y_true.max() + 1)
        if len(uq_y_true) == 2:
            lut[uq_y_true[1]] = 1
        y_true = lut[y_true]
        y_pred = lut[y_pred]

    dict_metrics = {
        'ARS': metrics.adjusted_rand_score(y_true, y_pred),
        # 'f1':  metrics.f1_score(y_true, y_pred),
        'accuracy':  metrics.accuracy_score(y_true, y_pred),
        # 'precision':  metrics.precision_score(y_true, y_pred),
        'confusion': metrics.confusion_matrix(y_true, y_pred).tolist(),
        # 'report':    metrics.classification_report(labels, predicted),
    }

    # compute aggregated precision, recall, f-score, support
    names = ['f1', 'precision', 'recall', 'support']
    for avg in metric_averages:
        try:
            mtr = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                          average=avg)
            res = dict(zip(['{}_{}'.format(n, avg) for n in names], mtr))
        except Exception:
            logging.error(traceback.format_exc())
            res = dict(zip(['{}_{}'.format(n, avg) for n in names], [0] * 4))
        dict_metrics.update(res)
    return dict_metrics
