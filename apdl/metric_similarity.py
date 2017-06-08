"""
Introducing some used similarity measures fro atlases and etc.

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging

import numpy as np
from sklearn import metrics


def compare_atlas_rnd_pairs(a1, a2):
    """ compare two atlases as taking random pixels pairs from both
    and evaluate that the are labeled equally of differently

    :param a1: np.array<height, width>
    :param a2: np.array<height, width>
    :return float: with 0 means no difference
    """
    logging.debug('comparing two atlases of shapes %s <-> %s',
                  repr(a1.shape), repr(a2.shape))
    assert np.array_equal(a1.shape, a2.shape)
    # assert A1.shape[0]==A2.shape[0] and A1.shape[1]==A2.shape[1]
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
        if not b1 == b2: # T&F or F&T
            diffs += 1
    res = diffs / float(len(vec_x))
    return res


def compare_atlas_adjusted_rand(a1, a2):
    """ using adjusted rand and transform original values from (-1, 1) to (0, 1)
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    :param a1: np.array<height, width>
    :param a2: np.array<height, width>
    :return float: with 0 means no difference
    """
    assert np.array_equal(a1.shape, a2.shape)
    ars = metrics.adjusted_rand_score(a1.ravel(), a2.ravel())
    res = 1 - abs(ars)
    return res


def overlap_matrix_mlabel_segm(seg1, seg2):
    """
    :param seg1: np.array<height, width>
    :param seg2: np.array<height, width>
    :return: np.array<height, width>
    """
    logging.debug('computing overlap of two seg_pipe of shapes %s <-> %s',
                 repr(seg1.shape), repr(seg2.shape))
    assert np.array_equal(seg1.shape, seg2.shape)
    u_lb1 = np.unique(seg1)
    u_lb2 = np.unique(seg2)
    u_lb1 = dict(zip(u_lb1, range(len(u_lb1))))
    u_lb2 = dict(zip(u_lb2, range(len(u_lb2))))
    logging.debug('unique labels:\n  %s\n  %s', repr(u_lb1), repr(u_lb2))
    res = np.zeros([len(u_lb1), len(u_lb2)])
    for i in range(seg1.shape[0]):
        for j in range(seg1.shape[1]):
            u1, u2 = u_lb1[seg1[i, j]], u_lb2[seg2[i, j]]
            res[u1, u2] += 1
            res[u2, u1] += 1
    # logging.debug(res)
    return res


def compare_matrices(m1, m2):
    """
    :param m1: np.array<height, width>
    :param m2: np.array<height, width>
    :return float:
    """
    assert np.array_equal(m1.shape, m2.shape)
    diff = np.sum(abs(m1 - m2))
    return diff / float(np.product(m1.shape))


def compare_weights(c1, c2):
    """
    :param c1: np.array<height, width>
    :param c2: np.array<height, width>
    :return float:
    """
    return compare_matrices(c1, c2)


def test_atlases():
    """    """
    logging.info('testing METRIC')
    a = np.random.randint(0, 4, (5, 5))
    a2 = a.copy()
    a2[a2==0] = -1
    b = np.random.randint(0, 4, (5, 5))

    logging.debug('compare_atlas_rnd_pairs, a <-> a: %d',
                  compare_atlas_rnd_pairs(a, a2))
    logging.debug('compare_atlas_rnd_pairs, a <-> b: %d',
                  compare_atlas_rnd_pairs(a, b))
    logging.debug('compare_atlas_adjusted_rand, a <-> a: %d',
                  compare_atlas_adjusted_rand(a, a2))
    logging.debug('compare_atlas_adjusted_rand, a <-> b: %d',
                  compare_atlas_adjusted_rand(a, b))
