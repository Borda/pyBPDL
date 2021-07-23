"""
The basic module for generating synthetic images and also loading / exporting

Copyright (C) 2015-2020 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import logging
# from __future__ import absolute_import
import os
import re
import shutil
import types

import numpy as np
from scipy import stats
from scipy.spatial import distance

# import multiprocessing.pool
# import multiprocessing as mproc
# from functools import wraps

# def update_path(path_file, lim_depth=5, absolute=True):
#     """ bubble in the folder tree up intil it found desired file
#     otherwise return original one
#
#     :param str path_file: path
#     :param int lim_depth: length of bubble attempted
#     :param bool absolute: absolute path
#     :return str:
#
#     >>> path = 'sample_file.test'
#     >>> f = open(path, 'w')
#     >>> update_path(path, absolute=False)
#     'sample_file.test'
#     >>> os.remove(path)
#     """
#     if path_file.startswith('/'):
#         return path_file
#     elif path_file.startswith('~'):
#         path_file = os.path.expanduser(path_file)
#     else:
#         for _ in range(lim_depth):
#             if os.path.exists(path_file):
#                 break
#             path_file = os.path.join('..', path_file)
#     if absolute:
#         path_file = os.path.abspath(path_file)
#     return path_file


def convert_numerical(s):
    """ try to convert a string tu numerical

    :param str s: input string
    :return:

    >>> convert_numerical('-1')
    -1
    >>> convert_numerical('-2.0')
    -2.0
    >>> convert_numerical('.1')
    0.1
    >>> convert_numerical('-0.')
    -0.0
    >>> convert_numerical('abc58')
    'abc58'
    """
    re_int = re.compile(r"^[-]?\d+$")
    re_float1 = re.compile(r"^[-]?\d+.\d*$")
    re_float2 = re.compile(r"^[-]?\d*.\d+$")

    if re_int.match(str(s)) is not None:
        return int(s)
    elif re_float1.match(str(s)) is not None:
        return float(s)
    elif re_float2.match(str(s)) is not None:
        return float(s)
    else:
        return s


def create_clean_folder(path_dir):
    """ create empty folder and while the folder exist clean all files

    :param str path_dir: path
    :return str:

    >>> path_dir = os.path.abspath('sample_dir')
    >>> path_dir = create_clean_folder(path_dir)
    >>> os.path.exists(path_dir)
    True
    >>> shutil.rmtree(path_dir, ignore_errors=True)
    """
    if os.path.isdir(os.path.dirname(path_dir)):
        logging.warning('existing folder will be cleaned: %s', path_dir)
    logging.info('create clean folder "%s"', path_dir)
    if os.path.exists(path_dir):
        shutil.rmtree(path_dir, ignore_errors=True)
    os.mkdir(path_dir)
    return path_dir


def generate_gauss_2d(mean, std, im_size=None, norm=None):
    """ Generating a Gaussian distribution in 2D image

    :param float norm: normalise the maximal value
    :param list(int) mean: mean position
    :param list(list(int)) std: STD
    :param tuple(int,int) im_size: optional image size
    :return ndarray:

    >>> im = generate_gauss_2d([4, 5], [[1, 0], [0, 2]], (8, 10), norm=1.)
    >>> np.round(im, 1)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0.1,  0.1,  0.1,  0.1,  0.1,  0. ,  0. ],
           [ 0. ,  0.1,  0.2,  0.4,  0.5,  0.6,  0.5,  0.4,  0.2,  0.1],
           [ 0. ,  0.1,  0.3,  0.6,  0.9,  1. ,  0.9,  0.6,  0.3,  0.1],
           [ 0. ,  0.1,  0.2,  0.4,  0.5,  0.6,  0.5,  0.4,  0.2,  0.1],
           [ 0. ,  0. ,  0. ,  0.1,  0.1,  0.1,  0.1,  0.1,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])
    >>> im = generate_gauss_2d([2, 3], [[1., 0], [0, 1.2]])
    >>> np.round(im, 2)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.  ,  0.  ,  0.01,  0.02,  0.01,  0.  ,  0.  ,  0.  ],
           [ 0.  ,  0.02,  0.06,  0.08,  0.06,  0.02,  0.  ,  0.  ],
           [ 0.01,  0.03,  0.09,  0.13,  0.09,  0.03,  0.01,  0.  ],
           [ 0.  ,  0.02,  0.06,  0.08,  0.06,  0.02,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  0.01,  0.02,  0.01,  0.  ,  0.  ,  0.  ]])
    """
    covar = np.array(std)**2
    if im_size is None:
        im_size = np.array(mean) + covar.diagonal() * 3

    x, y = np.mgrid[0:im_size[0], 0:im_size[1]]
    pos = np.rollaxis(np.array([x, y]), 0, 3)
    gauss = stats.multivariate_normal(mean, covar)
    pdf = gauss.pdf(pos)

    if norm is not None:
        pdf *= norm / np.max(pdf)
    return pdf


# class NonDaemonPool(multiprocessing.pool.Pool):
#     """ We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
#     because the latter is only a wrapper function, not a proper class.
#
#     See: https://github.com/nipy/nipype/pull/2754
#
#     >>> pool = NonDaemonPool(1)
#     """
#     def Process(self, *args, **kwds):
#         proc = super(NonDaemonPool, self).Process(*args, **kwds)
#
#         class NonDaemonProcess(proc.__class__):
#             """Monkey-patch process to ensure it is never daemonized"""
#             @property
#             def daemon(self):
#                 return False
#
#             @daemon.setter
#             def daemon(self, val):
#                 pass
#
#         proc.__class__ = NonDaemonProcess
#         return proc

# def wrap_execute_sequence(wrap_func, iterate_vals, nb_workers=NB_WORKERS,
#                           desc='', ordered=False):
#     """ wrapper for execution parallel of single thread as for...
#
#     :param wrap_func: function which will be excited in the iterations
#     :param list iterate_vals: list or iterator which will ide in iterations
#     :param int nb_workers: number og jobs running in parallel
#     :param str desc: description for the bar,
#         if it is set None, bar is suppressed
#     :param bool ordered: whether enforce ordering in the parallelism
#
#     >>> [o for o in wrap_execute_sequence(lambda x: x ** 2, range(5),
#     ...                                   nb_workers=1, ordered=True)]
#     [0, 1, 4, 9, 16]
#     >>> [o for o in wrap_execute_sequence(sum, [[0, 1]] * 5,
#     ...                                   nb_workers=2, desc=None)]
#     [1, 1, 1, 1, 1]
#     """
#     iterate_vals = list(iterate_vals)
#
#     tqdm_bar = None
#     if desc is not None:
#         desc = '%r @%i-threads' % (desc, nb_workers)
#         desc = str(desc.encode('utf-8').decode())
#         tqdm_bar = tqdm.tqdm(total=len(iterate_vals), desc=desc)
#
#     if nb_workers > 1:
#         logging.debug('perform sequential in %i threads', nb_workers)
#         # Standard mproc.Pool created a demon processes which can be called
#         # inside its children, cascade or multiprocessing
#         # https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
#         pool = NonDaemonPool(nb_workers)
#         pooling = pool.imap if ordered else pool.imap_unordered
#
#         for out in pooling(wrap_func, iterate_vals):
#             tqdm_bar.update() if tqdm_bar is not None else None
#             yield out
#         pool.close()
#         pool.join()
#     else:
#         for out in map(wrap_func, iterate_vals):
#             tqdm_bar.update() if tqdm_bar is not None else None
#             yield out
#
#     tqdm_bar.close() if tqdm_bar is not None else None


def estimate_rolling_ball(points, tangent_smooth=1, max_diam=1e6, step_tol=1e-3):
    """ roll a ball over curve and get for each particular position a maximal
    ball which does not intersect the rest of curve

    :param points:
    :param tangent_smooth:
    :param max_diam:
    :param step_tol:
    :return:

    >>> y = [1] * 6 + [2] * 4
    >>> pts = np.array(list(zip(range(len(y)), y)))
    >>> diams = estimate_rolling_ball(pts)
    >>> list(map(int, diams[0]))
    [24, 18, 12, 8, 4, 1, 9, 999999, 999999, 999999]
    >>> list(map(int, diams[1]))
    [999999, 999999, 999999, 999999, 999999, 10, 1, 4, 8, 12]
    """
    # points = np.array(sorted(points, key=lambda p: p[0]))
    dir_diams = []
    for d in [1., -1]:
        diams = [
            estimate_point_max_circle(i, points, tangent_smooth, d, max_diam, step_tol) for i in range(len(points))
        ]
        dir_diams.append(diams)
    return dir_diams


def estimate_point_max_circle(idx, points, tangent_smooth=1, orient=1., max_diam=1e6, step_tol=1e-3):
    """ estimate maximal circle from a particular point on curve

    :param int idx: index or point on curve
    :param [[float, float]] points: list of point on curve
    :param int tangent_smooth: distance for tangent
    :param float direct: positive or negative ortogonal
    :param float max_diam: maximal diameter
    :param float step_tol: tolerance step in dividing diameter interval
    :return:

    >>> y = [1] * 25 + list(range(1, 50)) + [50] * 25
    >>> pts = np.array(list(zip(range(len(y)), y)))
    >>> estimate_point_max_circle(0, pts)   # doctest: +ELLIPSIS
    60.38...
    >>> estimate_point_max_circle(30, pts)   # doctest: +ELLIPSIS
    17.14...
    >>> estimate_point_max_circle(90, pts)   # doctest: +ELLIPSIS
    999999.99...
    """
    # norm around selected point
    idx_left = idx - tangent_smooth
    idx_left = 0 if idx_left < 0 else idx_left
    idx_right = idx + tangent_smooth
    idx_right = len(points) - 1 if idx_right >= len(points) else idx_right

    # compute the tanget from neighboring points
    tangent = points[idx_right] - points[idx_left]

    # rotate by 90 degree
    direction = np.array([[0, -1], [1, 0]]).dot(tangent)
    # set positive or negative direction
    direction = direction * orient
    # normalisation
    direction = direction / np.sqrt(np.sum(direction**2))

    diam = estimate_max_circle(points[idx], direction, points, max_diam, step_tol)
    return diam


def estimate_max_circle(point, direction, points, max_diam=1000, step_tol=1e-3):
    """ find maximal circe from a given pont in orthogonal direction
    which just touch the curve with points

    :param tuple(float,float) point: particular point on curve
    :param tuple(float,float) direction: orthogonal direction
    :param [[float, float]] points: list of point on curve
    :param float max_diam: maximal diameter
    :param float step_tol: tolerance step in dividing diameter interval
    :return:

    >>> y = [1] * 10
    >>> pts = np.array(list(zip(range(len(y)), y)))
    >>> estimate_max_circle([5, 1], [0, 1], pts)   # doctest: +ELLIPSIS
    999.99...
    >>> y = [1] * 6 + [2] * 4
    >>> pts = np.array(list(zip(range(len(y)), y)))
    >>> estimate_max_circle([4, 1], [0, 1], pts)   # doctest: +ELLIPSIS
    4.99...
    """
    # set initial interval bounds
    diam_min, diam_max = 0, max_diam
    # iterate until the step diff is minimal
    while (diam_max - diam_min) >= step_tol:
        diam = np.mean([diam_min, diam_max])
        # set circle center from particula point in given direction
        center = np.asarray(point) + (np.asarray(direction) * diam)

        # count number of inliers in the circle
        dists = distance.cdist(np.asarray([center]), points)[0]
        count = np.sum(dists < diam)
        if count > 1:
            diam_max = diam
        else:
            diam_min = diam

    return np.mean([diam_min, diam_max])


# def try_decorator(func):
#     """ costume decorator to wrap function in try/except
#
#     :param func:
#     :return:
#     """
#     @wraps(func)
#     def wrap(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception:
#             logging.exception('%r with %r and %r', func.__name__, args, kwargs)
#     return wrap


def is_list_like(var):
    """ check if the variable is iterable

    :param var:
    :return bool:

    >>> is_list_like('abc')
    False
    >>> is_list_like(123.)
    False
    >>> is_list_like([0])
    True
    >>> is_list_like((1, ))
    True
    >>> is_list_like(range(2))
    True
    """
    try:  # for python 3
        is_iter = [isinstance(var, tp) for tp in (list, tuple, range, np.ndarray, types.GeneratorType)]
    except Exception:  # for python 2
        is_iter = [isinstance(var, tp) for tp in (list, tuple, np.ndarray, types.GeneratorType)]
    return any(is_iter)


def is_iterable(var):
    """ check if the variable is iterable

    :param var:
    :return bool:

    >>> is_iterable('abc')
    False
    >>> is_iterable(123.)
    False
    >>> is_iterable((1, ))
    True
    >>> is_iterable(range(2))
    True
    """
    res = (hasattr(var, '__iter__') and not isinstance(var, str))
    return res


# def string_dict(d, desc='DICTIONARY:', offset=30):
#     """ transform dictionary to a formatted string
#
#     :param dict d: dictionary with parameters
#     :param int offset: length between name and value
#     :param str desc: dictionary title
#     :return str:
#
#     >>> string_dict({'abc': 123})  #doctest: +NORMALIZE_WHITESPACE
#     \'DICTIONARY:\\n"abc": 123\'
#     """
#     s = desc + '\n'
#     tmp_name = '{:' + str(offset) + 's} {}'
#     rows = [tmp_name.format('"{}":'.format(n), repr(d[n])) for n in sorted(d)]
#     s += '\n'.join(rows)
#     return str(s)

# def load_config_yaml(path_config):
#     """ loading the
#
#     :param str path_config:
#     :return dict:
#
#     >>> p_conf = './testing-congif.yaml'
#     >>> save_config_yaml(p_conf, {'a': 2})
#     >>> load_config_yaml(p_conf)
#     {'a': 2}
#     >>> os.remove(p_conf)
#     """
#     with open(path_config, 'r') as fp:
#         config = yaml.load(fp)
#     return config
#
#
# def save_config_yaml(path_config, config):
#     """ exporting configuration as YAML file
#
#     :param str path_config:
#     :param dict config:
#     """
#     with open(path_config, 'w') as fp:
#         yaml.dump(config, fp, default_flow_style=False)
