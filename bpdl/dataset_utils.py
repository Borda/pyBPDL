"""
The basic module for generating synthetic images and also loading / exporting

Copyright (C) 2015-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""
from __future__ import absolute_import
import os
import glob
import logging
import itertools
import multiprocessing as mproc
from functools import partial
import shutil

# to suppress all visual, has to be on the beginning
import matplotlib
if os.environ.get('DISPLAY','') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, draw, transform, filters
from PIL import Image

NB_THREADS = mproc.cpu_count()
IMAGE_SIZE_2D = (128, 128)
IMAGE_SIZE_3D = (16, 128, 128)
NB_BIN_PATTERNS = 9
NB_SAMPLES = 50
RND_PATTERN_OCCLUSION = 0.25
IMAGE_POSIXS = ['.png', '.tif', '.tiff']
IMAGE_PATTERN = 'pattern_{:03d}'
SEGM_PATTERN = 'sample_{:05d}'
BLOCK_NB_LOAD_IMAGES = 50
DIR_MANE_SYNTH_DATASET = 'syntheticDataset_vX'
DIR_NAME_DICTIONARY = 'dictionary'
CSV_NAME_WEIGHTS = 'binary_weights.csv'
DEFAULT_NAME_DATASET = 'datasetBinary_raw'
COLUMN_NAME = 'ptn_{:02d}'


def update_path(path_file, lim_depth=5, absolute=True):
    """ bubble in the folder tree up intil it found desired file
    otherwise return original one

    :param str path_file:
    :param int lim_depth:
    :return str:

    >>> path = 'sample_file.test'
    >>> f = open(path, 'w')
    >>> update_path(path, absolute=False)
    'sample_file.test'
    >>> os.remove(path)
    """
    if path_file.startswith('/'):
        return path_file
    elif path_file.startswith('~'):
        path_file = os.path.expanduser(path_file)
    else:
        for depth in range(lim_depth):
            if os.path.exists(path_file): break
            path_file = os.path.join('..', path_file)
    if absolute:
        path_file = os.path.abspath(path_file)
    return path_file


def create_elastic_deform_2d(im_size, coef=0.5, grid_size=(20, 20), rand_seed=None):
    """

    :param (int, int) im_size:
    :param float coef:
    :param (int, int) grid_size:
    :return:

    >>> tf = create_elastic_deform_2d((100, 100))
    >>> type(tf)
    <class 'skimage.transform._geometric.PiecewiseAffineTransform'>
    """
    np.random.seed(rand_seed)
    rows, cols = np.meshgrid(np.linspace(0, im_size[0], grid_size[0]),
                             np.linspace(0, im_size[1], grid_size[1]))
    mesh_src = np.dstack([cols.flat, rows.flat])[0]
    # logging.debug(src)
    mesh_dst = mesh_src.copy()
    for i in range(2):
        rnd = np.random.random((mesh_src.shape[0], 1)) - 0.5
        mesh_dst[:, i] += rnd[:, 0] * (im_size[i] / grid_size[i] * coef)
    mesh_dst = filters.gaussian_filter(mesh_dst, 0.1)
    # logging.debug(dst)
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(mesh_src, mesh_dst)
    return tform


def image_deform_elastic(im, coef=0.5, grid_size=(20, 20), rand_seed=None):
    """ deform an image bu elastic transform in size of specific regular grid

    :param np.array<height, width> im: image
    :param float coef: a param describing the how much it is deformed (0 = None)
    :param (int, int) grid_size: is size of elastic grid for deformation
    :return: np.array<height, width>

    >>> img = np.zeros((10, 15), dtype=int)
    >>> img[2:8, 3:7] = 1
    >>> img[6:, 9:] = 2
    >>> image_deform_elastic(img, coef=0.3, grid_size=(5, 5), rand_seed=0)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]], dtype=uint8)
    """
    logging.debug('deform image plane by elastic transform with grid %s',
                 repr(grid_size))
    # logging.debug(im.shape)
    im_size = im.shape[-2:]
    tform = create_elastic_deform_2d(im_size, coef, grid_size, rand_seed)
    if im.ndim == 2:
        bg = np.median(np.hstack([im[0, :], im[:, 0], im[:, -1], im[-1, :]]))
        img = transform.warp(im.astype(float), tform, output_shape=im_size,
                             order=0, cval=bg)
    elif im.ndim == 3:
        bg = np.median(np.hstack([im[0, :, 0], im[:, 0, 0],
                                  im[:, -1, 0], im[-1, :, 0]]))
        im_stack = [transform.warp(im[i].astype(float), tform,
                                   output_shape=im_size, order=0, cval=bg)
                    for i in range(im.shape[0])]
        img = np.array(im_stack)
    else:
        logging.error('not supported image dimension - %s' % repr(im.shape))
    img = np.array(img, dtype=np.uint8)
    return img


def generate_rand_center_radius(img, ratio, rand_seed=None):
    """

    :param np.array<height, width> img:
    :param float ratio:
    :return (int, ), (float, ):

    >>> generate_rand_center_radius(np.zeros((50, 50)), 0.2, rand_seed=0)
    ([27, 15], [8.5, 6.5])
    """
    np.random.seed(rand_seed)
    center, radius = [0] * img.ndim, [0] * img.ndim
    for i in range(img.ndim):
        size = img.shape[i]
        center[i] = np.random.randint(int(1.5 * ratio * size),
                                      int((1. - 1.5 * ratio) * size) + 1)
        radius[i] = np.random.randint(int(0.25 * ratio * size),
                                      int(1. * ratio * size) + 1)
        radius[i] += 0.03 * size
    return center, radius


def draw_ellipse(img, ratio=0.1, color=255, rand_seed=None):
    """ draw an ellipse to image plane with specific value
    SEE: https://en.wikipedia.org/wiki/Ellipse

    :param np.array<height, width> img: while None, create empty one
    :param float ratio: defining size of the ellipse to the image plane
    :param int color: value (0, 255) of an image intensity
    :return: np.array<height, width>

    >>> img = draw_ellipse(np.zeros((10, 15)), ratio=0.3, color=1, rand_seed=0)
    >>> img.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    logging.debug('draw an ellipse to an image with value %i', color)
    center, radius = generate_rand_center_radius(img, ratio, rand_seed)
    x, y = draw.ellipse(center[0], center[1], radius[0], radius[1], shape=img.shape)
    img[x, y] = color
    return img


def draw_ellipsoid(img, ratio=0.1, clr=255, rand_seed=None):
    """ draw an ellipsoid to image plane with specific value
    SEE: https://en.wikipedia.org/wiki/Ellipsoid

    :param float ratio: defining size of the ellipse to the image plane
    :param np.array<depth, height, width> img: image / volume
    :param int clr: value (0, 255) of an image intensity
    :return: np.array<depth, height, width>

    >>> img = draw_ellipsoid(np.zeros((20, 10, 15)), clr=255, rand_seed=0)
    """
    logging.debug('draw an ellipse to an image with value %i', clr)
    center, radius = generate_rand_center_radius(img, ratio, rand_seed)
    vec_dims = [np.arange(0, img.shape[i]) - center[i] for i in range(img.ndim)]
    Z, X, Y = np.meshgrid(*vec_dims, indexing='ij')
    a, b, c = radius
    dist = (Z ** 2 / a ** 2) + (X ** 2 / b ** 2) + (Y ** 2 / c ** 2)
    img[dist < 1.] = clr
    return img


def create_clean_folder(path_dir):
    """ create empty folder and while the folder exist clean all files

    :param str path_dir: path
    :return str:

    >>> path_dir = os.path.abspath('sample_dir')
    >>> path_dir = create_clean_folder(path_dir)
    >>> os.path.exists(path_dir)
    True
    >>> shutil.rmtree(path_dir)
    """
    assert os.path.exists(os.path.dirname(path_dir)), \
        'missing: %s' % os.path.dirname(path_dir)
    logging.info('create clean folder "%s"', path_dir)
    if os.path.exists(path_dir):
        shutil.rmtree(path_dir)
    os.mkdir(path_dir)
    return path_dir


def extract_image_largest_element(img_binary, labeled=None):
    """ take a binary image and find all independent segments,
    then keep just the largest segment and rest set as 0

    :param np.array<height, width> img_binary: image of values {0, 1}
    :return: np.array<height, width> of values {0, 1}

    >>> img = np.zeros((7, 15), dtype=int)
    >>> img[1:4, 5:10] = 1
    >>> img[5:7, 6:13] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    >>> extract_image_largest_element(img)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    if labeled is None or len(np.unique(labeled)) < 2:
        labeled, _ = ndimage.label(img_binary)
    areas = [(j, np.sum(labeled == j)) for j in np.unique(labeled)]
    areas = sorted(areas, key=lambda x: x[1], reverse=True)
    # logging.debug('... elements area: %s', repr(areas))
    img_ptn = img_binary.copy()
    if len(areas) > 1:
        img_ptn = np.zeros_like(img_binary)
        # skip largest, assuming to be background
        img_ptn[labeled == areas[1][0]] = 1
    return img_ptn


def atlas_filter_larges_components(atlas):
    """

    :param np.array<height, width> atlas:
    :return: np.array<height, width>, [np.array<height, width>]

    >>> atlas = np.zeros((7, 15), dtype=int)
    >>> atlas[1:4, 5:10] = 1
    >>> atlas[5:7, 6:13] = 2
    >>> atlas_new, imgs_patterns = atlas_filter_larges_components(atlas)
    >>> atlas
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0]])
    >>> imgs_patterns[1]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=uint8)
    """
    # export to dictionary
    logging.info('... post-processing over generated patterns: %s',
                 repr(np.unique(atlas).tolist()))
    atlas_new = np.zeros(atlas.shape, dtype=np.uint8)
    imgs_patterns = []
    for i, idx in enumerate(np.unique(atlas)[1:]):
        im = np.zeros(atlas.shape, dtype=np.uint8)
        # create pattern
        im[atlas == idx] = 1
        # remove all smaller unconnected elements
        im = extract_image_largest_element(im)
        if np.sum(im) == 0: continue
        imgs_patterns.append(im)
        # add them to the final arlas
        atlas_new[im == 1] = i + 1
    return atlas_new, imgs_patterns


def dictionary_generate_atlas(path_out, dir_name=DIR_NAME_DICTIONARY,
                              nb_ptns=NB_BIN_PATTERNS, im_size=IMAGE_SIZE_2D,
                              temp_img_name=IMAGE_PATTERN):
    """ generate pattern dictionary as atlas, no overlapping

    :param str path_out: path to the results directory
    :param int nb_ptns: number of patterns / labels
    :param (int, int) im_size: image size
    :return: [np.array<height, width>] independent patters in the dictionary

    >>> path_dir = os.path.abspath('sample_dataset')
    >>> path_dir = create_clean_folder(path_dir)
    >>> imgs_patterns = dictionary_generate_atlas(path_dir)
    >>> shutil.rmtree(path_dir)
    """
    logging.info('generate an Atlas composed from %i patterns and image size %s',
                 nb_ptns, repr(im_size))
    out_dir = os.path.join(path_out, dir_name)
    create_clean_folder(out_dir)
    atlas = np.zeros(im_size, dtype=np.uint8)
    for i in range(nb_ptns):
        label = (i + 1)
        if len(im_size) == 2:
            atlas = draw_ellipse(atlas, color=label)
        elif len(im_size) == 3:
            atlas = draw_ellipsoid(atlas, clr=label)
    # logging.debug(type(atlas))
    atlas_def = image_deform_elastic(atlas)
    # logging.debug(np.unique(atlas))
    export_image(out_dir, atlas_def, 'atlas')
    # in case run in DEBUG show atlas and wait till close
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        logging.debug('labels: %s', repr(np.unique(atlas_def)))
        if atlas_def.ndim == 2:
            plt.imshow(atlas_def)
        else:
            plt.imshow(atlas_def[int(atlas_def.shape[0] / 2)])
        plt.show()
    atlas_new, imgs_patterns = atlas_filter_larges_components(atlas_def)
    export_image(out_dir, atlas_new, 'atlas')
    for i, img in enumerate(imgs_patterns):
        export_image(out_dir, img, i, temp_img_name)
    return imgs_patterns


def dictionary_generate_rnd_pattern(path_out=None, dir_name=DIR_NAME_DICTIONARY,
                                    nb_ptns=NB_BIN_PATTERNS, im_size=IMAGE_SIZE_2D,
                                    temp_img_name=IMAGE_PATTERN, rand_seed=None):
    """ generate pattern dictionary and allow overlapping

    :param str path_out: path to the results directory
    :param int nb_ptns: number of patterns / labels
    :param (int, int) im_size: image size
    :return: [np.array<height, width>] list of independent patters in the dictionary

    >>> list_imgs = dictionary_generate_rnd_pattern(nb_ptns=3, im_size=(10, 8), rand_seed=0)
    >>> len(list_imgs)
    3
    >>> list_imgs[1]
    array([[  0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0, 255,   0],
           [  0,   0,   0,   0,   0,   0, 255,   0],
           [  0,   0,   0,   0,   0,   0, 255,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0,   0,   0,   0]], dtype=uint8)
    """
    logging.info('generate Dict. composed from %i patterns and img. size %s',
                 nb_ptns, repr(im_size))
    if path_out is not None:
        out_dir = os.path.join(path_out, dir_name)
        create_clean_folder(out_dir)
    list_imgs = []
    for i in range(nb_ptns):
        im = draw_ellipse(np.zeros(im_size, dtype=np.uint8), rand_seed=rand_seed)
        im = image_deform_elastic(im, rand_seed=rand_seed)
        list_imgs.append(im)
        if path_out is not None:
            export_image(out_dir, im, i, temp_img_name)
    return list_imgs


def generate_rand_patterns_occlusion(idx, im_ptns, out_dir=None,
                                     ptn_ration=RND_PATTERN_OCCLUSION, rand_seed=None):
    """ generate the new sample from list of pattern with specific ration

    :param int idx: index
    :param [np.array] im_ptns: images with patterns
    :param str out_dir: name of directory
    :param float ptn_ration: number in range (0, 1)
    :return: int, np.array, str, [int]

    >>> img1 = np.zeros((6, 15), dtype=int)
    >>> img1[2:5, 5:10] = 1
    >>> img2 = np.zeros((6, 15), dtype=int)
    >>> img2[3:6, 2:13] = 1
    >>> idx, im, im_name, ptn_weights = generate_rand_patterns_occlusion(
    ...                                     0, [img1, img2], rand_seed=0)
    >>> idx
    0
    >>> im
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    >>> im_name
    'sample_00000'
    >>> ptn_weights
    [0, 1]
    """
    # reinit seed to have random samples even in the same time
    np.random.seed(rand_seed)
    bool_combine = np.random.random(len(im_ptns)) < ptn_ration
    # if there is non above threshold select one random
    if not any(bool_combine):
        bool_combine[np.random.randint(0, len(bool_combine))] = True
    logging.debug('combination vector is %s', repr(bool_combine.tolist()))
    im = sum(np.asarray(im_ptns)[bool_combine])
    # convert sum to union such as all above 0 set as 1
    im[im > 0.] = 1
    im = im.astype(im_ptns[0].dtype)
    im_name = SEGM_PATTERN.format(idx)
    if out_dir is not None and os.path.exists(out_dir):
        export_image(out_dir, im, idx)
    ptn_weights = [int(x) for x in bool_combine]
    return idx, im, im_name, ptn_weights


def dataset_binary_combine_patterns(im_ptns, out_dir=None, nb_samples=NB_SAMPLES,
                                    ptn_ration=RND_PATTERN_OCCLUSION,
                                    nb_jobs=NB_THREADS, rand_seed=None):
    """ generate a Binary dataset composed from N samples and given ration
    of pattern occlusion

    :param int nb_jobs: number of running jobs
    :param [np.array<height, width>] im_ptns: list of ind. patters in the dictionary
    :param str out_dir: path to the results directory
    :param int nb_samples: number of samples in dataset
    :param float ptn_ration: ration of how many patterns are used to create
        an input observation / image
    :return: [np.array<height, width>], df<nb_imgs, nb_lbs>

    >>> img1 = np.zeros((6, 15), dtype=int)
    >>> img1[2:5, 5:10] = 1
    >>> img2 = np.zeros((6, 15), dtype=int)
    >>> img2[3:6, 2:13] = 1
    >>> im_spls, df_weights = dataset_binary_combine_patterns([img1, img2],
    ...                                             nb_samples=5, rand_seed=0)
    >>> len(im_spls)
    5
    >>> im_spls[1]
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    >>> df_weights  # doctest: +NORMALIZE_WHITESPACE
                  ptn_01  ptn_02
    image
    sample_00000     0.0     1.0
    sample_00001     0.0     1.0
    sample_00002     0.0     1.0
    sample_00003     0.0     1.0
    sample_00004     0.0     1.0
    """
    logging.info('generate a Binary dataset composed from %i samples  '
                'and ration pattern occlusion %f', nb_samples, ptn_ration)
    if out_dir is not None:
        create_clean_folder(out_dir)
    df_weights = pd.DataFrame()
    im_spls = [None] * nb_samples
    mproc_pool = mproc.Pool(nb_jobs)
    logging.debug('running in %i threads...', nb_jobs)
    tqdm_bar = tqdm.tqdm(total=nb_samples)
    wrapper_generate = partial(generate_rand_patterns_occlusion,
                               im_ptns=im_ptns, out_dir=out_dir,
                               ptn_ration=ptn_ration, rand_seed=rand_seed)
    for idx, im, im_name, ptn_weights in mproc_pool.imap_unordered(
                                            wrapper_generate, range(nb_samples)):
        im_spls[idx] = im
        df_weights = df_weights.append(pd.Series([im_name] + ptn_weights),
                                       ignore_index=True)

        tqdm_bar.update(1)
    mproc_pool.close()
    mproc_pool.join()
    df_weights.columns = ['image'] + [COLUMN_NAME.format(i + 1)
                                      for i in range(len(df_weights.columns) - 1)]
    df_weights.sort_values(['image'], inplace=True)
    df_weights.set_index('image', inplace=True)
    logging.debug(df_weights.head())
    return im_spls, df_weights


def add_image_binary_noise(im, ration=0.1, rand_seed=None):
    """ generate and add a binary noise to an image

    :param np.array<height, width> im: input binary image
    :param float ration: number (0, 1) means 0 = no noise
    :return: np.array<height, width> binary image

    >>> img = np.zeros((5, 15), dtype=int)
    >>> img[1:4, 3:7] = 1
    >>> add_image_binary_noise(img, ration=0.1, rand_seed=0)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]], dtype=int16)
    """
    logging.debug('... add random noise to a binary image')
    np.random.seed(rand_seed)
    rnd = np.random.random(im.shape)
    rnd = np.array(rnd < ration, dtype=np.int16)
    im_noise = np.abs(np.asanyarray(im, dtype=np.int16) - rnd)
    # plt.subplot(1,3,1), plt.imshow(im)
    # plt.subplot(1,3,2), plt.imshow(rnd)
    # plt.subplot(1,3,3), plt.imshow(im - rnd)
    # plt.show()
    return np.array(im_noise, dtype=np.int16)


def export_image(path_out, img, im_name, name_template=SEGM_PATTERN,
                 stretch_range=True):
    """ export an image with given path and optional pattern for image name

    :param str path_out: path to the results directory
    :param np.array<height, width> img: image
    :param str/int im_name: image nea of index to be place to patterns name
    :param str name_template: str, while the name is not string generate image according
        specific pattern, like format fn
    :return str: path to the image

    Image - PNG
    >>> np.random.seed(0)
    >>> img = np.random.random([5, 10])
    >>> path_img = export_image('.', img, 'testing-image')
    >>> path_img
    './testing-image.png'
    >>> os.path.exists(path_img)
    True
    >>> name, im = load_image(path_img)
    >>> im.shape
    (5, 10)
    >>> np.round(im, 1)
    array([[ 0.6,  0.7,  0.6,  0.6,  0.4,  0.7,  0.4,  0.9,  1. ,  0.4],
           [ 0.8,  0.5,  0.6,  0.9,  0.1,  0.1,  0. ,  0.8,  0.8,  0.9],
           [ 1. ,  0.8,  0.5,  0.8,  0.1,  0.7,  0.1,  1. ,  0.5,  0.4],
           [ 0.3,  0.8,  0.5,  0.6,  0. ,  0.6,  0.6,  0.6,  1. ,  0.7],
           [ 0.4,  0.4,  0.7,  0.1,  0.7,  0.7,  0.2,  0.1,  0.3,  0.4]])
    >>> os.remove(path_img)

    Image - TIFF
    >>> img = np.random.random([5, 20, 25])
    >>> path_img = export_image('.', img, 'testing-image')
    >>> path_img
    './testing-image.tiff'
    >>> os.path.exists(path_img)
    True
    >>> name, im = load_image(path_img)
    >>> im.shape
    (5, 20, 25)
    >>> os.remove(path_img)
    """
    assert img.ndim >= 2, 'wrong image dim: %s' % repr(img.shape)
    if not os.path.exists(path_out):
        return ''
    if not isinstance(im_name, str):
        im_name = name_template.format(im_name)
    path_img = os.path.join(path_out, im_name)
    logging.debug(' .. saving image %s with %s to "%s"', repr(img.shape),
                  repr(np.unique(img)), path_img)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3):
        if stretch_range and img.max() > 0:
            img = img / float(img.max()) * 255
        # io.imsave(path_img, im_norm)
        path_img += '.png'
        Image.fromarray(img.astype(np.uint8)).save(path_img)
    elif img.ndim == 3:
        if stretch_range and img.max() > 0:
            img = img / float(img.max()) * 255 ** 2
        path_img += '.tiff'
        io.imsave(path_img, img)
        # tif = libtiff.TIFF.open(path_img, mode='w')
        # tif.write_image(img_clip.astype(np.uint16))
    else:
        logging.warning('not supported image format: %s', repr(img.shape))
    return path_img


def wrapper_apply_function(i_img, func, coef, out_dir):
    """

    :param (int, np.array<height, width>) i_img:
    :param func:
    :param float coef:
    :param str out_dir:
    :return: int, np.array<height, width>
    """
    i, img = i_img
    img_def = func(img, coef)
    export_image(out_dir, img_def, i)
    return i, img_def


def dataset_apply_image_function(imgs, out_dir, func, coef=0.5, nb_jobs=NB_THREADS):
    """ having list if input images create an dataset with randomly deform set
    of these images and export them to the results folder

    :param int nb_jobs:
    :param func:
    :param [np.array<height, width>] imgs: raw input images
    :param str out_dir: path to the results directory
    :param float coef: a param describing the how much it is deformed (0 = None)
    :return: [np.array<height, width>]
    """
    logging.info('apply costume function "%s" on %i samples with coef. %f',
                 func.__name__, len(imgs), coef)
    create_clean_folder(out_dir)

    imgs_new = [None] * len(imgs)
    mproc_pool = mproc.Pool(nb_jobs)
    logging.debug('running in %i threads...', nb_jobs)
    tqdm_bar = tqdm.tqdm(total=len(imgs))
    for i, im in mproc_pool.imap_unordered(partial(wrapper_apply_function,
                                        func=func, coef=coef, out_dir=out_dir),
                                enumerate(imgs)):
        imgs_new[i] = im
        tqdm_bar.update(1)
    mproc_pool.close()
    mproc_pool.join()

    return imgs_new


def image_transform_binary2prob(im, coef=0.1):
    """ convert a binary image to probability while computing distance function
    on the binary function (contours)

    :param np.array<height, width> im: input binary image
    :param float coef: float, influence hoe strict the boundary between F-B is
    :return: np.array<height, width> float image

    >>> img = np.zeros((5, 15), dtype=int)
    >>> img[1:4, 3:7] = 1
    >>> img
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> img_prob = image_transform_binary2prob(img, coef=0.1)
    >>> np.round(img_prob[2, :], 2).tolist() # doctest: +NORMALIZE_WHITESPACE
    [0.43, 0.45, 0.48, 0.52, 0.55, 0.55, 0.52, 0.48, 0.45, 0.43, 0.4, 0.38,
     0.35, 0.33, 0.31]
    """
    logging.debug('... transform binary image to probability')
    im_dist = ndimage.distance_transform_edt(im)
    im_dist -= ndimage.distance_transform_edt(1 - im)
    im_prob = 1. / (1. + np.exp(-coef * im_dist))
    # plt.subplot(1,3,1), plt.imshow(im)
    # plt.subplot(1,3,2), plt.imshow(im_dist)
    # plt.subplot(1,3,3), plt.imshow(im_prob)
    # plt.show()
    return im_prob


def add_image_prob_pepper_noise(im, ration=0.1, rand_seed=None):
    """ generate and add a continues noise to an image

    :param np.array<height, width> im: input float image
    :param float ration: number means 0 = no noise
    :return: np.array<height, width> float image

    >>> img = np.zeros((5, 9), dtype=int)
    >>> img[1:4, 2:7] = 1
    >>> img = add_image_prob_pepper_noise(img, ration=0.5, rand_seed=0)
    >>> np.round(img, 2)
    array([[ 0.1 ,  0.43,  0.21,  0.09,  0.15,  0.29,  0.12,  0.  ,  0.  ],
           [ 0.23,  0.  ,  0.94,  0.86,  1.  ,  1.  ,  1.  ,  0.  ,  0.  ],
           [ 0.  ,  0.  ,  1.  ,  1.  ,  1.08,  1.  ,  1.  ,  0.28,  0.  ],
           [ 0.  ,  0.04,  1.17,  1.47,  1.  ,  1.09,  0.86,  0.  ,  0.24],
           [ 0.22,  0.23,  0.  ,  0.36,  0.28,  0.13,  0.4 ,  0.  ,  0.33]])
    """
    logging.debug('... add smooth noise to a probability image')
    np.random.seed(rand_seed)
    rnd = 2 * (np.random.random(im.shape) - 0.5)
    rnd[abs(rnd) > ration] = 0
    im_noise = np.abs(im - rnd)
    # plt.subplot(1,3,1), plt.imshow(im)
    # plt.subplot(1,3,2), plt.imshow(rnd)
    # plt.subplot(1,3,3), plt.imshow(im - rnd)
    # plt.show()
    return im_noise


def add_image_prob_gauss_noise(im, sigma=0.1, rand_seed=None):
    """ generate and add a continues noise to an image

    :param np.array<height, width> im: input float image
    :param float ration: float (0, 1) means 0 = no noise
    :return: np.array<height, width> float image

    >>> img = np.zeros((5, 9), dtype=int)
    >>> img[1:4, 2:7] = 1
    >>> img = add_image_prob_gauss_noise(img, sigma=0.5, rand_seed=0)
    >>> np.round(img, 2)
    array([[ 0.88,  0.2 ,  0.49,  1.  ,  0.93,  0.  ,  0.48,  0.  ,  0.  ],
           [ 0.21,  0.07,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  0.75,  0.  ],
           [ 0.16,  0.  ,  0.  ,  1.  ,  1.  ,  0.63,  1.  ,  0.  ,  0.02],
           [ 0.  ,  0.77,  1.  ,  1.  ,  1.  ,  0.56,  0.01,  0.  ,  0.08],
           [ 0.62,  0.6 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.98,  0.  ]])
    """
    logging.debug('... add Gauss. noise to a probability image')
    np.random.seed(rand_seed)
    noise = np.random.normal(0., scale=sigma, size=im.shape)
    im_noise = im.astype(float) + noise
    im_noise[im_noise < 0] = 0.
    im_noise[im_noise > 1] = 1.
    return im_noise


def wrapper_load_images(list_path_img):
    """

    :param [str] list_path_img:
    :return: [(str, np.array<height, width>)]
    """
    logging.debug('parallel loading %i images', len(list_path_img))
    list_names_imgs = map(load_image, list_path_img)
    return list_names_imgs


def find_images(path_dir, im_pattern='*', img_posixs=IMAGE_POSIXS):
    """ in given folder find largest group of equal images types

    :param str path_dir:
    :param str im_pattern:
    :param [str] img_posixs:
    :return [str]:

    >>> np.random.seed(0)
    >>> img = np.random.random([5, 10])
    >>> path_img = export_image('.', img, 'testing-image')
    >>> find_images(os.path.dirname(path_img))
    ['./testing-image.png']
    >>> os.remove(path_img)
    """
    paths_img_most = []
    for im_posix in img_posixs:
        path_search = os.path.join(path_dir, im_pattern + im_posix)
        paths_img = glob.glob(path_search)
        logging.debug('images found %i for search "%s"', len(paths_img), path_search)
        if len(paths_img) > len(paths_img_most):
            paths_img_most = paths_img
    return paths_img_most


def dataset_load_images(path_dir, im_pattern='*', nb_spls=None, path_imgs=None, nb_jobs=1):
    """ load complete dataset or just a subset

    :param str name: name od particular dataset
    :param str path_base: path to the results directory
    :param str im_pattern: specific pattern of loaded images
    :param str im_posix: image pattern line '.png'
    :param int nb_spls: number of samples to be loaded, None means all
    :param int nb_jobs:
    :return [np.array], [str]:
    """
    logging.debug('loading folder (%s) <- "%s"',
                  os.path.exists(path_dir), path_dir)
    assert os.path.exists(path_dir), '%s' % path_dir
    if path_imgs is None:
        path_imgs = find_images(path_dir, im_pattern)
    assert all(os.path.exists(p) for p in path_imgs)
    path_imgs = sorted(path_imgs)[:nb_spls]
    logging.debug('number samples %i in dataset "%s"', len(path_imgs),
                  os.path.basename(path_dir))
    if nb_jobs > 1:
        logging.debug('running in %i threads...', nb_jobs)
        nb_load_blocks = len(path_imgs) / float(BLOCK_NB_LOAD_IMAGES)
        nb_load_blocks = int(np.ceil(nb_load_blocks))
        logging.debug('estimated %i loading blocks', nb_load_blocks)
        block_paths_img = (path_imgs[i::nb_load_blocks]
                           for i in range(nb_load_blocks))

        mproc_pool = mproc.Pool(nb_jobs)
        list_names_imgs = mproc_pool.map(wrapper_load_images, block_paths_img)
        mproc_pool.close()
        mproc_pool.join()

        logging.debug('transforming the parallel results')
        names_imgs = sorted(itertools.chain(*list_names_imgs))
        im_names, imgs = zip(*names_imgs)
    else:
        logging.debug('running in single thread...')
        names_imgs = [load_image(p) for p in path_imgs]
        logging.debug('split the resulting tuples')
        im_names, imgs = zip(*names_imgs)
    assert len(path_imgs) == len(imgs)
    return imgs, im_names


def load_image(path_img, bool_val=True):
    """

    :param str path_img:
    :return str, np.array<height, width>:

    PNG image
    >>> img_name = 'testing_image'
    >>> img = np.random.randint(0, 255, size=(20, 20))
    >>> path_img = export_image('.', img, img_name, stretch_range=False)
    >>> path_img
    './testing_image.png'
    >>> os.path.exists(path_img)
    True
    >>> _, img_new = load_image(path_img, bool_val=False)
    >>> np.array_equal(img, img_new)
    True
    >>> os.remove(path_img)

    TIFF image
    >>> img_name = 'testing_image'
    >>> img = np.random.randint(0, 255, size=(5, 20, 20))
    >>> path_img = export_image('.', img, img_name, stretch_range=False)
    >>> path_img
    './testing_image.tiff'
    >>> os.path.exists(path_img)
    True
    >>> _, img_new = load_image(path_img, bool_val=False)
    >>> img_new.shape
    (5, 20, 20)
    >>> np.array_equal(img, img_new)
    True
    >>> os.remove(path_img)
    """
    assert os.path.exists(path_img), path_img
    n_img, img_ext = os.path.splitext(os.path.basename(path_img))
    if img_ext in ['.tif', '.tiff']:
        img = io.imread(path_img)
        # im = libtiff.TiffFile().get_tiff_array()
        # img = np.empty(im.shape)
        # for i in range(img.shape[0]):
        #     img[i, :, :] = im[i]
        # img = np.array(img.tolist())
    else:
        img = io.imread(path_img)
        # img = np.array(Image.open(path_img))
    if bool_val and img.max() > 0:
        img = (img / float(img.max()))
    return n_img, img


def dataset_load_weights(path_base, name_csv=CSV_NAME_WEIGHTS, img_names=None):
    """ loading all true weights for given dataset

    :param str path_base: path to the results directory
    :param str name_csv: name of file with weights
    :return: np.array<nb_imgs, nb_lbs>
    """
    path_csv = os.path.join(path_base, name_csv)
    assert os.path.exists(path_csv), 'missing %s' % path_csv
    df = pd.DataFrame().from_csv(path_csv)
    # load according a list
    if img_names is not None:
        df = df[df['name'].isin(img_names)]
    # for the original encoding as string in single column
    if 'combination' in df.columns:
        coding = df['combination'].values.tolist()
        logging.debug('encoding of length: %i', len(coding))
        encoding = np.array([[int(x) for x in c.split(';')] for c in coding])
    # the new encoding with pattern names
    else:
        encoding = df.as_matrix()
    return np.array(encoding)


def dataset_compose_atlas(path_base, name=DIR_NAME_DICTIONARY,
                          img_temp_name='pattern_*'):
    """ load all independent patterns and compose them into single m-label atlas

    :param str name: name of dataset
    :param str path_base: path to the results directory
    :param str img_temp_name:
    :return: np.array<height, width>
    """
    imgs, _ = dataset_load_images(os.path.join(path_base, name),
                                  im_pattern=img_temp_name)
    assert len(imgs) > 0
    atlas = np.zeros_like(imgs[0])
    for i, im in enumerate(imgs):
        atlas[im == 1] = i+1
    return np.array(atlas, dtype=np.uint8)


def dataset_export_images(path_out, imgs, names=None, nb_jobs=1):
    """ export complete dataset

    :param str path_out:
    :param [np.array<height, width>] imgs:
    :param names: [str] or None (use indexes)
    :param int nb_jobs:

    >>> np.random.seed(0)
    >>> images = [np.random.random([15, 10]) for i in range(36)]
    >>> path_dir = os.path.abspath('sample_dataset')
    >>> os.mkdir(path_dir)
    >>> dataset_export_images(path_dir, images, nb_jobs=2)
    >>> imgs, im_names = dataset_load_images(path_dir, nb_jobs=2)
    >>> len(imgs)
    36
    >>> np.round(imgs[0], 1)
    array([[ 0.5,  0.7,  0.6,  0.5,  0.4,  0.6,  0.4,  0.9,  1. ,  0.4],
           [ 0.8,  0.5,  0.6,  0.9,  0.1,  0.1,  0. ,  0.8,  0.8,  0.9],
           [ 1. ,  0.8,  0.5,  0.8,  0.1,  0.6,  0.1,  0.9,  0.5,  0.4],
           [ 0.3,  0.8,  0.5,  0.6,  0. ,  0.6,  0.6,  0.6,  0.9,  0.7],
           [ 0.4,  0.4,  0.7,  0.1,  0.7,  0.7,  0.2,  0.1,  0.3,  0.4],
           [ 0.6,  0.4,  1. ,  0.1,  0.2,  0.2,  0.7,  0.3,  0.5,  0.2],
           [ 0.2,  0.1,  0.7,  0.1,  0.2,  0.4,  0.8,  0.1,  0.8,  0.1],
           [ 1. ,  0.5,  1. ,  0.6,  0.7,  0. ,  0.3,  0.1,  0.3,  0.1],
           [ 0.3,  0.4,  0.1,  0.7,  0.6,  0.3,  0.5,  0.1,  0.6,  0.9],
           [ 0.3,  0.7,  0.1,  0.7,  0.3,  0.2,  0.6,  0. ,  0.8,  0. ],
           [ 0.7,  0.3,  0.7,  1. ,  0.2,  0.6,  0.6,  0.6,  0.2,  1. ],
           [ 0.4,  0.8,  0.7,  0.3,  0.8,  0.4,  0.9,  0.6,  0.9,  0.7],
           [ 0.7,  0.5,  1. ,  0.6,  0.4,  0.6,  0. ,  0.3,  0.7,  0.3],
           [ 0.6,  0.4,  0.1,  0.3,  0.6,  0.6,  0.6,  0.7,  0.7,  0.4],
           [ 0.9,  0.4,  0.4,  0.9,  0.8,  0.7,  0.1,  0.9,  0.7,  1. ]])
    >>> im_names   # doctest: +ELLIPSIS
    ('sample_00000', 'sample_00001', ..., 'sample_00034', 'sample_00035')
    >>> shutil.rmtree(path_dir)
    """
    create_clean_folder(path_out)
    logging.debug('export %i images into "%s"', len(imgs), path_out)
    if names is None:
        names = range(len(imgs))

    mp_set = [(path_out, im, names[i]) for i, im in enumerate(imgs)]
    if nb_jobs > 1:
        logging.debug('running in %i threads...', nb_jobs)
        mproc_pool = mproc.Pool(nb_jobs)
        mproc_pool.map(wrapper_export_image, mp_set)
        mproc_pool.close()
        mproc_pool.join()
    else:
        logging.debug('running in single thread...')
        map(wrapper_export_image, mp_set)


def wrapper_export_image(mp_set):
    export_image(*mp_set)


# def dataset_convert_nifti(path_in, path_out, img_posix=IMAGE_POSIX):
#     """ having a datset of png images conver them into nifti images
#
#     :param path_in: str
#     :param path_out: str
#     :param img_posix: str, like '.png'
#     :return:
#     """
#     import src.own_utils.data_io as tl_data
#     logging.info('convert a dataset to Nifti')
#     p_imgs = glob.glob(os.path.join(path_in, '*' + img_posix))
#     create_clean_folder(path_out)
#     p_imgs = sorted(p_imgs)
#     for path_im in p_imgs:
#         name = os.path.splitext(os.path.basename(path_im))[0]
#         path_out = os.path.join(path_out, name)
#         logging.debug('... converting "%s" -> "%s"', path_im, path_out)
#         tl_data.convert_img_2_nifti_gray(path_im, path_out)
#     return None


def create_simple_atlas():
    """ create a simple atlas with split 3 patterns

    :return np.array:

    >>> create_simple_atlas()
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    atlas = np.zeros((20, 20), dtype=int)
    atlas[2:8, 2:8] = 1
    atlas[12:18, 12:18] = 2
    atlas[2:8, 12:18] = 3
    return atlas


def create_sample_images(atlas):
    """ create 3 simple images according simple atlas with 3 patterns

    :param np.array atlas:
    :return [np.array]:

    >>> atlas = create_simple_atlas()
    >>> im1, im2, im3 = create_sample_images(atlas)
    >>> im2
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    im1 = atlas.copy()
    im1[im1 >= 2] = 0
    im2 = atlas.copy()
    im2[im2 <= 1] = 0
    im2[im2 > 0] = 1
    im3 = atlas.copy()
    im3[atlas < 2] = 0
    im3[atlas > 2] = 0
    im3[im3 > 0] = 1
    return im1, im2, im3