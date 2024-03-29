"""
cut the minimal image size over whole set

EXAMPLES:
>> python run_cut_minimal_images.py \
    -i "./data_images/imaginal_discs/gene/*.png" \
    -o ./data_images/imaginal_discs/gene_cut

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import argparse
import glob
import json
import logging
import multiprocessing as mproc
import os
import sys
from functools import partial

import numpy as np
from imsegm.utilities.data_io import update_path
from imsegm.utilities.experiments import WrapExecuteSequence
from scipy.ndimage import filters

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bpdl.data_utils import export_image, load_image
from bpdl.utilities import estimate_rolling_ball

NB_WORKERS = int(mproc.cpu_count() * .75)
NAME_JSON_BBOX = 'cut_bounding_box.json'
LOAD_SUBSET_COEF = 5
METHODS = ['cum-info', 'line-sum', 'line-grad']
DEFAULT_PARAMS = {
    'path_in': os.path.join(update_path('data_images'), 'imaginal_discs', 'gene', '*.png'),
    'path_out': os.path.join(update_path('data_images'), 'imaginal_discs', 'gene_cut'),
}


def args_parse_params(params):
    """ create simple arg parser with default values (input, output)

    :param dict dict_params:
    :return obj: object argparse<...>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--path_in',
        type=str,
        required=True,
        default=params['path_in'],
        help='path to the folder with input image dataset'
    )
    parser.add_argument(
        '-o',
        '--path_out',
        type=str,
        required=True,
        default=params['path_out'],
        help='path to the output with experiment results'
    )
    parser.add_argument(
        '-t', '--threshold', type=float, required=False, default=0.001, help='threshold for image information'
    )
    parser.add_argument(
        '-m', '--thr_method', type=str, required=False, default='', choices=METHODS, help='used methods'
    )
    parser.add_argument(
        '--nb_workers', type=int, required=False, default=NB_WORKERS, help='number of parallel processes'
    )

    args = vars(parser.parse_args())
    for k in (k for k in args if k.startswith('path_')):
        p = update_path(os.path.dirname(args[k]))
        assert os.path.exists(p), 'missing (%s): %s' % (k, p)
        args[k] = os.path.join(p, os.path.basename(args[k]))
    return args


def load_mean_image(paths_img):
    img_cum = None
    for p_img in paths_img:
        _, img = load_image(p_img, fuzzy_val=True)
        img = img.astype(np.float64)
        if img_cum is None:
            img_cum = img
        else:
            img_cum = img_cum + img
    img_mean = img_cum / float(len(paths_img))
    return img_mean


def check_bounding_box(bbox, img_size):
    for i in range(2):
        # if left cut is over right cut reset it
        if bbox[i] > (img_size[i] - bbox[i + 2]):
            logging.debug('reset BBox (%i, %i) for size %i', bbox[i], bbox[i + 2], img_size[i])
            bbox[i] = 0
            bbox[i + 2] = 0
    return bbox


def find_min_bbox_cumul_sum(img, threshold=0):
    logging.info('find bbox using cumulative info with thr=%f', threshold)
    img_norm = img / np.sum(img)
    bbox = []
    for _ in range(4):
        rows = np.sum(img_norm, axis=0)
        for i, _ in enumerate(rows):
            if np.sum(rows[:i]) >= threshold:
                bbox.append(i)
                break
        img_norm = np.rot90(img_norm)
    return bbox


def find_min_bbox_line_sum(img, threshold=0):
    logging.info('find bbox using line sum with thr=%f', threshold)
    bbox = []
    for _ in range(4):
        rows = np.mean(img, axis=0)
        rows = rows / np.max(rows)
        for i, r in enumerate(rows):
            if r >= threshold:
                bbox.append(i)
                break
        img = np.rot90(img)
    return bbox


def find_min_bbox_grad(img):
    logging.info('find bbox using Gradient')
    bbox = []
    for _ in range(4):
        rows = np.mean(img, axis=0)
        rows_cum = np.cumsum(rows / np.sum(rows))
        rows_cum = filters.gaussian_filter1d(rows_cum, sigma=1)

        pts = np.array(list(zip(range(len(rows_cum)), rows_cum)))
        diams = estimate_rolling_ball(pts, tangent_smooth=1)
        bbox.append(np.argmin(diams[0]))

        img = np.rot90(img)
    return bbox


def export_bbox_json(path_dir, bbox, name=NAME_JSON_BBOX):
    d_bbox = dict(zip(['left', 'top', 'right', 'bottom'], bbox))
    path_json = os.path.join(path_dir, name)
    logging.info('exporting JSON: %s', path_json)
    with open(path_json, 'w') as fp:
        json.dump(d_bbox, fp)
    return d_bbox


def export_cut_image(path_img, d_bbox, path_out):
    name, im = load_image(path_img)
    im_cut = im[d_bbox['top']:-d_bbox['bottom'], d_bbox['left']:-d_bbox['right']]
    export_image(path_out, im_cut, name)


def main(path_pattern_in, path_out, nb_workers=NB_WORKERS):
    assert os.path.isdir(os.path.dirname(path_pattern_in)), 'missing: %s' % path_pattern_in
    assert os.path.isdir(os.path.dirname(path_out)), 'missing: %s' % os.path.dirname(path_out)

    if not os.path.isdir(path_out):
        logging.info('create dir: %s', path_out)
        os.mkdir(path_out)

    list_img_paths = glob.glob(path_pattern_in)
    logging.info('found images: %i', len(list_img_paths))

    # create partial subset with image pathes
    list_img_paths_partial = [
        list_img_paths[i::nb_workers * LOAD_SUBSET_COEF] for i in range(nb_workers * LOAD_SUBSET_COEF)
    ]
    list_img_paths_partial = [ls for ls in list_img_paths_partial if ls]
    mean_imgs = list(
        WrapExecuteSequence(load_mean_image, list_img_paths_partial, nb_workers=nb_workers, desc='loading mean images')
    )
    # imgs, im_names = tl_data.dataset_load_images(list_img_paths, nb_workers=1)
    img_mean = np.mean(np.asarray(mean_imgs), axis=0)
    export_image(path_out, img_mean, 'mean_image')

    logging.info('original image size: %r', img_mean.shape)
    # bbox = find_min_bbox_cumul_sum(img_mean, params['threshold'])
    if params['thr_method'] == 'line-grad':
        bbox = find_min_bbox_grad(img_mean)
    elif params['threshold'] == 0:
        bbox = [0] * 4
    elif params['thr_method'] == 'line-sum':
        bbox = find_min_bbox_line_sum(img_mean, params['threshold'])
    else:
        bbox = find_min_bbox_cumul_sum(img_mean, params['threshold'])
    d_bbox = export_bbox_json(path_out, bbox)
    logging.info('found BBox: %r', d_bbox)

    _cut_export = partial(export_cut_image, d_bbox=d_bbox, path_out=path_out)
    list(WrapExecuteSequence(_cut_export, list_img_paths, nb_workers, desc='exporting cut images'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = args_parse_params(DEFAULT_PARAMS)
    main(params['path_in'], params['path_out'], nb_workers=params['nb_workers'])

    logging.info('DONE')
