"""
cut the minimal image size over whole set

EXAMPLES:
>> python run_cut_minimal_images.py \
    -in "images/imaginal_discs/gene/*.png" \
    -out images/imaginal_discs/gene_cut

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import json
import argparse
import logging
import multiprocessing as mproc

import tqdm
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_data

NB_THREADS = int(mproc.cpu_count() * .75)
NAME_JSON_BBOX = 'cut_bounding_box.json'
PARAMS = {
    'path_in': os.path.join(tl_data.update_path('images/imaginal_discs/gene'),
                            '*.png'),
    'path_out': tl_data.update_path('images/imaginal_discs/gene_cut'),
}


def args_parse_params(params):
    """ create simple arg parser with default values (input, output)

    :param {str: ...} dict_params:
    :return: object argparse<...>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--path_in', type=str, required=True,
                        default=params['path_in'],
                        help='path to the folder with input image dataset')
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        default=params['path_out'],
                        help='path to the output with experiment results')
    parser.add_argument('-thr', '--threshold', type=float, required=False,
                        default=0, help='threshold for image processing')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        default=NB_THREADS, help='number of parallel processes')

    args = vars(parser.parse_args())
    for k in (k for k in args if k.startswith('path_')):
        p = tl_data.update_path(os.path.dirname(args[k]))
        assert os.path.exists(p), 'missing (%s): %s' % (k, p)
        args[k] = os.path.join(p, os.path.basename(args[k]))
    return args


def find_min_bbox(img, threshold=0):
    bbox = []
    for _ in range(4):
        means = np.mean(img, axis=0)
        for i in range(len(means)):
            if np.sum(means[:i]) > threshold:
                bbox.append(i)
                break
        img = np.rot90(img)
    return bbox


def export_bbox_json(path_dir, bbox, name=NAME_JSON_BBOX):
    d_bbox = dict(zip(['left', 'top', 'right', 'bottom'], bbox))
    path_json = os.path.join(path_dir, name)
    logging.info('exporting JSON: %s', path_json)
    with open(path_json, 'w') as fp:
        json.dump(d_bbox, fp)
    return d_bbox


def main(path_pattern_in, path_out, nb_jobs=NB_THREADS):
    assert os.path.isdir(os.path.dirname(path_pattern_in)), \
        'missing: %s' % path_pattern_in
    assert os.path.isdir(os.path.dirname(path_out)), \
        'missing: %s' % os.path.dirname(path_out)

    if not os.path.isdir(path_out):
        logging.info('create dir: %s', path_out)
        os.mkdir(path_out)

    list_img_paths = glob.glob(path_pattern_in)
    logging.info('found images: %i', len(list_img_paths))

    imgs, im_names = tl_data.dataset_load_images(list_img_paths, nb_jobs=nb_jobs)
    im_mean = np.mean(np.array(imgs), axis=0)
    logging.info('original image size: %s', repr(im_mean.shape))
    bbox = find_min_bbox(im_mean, params['threshold'])
    d_bbox = export_bbox_json(path_out, bbox)
    logging.info('found BBox: %s', repr(d_bbox))

    for im, name in tqdm.tqdm(zip(imgs, im_names)):
        im_cut = im[d_bbox['left']:-d_bbox['right'], d_bbox['top']:-d_bbox['bottom']]
        tl_data.export_image(path_out, im_cut, name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = args_parse_params(PARAMS)
    main(params['path_in'], params['path_out'],
         nb_jobs=params['nb_jobs'])

    logging.info('DONE')
