"""
Simple script for adding Gaussian noise to already generated images

>> python run_dataset_add_noise.py -p images -d syntheticDataset_vX

>> python run_dataset_add_noise.py -p ~/Medical-drosophila/synthetic_data \
    -d apdDataset_v0 apdDataset_v1 apdDataset_v2

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import logging
import argparse
import multiprocessing as mproc
from functools import partial

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.utilities as utils
import bpdl.data_utils as tl_data

NB_THREADS = int(mproc.cpu_count() * 0.7)
IMAGE_PATTERN = '*.png'
DIR_POSIX = '_gauss-%.3f'
NOISE_RANGE = tl_data.GAUSS_NOISE
LIST_DATASETS = [tl_data.DIR_MANE_SYNTH_DATASET]
BASE_IMAGE_SET = 'datasetFuzzy_raw'


def args_parser():
    """ create simple arg parser with default values (input, results, dataset)

    :return obj: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='path to set of experiments')
    parser.add_argument('-d', '--datasets', type=str, required=False, nargs='+',
                        help='result file name', default=LIST_DATASETS)
    parser.add_argument('-s', '--sigma', type=str, required=False, nargs='+',
                        help='Gaussian sigma of additive noise',
                        default=NOISE_RANGE)

    args = vars(parser.parse_args())
    args['path'] = tl_data.update_path(args['path'])
    assert os.path.isdir(args['path']), 'missing: %s' % args['path']

    return args


def add_noise_image(img_name, path_in, path_out, noise_level):
    """

    :param str img_name:
    :param str path_in:
    :param str path_out:
    :param float noise_level:
    """
    path_img = os.path.join(path_in, img_name)
    logging.debug('loading image: %s', path_img)
    name, img = tl_data.load_image(path_img)
    img_noise = tl_data.add_image_fuzzy_gauss_noise(img, noise_level)
    tl_data.export_image(path_out, img_noise, name)


def dataset_add_noise(path_in, path_out, noise_level,
                      img_pattern=IMAGE_PATTERN, nb_jobs=NB_THREADS):
    """

    :param str path_in:
    :param str path_out:
    :param float noise_level:
    :param str img_pattern:
    :param int nb_jobs:
    """
    logging.info('starting adding noise %f', noise_level)
    assert os.path.exists(path_in), 'missing: %s' % path_in
    assert os.path.exists(path_out), 'missing: %s' % path_out

    path_imgs = sorted(glob.glob(os.path.join(path_in, img_pattern)))
    name_imgs = [os.path.basename(p) for p in path_imgs]
    logging.info('found images: %i', len(name_imgs))

    dir_in = os.path.basename(path_in)
    dir_out = dir_in + DIR_POSIX % noise_level
    path_out = os.path.join(path_out, dir_out)
    logging.debug('creating dir: %s', path_out)
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    else:
        logging.warning('the output dir already exists')

    _wrapper_noise = partial(add_noise_image, path_in=path_in,
                             path_out=path_out, noise_level=noise_level)
    list(utils.wrap_execute_sequence(_wrapper_noise, name_imgs, nb_jobs))

    logging.info('DONE')


def main(base_path, datasets, noise_level=NOISE_RANGE):
    assert os.path.exists(base_path), 'missing: %s' % base_path

    for dataset in datasets:
        path_out = os.path.join(base_path, dataset)
        assert os.path.exists(path_out), 'missing: %s' % path_out
        path_in = os.path.join(path_out, BASE_IMAGE_SET)
        assert os.path.exists(path_in), 'missing: %s' % path_in

        for lvl in noise_level:
            dataset_add_noise(path_in, path_out, lvl)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    main(args['path'], args['datasets'], args['sigma'])
