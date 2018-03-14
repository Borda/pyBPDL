"""
Simple script for adding Gaussian noise to already generated images

>> python run_dataset_add_noise.py -p data -d syntheticDataset_vX

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

import tqdm

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_dataset

NB_THREADS = int(mproc.cpu_count() * 0.7)
IMAGE_PATTERN = '*.png'
DIR_POSIX = '_gauss-%.3f'
NOISE_RANGE = [0.2, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001]
LIST_DATASETS = [tl_dataset.DIR_MANE_SYNTH_DATASET]
BASE_IMAGE_SET = 'datasetProb_raw'


def args_parser():
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='path to set of experiments')
    parser.add_argument('-d', '--datasets', type=str, required=True, nargs='+',
                        help='result file name', default=LIST_DATASETS)
    parser.add_argument('-s', '--sigmas', type=str, required=False, nargs='+',
                        help='Gaussian sigma of additive noise',
                        default=NOISE_RANGE)
    args = vars(parser.parse_args())
    args['path'] = os.path.abspath(os.path.expanduser(args['path']))
    return args


def add_niose_image(img_name, path_in, path_out, noise_level):
    """

    :param str img_name:
    :param str path_in:
    :param str path_out:
    :param float noise_level:
    """
    path_img = os.path.join(path_in, img_name)
    logging.debug('loading image: %s', path_img)
    name, img = tl_dataset.load_image(path_img)
    img_noise = tl_dataset.add_image_prob_gauss_noise(img, noise_level)
    tl_dataset.export_image(path_out, img_noise, name)


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

    wrapper_image_niose = partial(add_niose_image, path_in=path_in,
                                  path_out=path_out, noise_level=noise_level)

    logging.debug('running in %i threads...', nb_jobs)
    mproc_pool = mproc.Pool(nb_jobs)
    tqdm_bar = tqdm.tqdm(total=len(name_imgs))
    for x in mproc_pool.imap_unordered(wrapper_image_niose, name_imgs):
        tqdm_bar.update()

    logging.info('DONE')


def main(base_path, datasets=['apdDataset_vX'], noise_lvl=NOISE_RANGE):
    assert os.path.exists(base_path), 'missing: %s' % base_path

    for dataset in datasets:
        path_out = os.path.join(base_path, dataset)
        assert os.path.exists(path_out), 'missing: %s' % path_out
        path_in = os.path.join(path_out, BASE_IMAGE_SET)
        assert os.path.exists(path_in), 'missing: %s' % path_in

        for lvl in noise_lvl:
            dataset_add_noise(path_in, path_out, lvl)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    main(args['path'], args['datasets'], args['sigmas'])
