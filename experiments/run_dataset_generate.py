"""
The main script for generating synthetic datasets

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import argparse
import inspect
import multiprocessing as mproc
from functools import partial

from imsegm.utilities.data_io import update_path
from imsegm.utilities.experiments import save_config_yaml

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bpdl.data_utils import (
    DIR_MANE_SYNTH_DATASET, CSV_NAME_WEIGHTS, dictionary_generate_atlas,
    dataset_binary_combine_patterns, dataset_apply_image_function, image_deform_elastic,
    add_image_binary_noise, image_transform_binary2fuzzy, add_image_fuzzy_pepper_noise)

NB_WORKERS = int(mproc.cpu_count() * 0.7)
DEFAULT_PATH_DATA = update_path('data_images')
DEFAULT_PATH_APD = os.path.join(DEFAULT_PATH_DATA, DIR_MANE_SYNTH_DATASET)
NAME_WEIGHTS = CSV_NAME_WEIGHTS
NAME_CONFIG = 'config.yml'
IMAGE_SIZE = {
    '2D': (64, 64),
    '3D': (16, 128, 128),
}
DATASET_TYPE = '2D'
NB_SAMPLES = 50
NB_ATM_PATTERNS = 4
NOISE_BINARY = 0.03
NOISE_FUZZY = 0.15


def aparse_params():
    """
    SEE: https://docs.python.org/3/library/argparse.html
    :return obj:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_out', type=str, required=False,
                        default=DEFAULT_PATH_APD,
                        help='path to the output dataset')
    parser.add_argument('--nb_samples', type=int, required=False,
                        default=NB_SAMPLES,
                        help='number of samples to be generated')
    parser.add_argument('--nb_patterns', type=int, required=False,
                        default=NB_ATM_PATTERNS,
                        help='number of atom. patterns in created dictionary')
    parser.add_argument('--image_size', type=int, required=False, nargs='+',
                        default=IMAGE_SIZE[DATASET_TYPE],
                        help='dimensions of generated images in axis Z, X, Y')
    parser.add_argument('--nb_workers', type=int, required=False,
                        default=NB_WORKERS,
                        help='number of processes in parallel')
    args = parser.parse_args()
    assert len(args.image_size) == 2 or len(args.image_size) == 3, \
        'unsupported image dimension of %r' % args.image_size
    args.path_out = os.path.abspath(os.path.expanduser(args.path_out))
    return args


def view_func_params(frame=inspect.currentframe(), path_out=''):
    """ view function parameters

    :param frame:
    :param str path_out:
    :return dict:

    >>> view_func_params()  # doctest: +ELLIPSIS
    {...}
    """
    _, _, _, values = inspect.getargvalues(frame)
    logging.info('PARAMETERS: \n%s',
                 '\n'.join('"{}": \t {}'.format(k, values[k]) for k in values))
    if os.path.exists(path_out):
        save_config_yaml(os.path.join(path_out, NAME_CONFIG), values)
    return values


def generate_all(path_out=DEFAULT_PATH_APD,
                 atlas_size=IMAGE_SIZE[DATASET_TYPE],
                 nb_patterns=NB_ATM_PATTERNS,
                 nb_samples=NB_SAMPLES,
                 nb_workers=NB_WORKERS):
    """ generate complete dataset containing dictionary od patterns and also
    input binary / fuzzy images with geometrical deformation and random noise

    :param tuple(int,int) atlas_size:
    :param int nb_samples:
    :param int nb_patterns:
    :param str csv_name:
    :param str path_out: path to the results directory
    """
    assert nb_patterns > 0, 'number of patterns has to be larger then 0'
    assert os.path.exists(os.path.dirname(path_out)), \
        'missing: %s' % os.path.dirname(path_out)
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    view_func_params(inspect.currentframe(), path_out)
    _path_dir = lambda d: os.path.join(path_out, d)
    # im_dict = dictionary_generate_rnd_pattern()
    im_dict = dictionary_generate_atlas(path_out, im_size=atlas_size, nb_patterns=nb_patterns)
    assert len(im_dict) > 0, 'dictionary has contain at least one pattern'

    im_comb, df_weights = dataset_binary_combine_patterns(
        im_dict, _path_dir('datasetBinary_raw'), nb_samples)
    df_weights.to_csv(os.path.join(path_out, NAME_WEIGHTS))

    _warp_ds_apply = partial(dataset_apply_image_function, nb_workers=nb_workers)

    im_deform = _warp_ds_apply(im_comb, _path_dir('datasetBinary_deform'),
                               image_deform_elastic)
    _warp_ds_apply(im_comb, _path_dir('datasetBinary_noise'),
                   add_image_binary_noise, NOISE_BINARY)
    _warp_ds_apply(im_deform, _path_dir('datasetBinary_defNoise'),
                   add_image_binary_noise, NOISE_BINARY)

    im_comb_prob = _warp_ds_apply(im_comb, _path_dir('datasetFuzzy_raw'),
                                  image_transform_binary2fuzzy, 0.5)
    im_def_prob = _warp_ds_apply(im_deform, _path_dir('datasetFuzzy_deform'),
                                 add_image_fuzzy_pepper_noise, 0.5)
    _warp_ds_apply(im_comb_prob, _path_dir('datasetFuzzy_noise'),
                   add_image_fuzzy_pepper_noise, NOISE_FUZZY)
    _warp_ds_apply(im_def_prob, _path_dir('datasetFuzzy_defNoise'),
                   add_image_fuzzy_pepper_noise, NOISE_FUZZY)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = aparse_params()
    generate_all(path_out=params.path_out,
                 atlas_size=params.image_size,
                 nb_patterns=params.nb_patterns,
                 nb_samples=params.nb_samples,
                 nb_workers=params.nb_workers)

    # test_Ellipse()

    logging.info('DONE')
