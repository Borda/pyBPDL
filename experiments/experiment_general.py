"""
The base class for all Atomic Pattern Dictionary methods
such as the stat of the art and our newly developed

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import argparse
import logging
import shutil
import random
import time
import json
import copy
import re
import traceback
import types
import multiprocessing as mproc

import matplotlib
if os.environ.get('DISPLAY', '') == '' \
        and matplotlib.rcParams['backend'] != 'agg':
    # logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pylab as plt

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.data_utils as tl_data
import bpdl.utilities as tl_utils
import bpdl.pattern_atlas as ptn_dict
import bpdl.pattern_weights as ptn_weight

FORMAT_DT = '%Y%m%d-%H%M%S'
CONFIG_JSON = 'config.json'
RESULTS_TXT = 'resultStat.txt'
RESULTS_CSV = 'results.csv'
FILE_LOGS = 'logging.txt'
NAME_ATLAS = 'atlas{}'
NAME_ENCODING = 'encoding{}.csv'
EVAL_COLUMNS = ['atlas ARS', 'reconst. diff GT', 'reconst. diff Input', 'time']
EVAL_COLUMNS_START = ['atlas', 'reconst', 'time']


# fixing ImportError: No module named 'copy_reg' for Python3
if sys.version_info.major == 2:
    import copy_reg

    def _reduce_method(m):
        """ REQURED FOR MPROC POOL
        ISSUE: cPickle.PicklingError:
          Can't pickle <type 'instancemethod'>:
          attribute lookup __builtin__.instancemethod failed
        SEE: http://stackoverflow.com/questions/25156768
        """
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    copy_reg.pickle(types.MethodType, _reduce_method)

NB_THREADS = int(mproc.cpu_count() * .8)
PATH_DATA_SYNTH = tl_data.update_path('data_images')
PATH_DATA_REAL_DISC = os.path.join(tl_data.update_path('data_images'), 'imaginal_discs')
PATH_DATA_REAL_OVARY = os.path.join(tl_data.update_path('data_images'), 'ovary_stage-2')
PATH_RESULTS = tl_data.update_path('results')
DEFAULT_PARAMS = {
    'type': None,
    'computer': repr(os.uname()),
    'nb_samples': None,
    'tol': 1e-5,
    'init_tp': 'random-mosaic-2',  # random, greedy, , GT-deform
    'max_iter': 150,  # 250, 25
    'gc_regul': 1e-9,
    'nb_labels': tl_data.NB_BIN_PATTERNS + 1,
    'runs': range(NB_THREADS),
    'gc_reinit': True,
    'ptn_split': False,
    'ptn_compact': False,
    'connect_diag': True,
    'overlap_major': True,
    'deform_coef': None,
    'path_in': '',
    'path_out': '',
    'dataset': [''],
}

SYNTH_DATASET_NAME = tl_data.DIR_MANE_SYNTH_DATASET
SYNTH_PATH_APD = os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME)

SYNTH_SUBSETS = ['raw', 'noise', 'deform', 'defNoise']
SYNTH_SUB_DATASETS_BINARY = ['datasetBinary_' + n for n in SYNTH_SUBSETS]
SYNTH_SUB_DATASETS_FUZZY = ['datasetFuzzy_' + n for n in SYNTH_SUBSETS]
SYNTH_SUB_DATASETS_FUZZY_NOISE = ['datasetFuzzy_raw_gauss-%.3f' % d
                                  for d in tl_data.GAUSS_NOISE]

SYNTH_PARAMS = DEFAULT_PARAMS.copy()
SYNTH_PARAMS.update({
    'type': 'synth',
    'path_in': SYNTH_PATH_APD,
    'dataset': SYNTH_SUB_DATASETS_FUZZY,
    'runs': range(3),
    'path_out': PATH_RESULTS,
})
# SYNTH_RESULTS_NAME = 'experiments_APD'

REAL_PARAMS = DEFAULT_PARAMS.copy()
REAL_PARAMS.update({
    'type': 'real',
    'path_in': PATH_DATA_REAL_DISC,
    'dataset': ['gene_small'],
    'max_iter': 50,
    'runs': 0,
    'path_out': PATH_RESULTS
})
# PATH_OUTPUT = os.path.join('..','..','results')


def create_args_parser(dict_params, methods):
    """ create simple arg parser with default values (input, output, dataset)

    :param {str: ...} dict_params:
    :return obj: object argparse<...>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--path_in', type=str, required=True,
                        help='path to the folder with input image dataset',
                        default=dict_params.get('path_in', ''))
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to the output with experiment results',
                        default=dict_params.get('path_out', ''))
    parser.add_argument('-t', '--type', type=str, required=False,
                        help='switch between real and synth. images',
                        default='real', choices=['real', 'synth'])
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='specific name for this experiment', default=None)
    parser.add_argument('-ds', '--dataset', type=str, required=False,
                        nargs='+', help='names of used datasets', default=None)
    parser.add_argument('-ptn', '--nb_patterns', type=int, required=False,
                        nargs='+', help='numbers of estimated patterns',
                        default = None)
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number of processes running in parallel',
                        default=NB_THREADS)
    parser.add_argument('--method', type=str, required=False, nargs='+',
                        default=None, help='possible APD methods',
                        choices=methods)
    parser.add_argument('-imgs', '--list_images', type=str, required=False,
                        help='CSV file with list of images, supress `path_in`',
                        default=None)
    parser.add_argument('-cfg', '--path_config', type=str, required=False,
                        help='path to JSON configuration file',
                        default=None)
    parser.add_argument('--debug', required=False, action='store_true',
                        help='run in debug mode', default=False)
    parser.add_argument('--unique', required=False, action='store_true',
                        help='use time stamp for each experiment', default=False)
    return parser


def parse_arg_params(parser):
    """ parse basic args and return as dictionary

    :param obj parser: argparse
    :return {str: ...}:
    """
    args = vars(parser.parse_args())
    # remove not filled parameters
    args = {k: args[k] for k in args if args[k] is not None}
    for n in (k for k in args if k.startswith('path_') and args[k] is not None):
        args[n] = tl_data.update_path(args[n])
        assert os.path.exists(args[n]), '%s' % args[n]
    for flag in ['nb_patterns', 'method']:
        if flag in args and not isinstance(args[flag], list):
            args[flag] = [args[flag]]

    if 'nb_patterns' in args:
        if is_list_like(args['nb_patterns']):
            args.update({'nb_labels': [l + 1 for l in args['nb_patterns']]})
        else:
            args['nb_labels'] = args['nb_patterns'] + 1
        del args['nb_patterns']

    return args


def parse_params(default_params, methods):
    """ parse arguments from command line

    :param {str: ...} default_params:
    :return {str: ...}:
    """
    parser = create_args_parser(default_params, methods)
    params = copy_dict(default_params)
    arg_params = parse_arg_params(parser)

    params.update(arg_params)

    # if json config exists update configuration
    if arg_params.get('path_config', None) is not None \
            and os.path.isfile(arg_params['path_config']):
        logging.info('loading config: %s', arg_params['path_config'])
        d_json = json.load(open(arg_params['path_config']))
        logging.debug(string_dict(d_json, desc='LOADED CONFIG:'))

        # skip al keys with path or passed from arg params
        d_update = {k: d_json[k] for k in d_json
                    if not k.startswith('path_') or not k in arg_params}
        logging.debug(string_dict(d_update, desc='TO BE UPDATED:'))
        params.update(d_update)

    return params


def load_list_img_names(path_csv, path_in=''):
    """ loading images from a given list and if necessary add default data path

    :param str path_csv:
    :param str path_in:
    :return [str]:
    """
    assert os.path.exists(path_csv), '%s' % path_csv
    df = pd.read_csv(path_csv, index_col=False, header=None)
    assert len(df.columns) == 1, 'assume just single column'
    list_names = df.as_matrix()[:, 0].tolist()
    # if the input path was set and the list are just names, no complete paths
    if os.path.exists(path_in) and not all(os.path.exists(p) for p in list_names):
        # to each image name add the input path
        list_names = [os.path.join(path_in, p) for p in list_names]
    return list_names


def create_experiment_folder(params, dir_name, stamp_unique=True, skip_load=True):
    """ create the experiment folder and iterate while there is no available

    :param {str: ...} params:
    :param str dir_name:
    :param bool stamp_unique:
    :param bool skip_load:
    :return {str: ...}:

    >>> p = {'path_out': '.'}
    >>> p = create_experiment_folder(p, 'my_test', False, skip_load=True)
    >>> 'computer' in p
    True
    >>> p['path_exp']
    './my_test_EXAMPLE'
    >>> shutil.rmtree(p['path_exp'], ignore_errors=True)
    """
    date = time.gmtime()
    name = params.get('name', 'EXAMPLE')
    if isinstance(name, str) and len(name) > 0:
        dir_name = '{}_{}'.format(dir_name, name)
    # if self.params.get('date_time') is None:
    #     self.params.set('date_time', time.gmtime())
    if stamp_unique:
        dir_name += '_' + time.strftime(FORMAT_DT, date)
    path_expt = os.path.join(params.get('path_out'), dir_name)
    while stamp_unique and os.path.exists(path_expt):
        logging.warning('particular out folder already exists')
        path_expt += ':' + str(random.randint(0, 9))
    logging.info('creating experiment folder "{}"'.format(path_expt))
    if not os.path.exists(path_expt):
        os.mkdir(path_expt)
    path_config = os.path.join(path_expt, CONFIG_JSON)
    if os.path.exists(path_config) and not skip_load:
        logging.debug('loading saved params from file "%s"', CONFIG_JSON)
        params = json.load(open(path_config, 'r'))
    params.update({'computer': repr(os.uname()),
                   'path_exp': path_expt})
    logging.debug('saving params to file "%s"', CONFIG_JSON)
    with open(path_config, 'w') as f:
        json.dump(params, f)
    return params


def set_experiment_logger(path_out, file_name=FILE_LOGS, reset=True):
    """ set the logger to file

    :param str path_out:
    :param str file_name:
    :param bool reset:
    """
    log = logging.getLogger()
    if reset:
        log.handlers = [h for h in log.handlers
                        if not isinstance(h, logging.FileHandler)]
    path_logger = os.path.join(path_out, file_name)
    logging.info('setting logger to "%s"', path_logger)
    fh = logging.FileHandler(path_logger)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)


def string_dict(d, desc='DICTIONARY:', offset=30):
    """ transform dictionary to a formatted string

    :param {} d: dictionary with parameters
    :param int offset: length between name and value
    :param str desc: dictionary title
    :return str:

    >>> string_dict({'abc': 123})  #doctest: +NORMALIZE_WHITESPACE
    \'DICTIONARY:\\n"abc": 123\'
    """
    s = desc + '\n'
    tmp_name = '{:' + str(offset) + 's} {}'
    rows = [tmp_name.format('"{}":'.format(n), repr(d[n]))
            for n in sorted(d)]
    s += '\n'.join(rows)
    return str(s)


def copy_dict(d):
    """ alternative of deep copy without pickle on in first level
    Nose testing - TypeError: can't pickle dict_keys objects

    :param {} d: dictionary
    :return {}:
    >>> d1 = {'a': [0, 1]}
    >>> d2 = copy_dict(d1)
    >>> d2['a'].append(3)
    >>> d1
    {'a': [0, 1]}
    >>> d2
    {'a': [0, 1, 3]}
    """
    d_new = copy.deepcopy(d)
    return d_new


def generate_conf_suffix(d_params):
    """ generating suffix strung according given params

    :param {} d_params: dictionary
    :return str:

    >>> params = {'my_Param': 15}
    >>> generate_conf_suffix(params)
    '_my-Param=15'
    >>> params.update({'new_Param': 'abc'})
    >>> generate_conf_suffix(params)
    '_my-Param=15_new-Param=abc'
    """
    suffix = '_'
    suffix += '_'.join('{}={}'.format(k.replace('_', '-'), d_params[k])
                      for k in sorted(d_params) if k != 'param_idx')
    return suffix

# =============================================================================
# =============================================================================


class Experiment(object):
    """
    main_train class for APD experiments State-of-the-Art and BPDL

    EXAMPLE:
    >>> params = {'dataset': tl_data.DEFAULT_NAME_DATASET,
    ...           'path_in': os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME),
    ...           'path_out': PATH_RESULTS}
    >>> expt = Experiment(params, time_stamp=False)
    >>> expt.run(gt=True)
    >>> shutil.rmtree(expt.params['path_exp'], ignore_errors=True)
    """

    REQUIRED_PARAMS = ['dataset', 'path_in', 'path_out']

    def __init__(self, dict_params, time_stamp=True):
        """ initialise class and set the experiment parameters

        :param {str: ...} dict_params:
        :param bool time_stamp: mark if you want an unique folder per experiment
        """
        assert all([n in dict_params for n in self.REQUIRED_PARAMS]), \
            'missing some required parameters'
        dict_params = simplify_params(dict_params)

        if not 'name' in dict_params:
            dataset_name = dict_params['dataset']
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[0]
            last_dir = os.path.basename(dict_params['path_in'])
            dict_params['name'] = '{}_{}_{}'.format(dict_params.get('type', ''),
                                                    last_dir,
                                                    dataset_name)

        dict_params['method'] = repr(self.__class__.__name__)
        if not os.path.exists(dict_params['path_out']):
            os.mkdir(dict_params['path_out'])

        self.params = dict_params
        self.params['class'] = self.__class__.__name__
        self.__check_exist_paths()
        self.__create_folder(stamp_unique=time_stamp)
        set_experiment_logger(self.params['path_exp'])
        self.df_results = pd.DataFrame()
        self._path_stat = os.path.join(self.params.get('path_exp'), RESULTS_TXT)
        self._list_img_paths = None
        # self.params.export_as(self._path_stat)
        with open(self._path_stat, 'w') as fp:
            fp.write(string_dict(self.params, desc='PARAMETERS:'))
        logging.info(string_dict(self.params, desc='PARAMETERS:'))

    def __check_exist_paths(self):
        """ Check all required paths in parameters whether they exist """
        for p in (self.params[n] for n in self.params
                  if 'dir' in n.lower() or 'path' in n.lower()):
            if not os.path.exists(p):
                raise Exception('given folder "%s" does not exist!' % p)
        for p in (self.params[n] for n in self.params if 'file' in n.lower()):
            if not os.path.exists(p):
                raise Exception('given folder "%s" does not exist!' % p)

    def __create_folder(self, stamp_unique=True):
        """ Create the experiment folder and iterate while there is no available

        :param bool stamp_unique: mark if you want an unique folder per experiment
        """
        # create results folder for experiments
        if not os.path.exists(self.params.get('path_out')):
            logging.error('no results folder "%s"' % self.params.get('path_out'))
            self.params['path_exp'] = os.path.join(self.params.get('path_out'), '')
            exit(1)
        self.params = create_experiment_folder(self.params,
                                               self.__class__.__name__,
                                               stamp_unique)

    def _load_data_ground_truth(self):
        """ loading all GT suh as atlas and reconstructed images from GT encoding

        :param {str: ...} params: parameter settings
        """
        path_atlas = os.path.join(self.params.get('path_in'),
                                  tl_data.DIR_NAME_DICTIONARY)
        self._gt_atlas = tl_data.dataset_compose_atlas(path_atlas)
        if self.params.get('list_images') is not None:
            img_names = [os.path.splitext(os.path.basename(p))[0]
                         for p in self._list_img_paths]
            self._gt_encoding = tl_data.dataset_load_weights(self.path_data,
                                                             img_names=img_names)
        else:
            self._gt_encoding = tl_data.dataset_load_weights(self.params.get('path_in'))
        self._gt_images = ptn_dict.reconstruct_samples(self._gt_atlas,
                                                       self._gt_encoding)
        # self._images = [im.astype(np.uint8, copy=False) for im in self._images]

    def _load_images(self):
        """ load image data """
        self._images, self._image_names = tl_data.dataset_load_images(
            self._list_img_paths, nb_jobs=1)

    def __load_data(self, gt=True):
        """ load all required data for APD and also ground-truth if required

        :param bool gt: search for the Ground Truth using standard names
        """
        logging.info('loading required data')
        self.path_data = os.path.join(self.params.get('path_in'),
                                      self.params.get('dataset'))
        # load according a csv list
        if self.params.get('list_images') is not None:
            # copy the list of selected images
            path_csv = os.path.expanduser(self.params.get('list_images'))
            if not os.path.exists(path_csv):
                path_csv = os.path.abspath(os.path.join(self.path_data, path_csv))
                shutil.copy(path_csv, os.path.join(self.params['path_exp'],
                                                   os.path.basename(path_csv)))
            self._list_img_paths = load_list_img_names(path_csv)
        else:
            self._list_img_paths = tl_data.find_images(self.path_data)

        assert len(self._list_img_paths) > 0, 'no images found'
        self._load_images()

        # loading  if it is set
        if gt:
            self._load_data_ground_truth()
            assert len(self._images) == len(self._gt_images), \
                'nb of input (%i) and reconst. (%i) images do not match' \
                % (len(self._images), len(self._gt_images))
        logging.debug('loaded %i images', len(self._images))

    def run(self, gt=False, iter_params=None):
        """ the main procedure that load, perform and evaluate experiment

        :param bool gt: search for the Ground Truth using standard names
        :param [] iter_params: list of possible configuration
        """
        logging.info('perform the complete experiment')

        # in case it single value make it iterable
        if is_list_like(iter_params):
            self.iter_params = copy_dict(iter_params)
        elif isinstance(iter_params, dict):
            logging.info(string_dict(iter_params, desc='ITERATE PARAMETERS:'))
            self.iter_params = expand_params(iter_params)
        else:
            self.iter_params = None

        self.__load_data(gt)
        self._perform()
        self.__summarise()
        logging.info('FINISHED >]')
        logging.getLogger().handlers = []

    def _perform(self):
        """ perform experiment as sequence of iterated configurations """
        if is_list_like(self.iter_params):
            logging.info('iterate over %i configurations', len(self.iter_params))
            self.__perform_sequence()
        else:
            logging.debug('perform single configuration')
            detail = self.__perform_once({})
            self.df_results = pd.DataFrame([detail])

    def __perform_sequence(self):
        """ Iteratively change a single experiment parameter with the same data
        """
        logging.info('perform_sequence in single thread')
        tqdm_bar = tqdm.tqdm(total=len(self.iter_params))
        for d_params in self.iter_params:
            # self.params.update(d_params)
            tqdm_bar.set_description(d_params.get('param_idx', ''))
            logging.debug(' -> set iterable %s', repr(d_params))

            detail = self.__perform_once(d_params)

            self.df_results = self.df_results.append(detail, ignore_index=True)
            # just partial export
            logging.debug('partial results: %s', repr(detail))
            tqdm_bar.update()

    def _estimate_atlas_weights(self, images, params):
        """ This is the method to be be over written by individual methods

        :param [ndarray] images:
        :param {} params:
        :return (ndarray, ndarray, {}):
        """
        del params
        atlas = np.zeros_like(self._images[0])
        weights = np.zeros((len(self._images), 0))
        return atlas, weights, None

    def __perform_once(self, d_params):
        """ perform single experiment

        :param {str: ...} d_params: used specific configuration
        :return {str: ...}: output statistic
        """
        detail = copy_dict(self.params)
        detail.update(copy_dict(d_params))
        detail['name_suffix'] = generate_conf_suffix(d_params)

        # in case you chose only a subset of images
        nb_samples = detail.get('nb_samples', None)
        if isinstance(nb_samples, float):
            nb_samples = int(len(self._images) * nb_samples)
        images = self._images[:nb_samples]

        try:
            t = time.time()
            atlas, weights, extras = self._estimate_atlas_weights(images, detail)
            detail['time'] = time.time() - t
        except Exception:  # todo, optionaly remove this try/catch
            logging.error('FAIL estimate atlas for %s with %s',
                          str(self.__class__), repr(detail))
            logging.error(traceback.format_exc())
            atlas = np.zeros_like(self._images[0])
            weights = np.zeros((len(self._images), 0))
            extras = None
            detail['time'] = -0.

        logging.debug('estimated atlas of size %s and labels %s',
                      repr(atlas.shape), repr(np.unique(atlas).tolist()))

        weights_all = [ptn_weight.weights_image_atlas_overlap_major(img, atlas)
                   for img in self._images]
        weights_all = np.array(weights_all)

        logging.debug('estimated weights of size %s and summing %s',
                      repr(weights_all.shape), repr(np.sum(weights_all, axis=0)))

        self._export_atlas(atlas, suffix=detail['name_suffix'])
        self._export_coding(weights_all, suffix=detail['name_suffix'])
        self._export_extras(extras, suffix=detail['name_suffix'])

        detail.update(self.__evaluate(atlas, weights_all))
        detail.update(self._evaluate_extras(atlas, weights, extras))


        return detail

    def _export_atlas(self, atlas, suffix=''):
        """ export estimated atlas

        :param ndarray atlas: np.array<height, width>
        :param str suffix:
        """
        n_img = NAME_ATLAS.format(suffix)
        tl_data.export_image(self.params.get('path_exp'), atlas, n_img,
                             stretch_range=False)
        path_atlas_rgb = os.path.join(self.params.get('path_exp'),
                                      n_img + '_rgb.png')
        logging.debug('exporting RGB atlas: %s', path_atlas_rgb)
        plt.imsave(path_atlas_rgb, atlas, cmap=plt.cm.jet)

    def _export_coding(self, weights, suffix=''):
        """ export estimated atlas

        :param ndarray weights:
        :param str suffix:
        """
        if not hasattr(self, '_image_names'):
            self._image_names = [str(i) for i in range(weights.shape[0])]
        df = tl_data.format_table_weights(self._image_names, weights)

        path_csv = os.path.join(self.params.get('path_exp'),
                                NAME_ENCODING.format(suffix))
        logging.debug('exporting encoding: %s', path_csv)
        df.to_csv(path_csv)

    def _export_extras(self, extras, suffix=''):
        """ export some extra parameters

        :param {} extras: dictionary with extra variables
        """
        pass

    def __evaluate_atlas(self, atlas):
        """ Evaluate atlas

        :param ndarray atlas:
        :return float:
        """
        if hasattr(self, '_gt_atlas'):
            logging.debug('... compute Atlas static')
            assert self._gt_atlas.shape == atlas.shape, 'atlases do not match'
            ars = metrics.adjusted_rand_score(self._gt_atlas.ravel(), atlas.ravel())
        else:
            ars = None
        return ars

    def _evaluate_reconstruct(self, images_rct, im_type='GT'):
        """ Evaluate the reconstructed images to GT if exists
         or just input images as difference sum

        :param [ndarray] images_rct: reconstructed images
        :return (str, float):
        """
        assert images_rct is not None, 'missing any images to compare with'
        # error estimation from original reconstruction
        if im_type == 'GT' and  hasattr(self, '_gt_images'):
            logging.debug('compute reconstruction - GT images')
            images_gt = self._gt_images[:len(images_rct)]
            diff = np.asarray(images_gt) - np.asarray(images_rct)
            nb_pixels = float(np.product(diff.shape))
            diff_norm = np.sum(abs(diff)) / nb_pixels
            return 'GT', diff_norm
        elif hasattr(self, '_images'):
            logging.debug('compute reconstruction - Input images')
            images = self._images[:len(images_rct)]
            diff = np.asarray(images) - np.asarray(images_rct)
            nb_pixels = float(np.product(diff.shape))
            diff_norm = np.sum(abs(diff)) / nb_pixels
            return 'Input', diff_norm
        return 'FAIL', np.nan

    def __evaluate(self, atlas, weights):
        """ Compute the statistic for GT and estimated atlas and reconst. images

        :param ndarray atlas: np.array<height, width>
        :param [ndarray] weights: np.array<nb_samples, nb_patterns>
        :return {str: ...}:
        """
        images_rct = ptn_dict.reconstruct_samples(atlas, weights)
        tag, diff = self._evaluate_reconstruct(images_rct)
        stat = {
            'atlas ARS': self.__evaluate_atlas(atlas),
            'reconst. diff %s' % tag: diff
        }
        return stat

    def _evaluate_extras(self, atlas, weights, extras):
        """ some extra evaluation

        :param ndarray atlas: np.array<height, width>
        :param [ndarray] weights: np.array<nb_samples, nb_patterns>
        :param {} extras:
        :return {}:
        """
        return {}

    def __summarise(self):
        """ summarise and export experiment results """
        logging.info('summarise the experiment')
        path_csv = os.path.join(self.params.get('path_exp'), RESULTS_CSV)
        logging.debug('save results: "%s"', path_csv)
        self.df_results.to_csv(path_csv)

        if hasattr(self, 'df_results') and not self.df_results.empty:
            df_stat = self.df_results.describe()
            # df_stat = df_stat[[c for c in EVAL_COLUMNS if c in df_stat.columns]]
            cols = [c for c in df_stat.columns
                    if any([c.startswith(cc) for cc in EVAL_COLUMNS_START])]
            df_stat = df_stat[cols]
            with open(self._path_stat, 'a') as fp:
                fp.write('\n' * 3 + 'RESULTS: \n' + '=' * 9)
                fp.write('\n{}'.format(df_stat))
            logging.debug('statistic: \n%s', repr(df_stat))

# =============================================================================
# =============================================================================


class ExperimentParallel(Experiment):
    """
    run the experiment in multiple threads

    EXAMPLE:
    >>> params = {'dataset': tl_data.DEFAULT_NAME_DATASET,
    ...           'path_in': os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME),
    ...           'path_out': PATH_RESULTS}
    >>> expt = ExperimentParallel(params, time_stamp=False)
    >>> expt.run(gt=True)
    >>> shutil.rmtree(expt.params['path_exp'], ignore_errors=True)
    """

    def __init__(self, dict_params, time_stamp=True):
        """ initialise parameters and nb jobs in parallel

        :param {str: ...} dict_params:
        :param bool time_stamp:
        """
        super(ExperimentParallel, self).__init__(dict_params, time_stamp)
        self.nb_jobs = dict_params.get('nb_jobs', NB_THREADS)

    def _load_images(self):
        """ load image data """
        self._images, self._image_names = tl_data.dataset_load_images(
            self._list_img_paths, nb_jobs=self.nb_jobs)

    def __perform_sequence(self):
        """ perform sequence in multiprocessing pool """
        logging.debug('perform_sequence in %i threads for %i values',
                      self.nb_jobs, len(self.iter_params))
        # ISSUE with passing large date to processes so the images are saved
        # and loaded in particular process again
        # p_imgs = os.path.join(self.params.get('path_exp'), 'input_images.npz')
        # np.savez(open(p_imgs, 'w'), imgs=self._images)

        tqdm_bar = tqdm.tqdm(total=len(self.iter_params))
        mproc_pool = mproc.Pool(self.nb_jobs)
        for detail in mproc_pool.map(self.__perform_once,
                                     self.iter_params):
            self.df_results = self.df_results.append(detail, ignore_index=True)
            logging.debug('partial results: %s', repr(detail))
            # just partial export
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()

        # remove temporary image file
        # os.remove(p_imgs)

# =============================================================================
# =============================================================================


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
        is_iter = [isinstance(var, tp) for tp
                   in (list, tuple, range, np.ndarray, types.GeneratorType)]
    except Exception:  # for python 2
        is_iter = [isinstance(var, tp) for tp
                   in (list, tuple, np.ndarray, types.GeneratorType)]
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


def extend_list_params(list_params, name_param, list_options):
    """ extend the parameter list by all sub-datasets

    :param [{str: ...}] list_params:
    :param str name_param:
    :param [] list_options:
    :return [{str: ...}]:

    >>> params = extend_list_params([{'a': 1}], 'a', [3, 4])
    >>> pd.DataFrame(params)  # doctest: +NORMALIZE_WHITESPACE
       a param_idx
    0  3     a-2#1
    1  4     a-2#2
    >>> params = extend_list_params([{'a': 1}], 'b', 5)
    >>> pd.DataFrame(params)  # doctest: +NORMALIZE_WHITESPACE
       a  b param_idx
    0  1  5     b-1#1
    """
    if not is_list_like(list_options):
        list_options = [list_options]
    list_params_new = []
    for p in list_params:
        p['param_idx'] = p.get('param_idx', '')
        for i, v in enumerate(list_options):
            p_new = p.copy()
            p_new.update({name_param: v})
            if len(p_new['param_idx']) > 0:
                p_new['param_idx'] += '_'
            p_new['param_idx'] += \
                '%s-%i#%i' % (name_param, len(list_options), i + 1)
            list_params_new.append(p_new)
    return list_params_new


def simplify_params(dict_params):
    """ extract simple configuration dictionary

    :return:

    >>> params = simplify_params({'t': 'abc', 'n': [1, 2]})
    >>> pd.Series(params).sort_index()  #doctest: +NORMALIZE_WHITESPACE
    n      1
    t    abc
    dtype: object
    """
    d_params = {}
    for k in dict_params:
        d_params[k] = dict_params[k][0] if is_list_like(dict_params[k]) \
            else dict_params[k]
    return d_params


def expand_params(dict_params, simple_config=None):
    """ extend parameters to a list

    :param {} simple_config:
    :param {} dict_params:
    :return:

    >>> params = expand_params({'t': ['abc'], 'n': [1, 2], 's': ('x', 'y')})
    >>> pd.DataFrame(params)  # doctest: +NORMALIZE_WHITESPACE
       n     param_idx  s    t
    0  1  n-2#1_s-2#1  x  abc
    1  1  n-2#1_s-2#2  y  abc
    2  2  n-2#2_s-2#1  x  abc
    3  2  n-2#2_s-2#2  y  abc
    >>> params = expand_params({'s': ('x', 'y')}, {'old': 123.})
    >>> pd.DataFrame(params)  # doctest: +NORMALIZE_WHITESPACE
         old param_idx  s
    0  123.0     s-2#1  x
    1  123.0     s-2#2  y
    """
    if simple_config is None:
        simple_config = {}
    simple_config.update(dict_params)
    simple_config = simplify_params(simple_config)

    list_configs = [simple_config]
    for k in sorted(dict_params):
        if is_list_like(dict_params[k]) and len(dict_params[k]) > 1:
            list_configs = extend_list_params(list_configs, k, dict_params[k])

    return list_configs


def parse_config_txt(path_config):
    """ open file with saved configuration and restore it

    :param str path_config:
    :return {str: str}:

    >>> p_txt = 'sample_config.txt'
    >>> with open(p_txt, 'w') as fp:
    ...     _= fp.write('"my":   ')  # it may return nb characters
    >>> parse_config_txt(p_txt)
    {}
    >>> with open(p_txt, 'w') as fp:
    ...     _= fp.write('"my":   123')  # it may return nb characters
    >>> parse_config_txt(p_txt)
    {'my': 123}
    >>> os.remove(p_txt)
    """
    if not os.path.exists(path_config):
        logging.error('config file "%s" does not exist!', path_config)
        return {}
    with open(path_config, 'r') as fp:
        text = ''.join(fp.readlines())
    rec = re.compile('"(\S+)":\s+(.*)')
    dict_config = {n: tl_utils.convert_numerical(v)
                   for n, v in rec.findall(text) if len(v) > 0}
    return dict_config
