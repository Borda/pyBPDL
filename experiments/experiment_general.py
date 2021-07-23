"""
The base class for all Atomic Pattern Dictionary methods
such as the stat of the art and our newly developed

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import platform
import sys
import time
import re
import logging
import argparse
import shutil
import copy
import types
import collections
import multiprocessing as mproc

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pylab as plt
from imsegm.utilities.data_io import update_path
from imsegm.utilities.experiments import (
    WrapExecuteSequence, string_dict, load_config_yaml, extend_list_params, Experiment as ExperimentBase
)

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bpdl.data_utils import (
    NB_BIN_PATTERNS, DIR_MANE_SYNTH_DATASET, GAUSS_NOISE, DIR_NAME_DICTIONARY, export_image, dataset_compose_atlas,
    dataset_load_weights, dataset_load_images, find_images, format_table_weights
)
from bpdl.utilities import convert_numerical, is_list_like
from bpdl.pattern_atlas import reconstruct_samples
from bpdl.pattern_weights import weights_image_atlas_overlap_major

#: default date-time format
FORMAT_DT = '%Y%m%d-%H%M%S'
#: default experiment configuration
CONFIG_YAML = 'config.yml'
#: default results statistics
RESULTS_TXT = 'resultStat.txt'
#: default complete results
RESULTS_CSV = 'results.csv'
#: default experiment logging file
FILE_LOGS = 'logging.txt'
#: default image name/template for atlas - collection of patterns
NAME_ATLAS = 'atlas{}'
#: default table name/template for activations per input image
NAME_ENCODING = 'encoding{}.csv'
EVAL_COLUMNS = ('atlas ARS', 'reconst. diff GT', 'reconst. diff Input', 'time')
EVAL_COLUMNS_START = ('atlas', 'reconst', 'time')

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

#: default number of avalaible threads to be used
NB_WORKERS = int(mproc.cpu_count() * .8)
#: default path to repository data/images
PATH_DATA_SYNTH = update_path('data_images')
#: default path with samples with drosophila imaginal discs
PATH_DATA_REAL_DISC = os.path.join(update_path('data_images'), 'imaginal_discs')
#: default path with samples with drosophila ovaries
PATH_DATA_REAL_OVARY = os.path.join(update_path('data_images'), 'ovary_stage-2')
#: default path to results
PATH_RESULTS = update_path('results')
#: default experiment configuration
DEFAULT_PARAMS = {
    'type': None,
    'computer': repr(platform.uname()),
    'nb_samples': None,
    'tol': 1e-5,
    'init_tp': 'random-mosaic-2',  # random, greedy, , GT-deform
    'max_iter': 150,  # 250, 25
    'gc_regul': 1e-9,
    'nb_labels': NB_BIN_PATTERNS + 1,
    'runs': range(NB_WORKERS),
    'gc_reinit': True,
    'ptn_compact': False,
    'connect_diag': True,
    'overlap_major': True,
    'deform_coef': None,
    'path_in': '',
    'path_out': '',
    'dataset': [''],
}

SYNTH_DATASET_NAME = DIR_MANE_SYNTH_DATASET
SYNTH_PATH_APD = os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME)

#: default types of synthetic datasets - different difficulty levels
SYNTH_SUBSETS = ['raw', 'noise', 'deform', 'defNoise']
#: create binary synthetic dataset names
SYNTH_SUB_DATASETS_BINARY = ['datasetBinary_' + n for n in SYNTH_SUBSETS]
#: create fuzzy synthetic dataset names
SYNTH_SUB_DATASETS_FUZZY = ['datasetFuzzy_' + n for n in SYNTH_SUBSETS]
SYNTH_SUB_DATASETS_FUZZY_NOISE = ['datasetFuzzy_raw_gauss-%.3f' % d for d in GAUSS_NOISE]

SYNTH_PARAMS = DEFAULT_PARAMS.copy()
#: adjust experiment parameters for synthetic datasets
SYNTH_PARAMS.update({
    'type': 'synth',
    'path_in': SYNTH_PATH_APD,
    'dataset': SYNTH_SUB_DATASETS_FUZZY,
    'runs': range(3),
    'path_out': PATH_RESULTS,
})
# SYNTH_RESULTS_NAME = 'experiments_APD'

REAL_PARAMS = DEFAULT_PARAMS.copy()
#: adjust experiment parameters for real-images datasets
REAL_PARAMS.update({
    'type': 'real',
    'path_in': PATH_DATA_REAL_DISC,
    'dataset': [],
    'max_iter': 50,
    'runs': 0,
    'path_out': PATH_RESULTS
})
# PATH_OUTPUT = os.path.join('..','..','results')


def create_args_parser(dict_params, methods):
    """ create simple arg parser with default values (input, output, dataset)

    :param dict dict_params:
    :param list(str) methods: list of possible methods
    :return obj: object argparse<...>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--path_in',
        type=str,
        required=True,
        help='path to the folder with input image dataset',
        default=dict_params.get('path_in', '')
    )
    parser.add_argument(
        '-o',
        '--path_out',
        type=str,
        required=True,
        help='path to the output with experiment results',
        default=dict_params.get('path_out', '')
    )
    parser.add_argument(
        '-t',
        '--type',
        type=str,
        required=False,
        help='switch between real and synth. images',
        default='real',
        choices=['real', 'synth']
    )
    parser.add_argument(
        '-n', '--name', type=str, required=False, help='specific name for this experiment', default=None
    )
    parser.add_argument(
        '-d', '--dataset', type=str, required=False, nargs='+', help='names of used datasets', default=None
    )
    parser.add_argument(
        '-p', '--nb_patterns', type=int, required=False, nargs='+', help='numbers of estimated patterns', default=None
    )
    parser.add_argument(
        '--nb_workers', type=int, required=False, default=NB_WORKERS, help='number of processes running in parallel'
    )
    parser.add_argument(
        '--method', type=str, required=False, nargs='+', default=None, help='possible APD methods', choices=methods
    )
    parser.add_argument(
        '--list_images', type=str, required=False, default=None, help='CSV file with list of images, supress `path_in`'
    )
    parser.add_argument(
        '-c', '--path_config', type=str, required=False, help='path to YAML configuration file', default=None
    )
    parser.add_argument('--debug', required=False, action='store_true', help='run in debug mode', default=False)
    parser.add_argument(
        '--unique', required=False, action='store_true', help='use time stamp for each experiment', default=False
    )
    return parser


def parse_arg_params(parser):
    """ parse basic args and return as dictionary

    :param obj parser: argparse
    :return dict:
    """
    args = vars(parser.parse_args())
    # remove not filled parameters
    args = {k: args[k] for k in args if args[k] is not None}
    for n in (k for k in args if k.startswith('path_') and args[k] is not None):
        args[n] = update_path(args[n])
        assert os.path.exists(args[n]), '%s' % args[n]
    for flag in ['nb_patterns', 'method']:
        if flag in args and not isinstance(args[flag], list):
            args[flag] = [args[flag]]

    if 'nb_patterns' in args:
        if is_list_like(args['nb_patterns']):
            args.update({'nb_labels': [lb + 1 for lb in args['nb_patterns']]})
        else:
            args['nb_labels'] = args['nb_patterns'] + 1
        del args['nb_patterns']

    return args


def parse_params(default_params, methods):
    """ parse arguments from command line

    :param dict default_params:
    :param list(str) methods: list of possible methods
    :return dict:
    """
    parser = create_args_parser(default_params, methods)
    params = copy.deepcopy(default_params)
    arg_params = parse_arg_params(parser)

    params.update(arg_params)

    # if YAML config exists update configuration
    if arg_params.get('path_config', None) is not None and os.path.isfile(arg_params['path_config']):
        logging.info('loading config: %s', arg_params['path_config'])
        d_config = load_config_yaml(arg_params['path_config'])
        logging.debug(string_dict(d_config, desc='LOADED CONFIG:'))

        # skip al keys with path or passed from arg params
        d_update = {k: d_config[k] for k in d_config if k not in arg_params or arg_params[k] is None}
        logging.debug(string_dict(d_update, desc='TO BE UPDATED:'))
        params.update(d_update)

    return params


def load_list_img_names(path_csv, path_in=''):
    """ loading images from a given list and if necessary add default data path

    :param str path_csv:
    :param str path_in:
    :return list(str):
    """
    assert os.path.exists(path_csv), '%s' % path_csv
    df = pd.read_csv(path_csv, index_col=False, header=None)
    assert len(df.columns) == 1, 'assume just single column'
    list_names = df.values[:, 0].tolist()
    # if the input path was set and the list are just names, no complete paths
    if os.path.exists(path_in) and not all(os.path.exists(p) for p in list_names):
        # to each image name add the input path
        list_names = [os.path.join(path_in, p) for p in list_names]
    return list_names


def generate_conf_suffix(d_params):
    """ generating suffix strung according given params

    :param dict d_params: dictionary
    :return str:

    >>> params = {'my_Param': 15}
    >>> generate_conf_suffix(params)
    '_my-Param=15'
    >>> params.update({'new_Param': 'abc'})
    >>> generate_conf_suffix(params)
    '_my-Param=15_new-Param=abc'
    """
    suffix = '_'
    suffix += '_'.join('{}={}'.format(k.replace('_', '-'), d_params[k]) for k in sorted(d_params) if k != 'param_idx')
    return suffix


# =============================================================================


class Experiment(ExperimentBase):
    """ main_train class for APD experiments State-of-the-Art and BPDL

    Examples
    --------

    >>> # SINGLE experiment
    >>> import glob
    >>> from bpdl.data_utils import DEFAULT_NAME_DATASET
    >>> params = {'dataset': DEFAULT_NAME_DATASET,
    ...           'path_in': os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME),
    ...           'path_out': PATH_RESULTS}
    >>> expt = Experiment(params, time_stamp=False)
    >>> expt.run(gt=True)
    >>> len(glob.glob(os.path.join(PATH_RESULTS, 'Experiment__*')))
    1
    >>> shutil.rmtree(expt.params['path_exp'], ignore_errors=True)

    >>> # SEQUENTIAL experiment
    >>> import glob
    >>> from bpdl.data_utils import DEFAULT_NAME_DATASET
    >>> params = {'dataset': DEFAULT_NAME_DATASET,
    ...           'path_in': os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME),
    ...           'path_out': PATH_RESULTS}
    >>> expt = Experiment(params, time_stamp=False)
    >>> expt.run(gt=False, iter_params=[{'r': 0}, {'r': 1}])
    >>> len(glob.glob(os.path.join(PATH_RESULTS, 'Experiment__*')))
    1
    >>> shutil.rmtree(expt.params['path_exp'], ignore_errors=True)

    >>> # PARALLEL experiment
    >>> import glob
    >>> from bpdl.data_utils import DEFAULT_NAME_DATASET
    >>> params = {'dataset': DEFAULT_NAME_DATASET,
    ...           'path_in': os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME),
    ...           'path_out': PATH_RESULTS,
    ...           'nb_workers': 2}
    >>> expt = Experiment(params, time_stamp=False)
    >>> expt.run(gt=False, iter_params=[{'r': 0}, {'r': 1}])
    >>> len(glob.glob(os.path.join(PATH_RESULTS, 'Experiment__*')))
    1
    >>> shutil.rmtree(expt.params['path_exp'], ignore_errors=True)
    """

    REQUIRED_PARAMS = ['dataset', 'path_in', 'path_out']

    def __init__(self, params, time_stamp=True):
        """ initialise class and set the experiment parameters

        :param dict params:
        :param bool time_stamp: mark if you want an unique folder per experiment
        """
        assert all([n in params for n in self.REQUIRED_PARAMS]), 'missing some required parameters'
        params = simplify_params(params)

        if 'name' not in params:
            dataset_name = params['dataset']
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[0]
            last_dir = os.path.basename(params['path_in'])
            params['name'] = '{}_{}_{}'.format(params.get('type', ''), last_dir, dataset_name)

        params['method'] = repr(self.__class__.__name__)
        if not os.path.exists(params['path_out']):
            os.mkdir(params['path_out'])

        # use constructor of parent class
        super(Experiment, self).__init__(params, time_stamp)

        self.df_results = pd.DataFrame()
        self._path_stat = os.path.join(self.params.get('path_exp'), RESULTS_TXT)
        self._list_img_paths = None
        # self.params.export_as(self._path_stat)
        with open(self._path_stat, 'w') as fp:
            fp.write(string_dict(self.params, desc='PARAMETERS:'))

    def _load_data_ground_truth(self):
        """ loading all GT suh as atlas and reconstructed images from GT encoding

        :param dict params: parameter settings
        """
        path_atlas = os.path.join(self.params.get('path_in'), DIR_NAME_DICTIONARY)
        self._gt_atlas = dataset_compose_atlas(path_atlas)
        if self.params.get('list_images') is not None:
            img_names = [os.path.splitext(os.path.basename(p))[0] for p in self._list_img_paths]
            self._gt_encoding = dataset_load_weights(self.path_data, img_names=img_names)
        else:
            self._gt_encoding = dataset_load_weights(self.params.get('path_in'))
        self._gt_images = reconstruct_samples(self._gt_atlas, self._gt_encoding)
        # self._images = [im.astype(np.uint8, copy=False) for im in self._images]

    def _load_images(self):
        """ load image data """
        self._images, self._image_names = dataset_load_images(
            self._list_img_paths, nb_workers=self.params.get('nb_workers', 1)
        )
        shapes = [im.shape for im in self._images]
        assert len(set(shapes)) == 1, 'multiple image sizes found: %r' % collections.Counter(shapes)

    def _load_data(self, gt=True):
        """ load all required data for APD and also ground-truth if required

        :param bool gt: search for the Ground Truth using standard names
        """
        logging.info('loading required data')
        self.path_data = os.path.join(self.params.get('path_in'), self.params.get('dataset'))
        # load according a csv list
        if self.params.get('list_images') is not None:
            # copy the list of selected images
            path_csv = os.path.expanduser(self.params.get('list_images'))
            if not os.path.exists(path_csv):
                path_csv = os.path.abspath(os.path.join(self.path_data, path_csv))
                shutil.copy(path_csv, os.path.join(self.params['path_exp'], os.path.basename(path_csv)))
            self._list_img_paths = load_list_img_names(path_csv)
        else:
            self._list_img_paths = find_images(self.path_data)

        assert len(self._list_img_paths) > 0, 'no images found'
        self._load_images()

        # loading  if it is set
        if gt:
            self._load_data_ground_truth()
            assert len(self._images) == len(self._gt_images), 'nb of input (%i) and reconst. (%i) images do not match' % (len(self._images), len(self._gt_images))
        logging.debug('loaded %i images', len(self._images))

    def run(self, gt=False, iter_params=None):
        """ the main procedure that load, perform and evaluate experiment

        :param bool gt: search for the Ground Truth using standard names
        :param list iter_params: list of possible configuration
        """
        logging.info('perform the complete experiment')

        # in case it single value make it iterable
        if is_list_like(iter_params):
            self.iter_params = copy.deepcopy(iter_params)
        elif isinstance(iter_params, dict):
            logging.info(string_dict(iter_params, desc='ITERATE PARAMETERS:'))
            self.iter_params = expand_params(iter_params)
        else:
            self.iter_params = None

        self._load_data(gt)
        self._perform()
        self._summarise()
        logging.info('FINISHED >]')
        logging.getLogger().handlers = []

    def _perform(self):
        """ perform experiment as sequence of iterated configurations """
        if is_list_like(self.iter_params):
            logging.info('iterate over %i configurations', len(self.iter_params))
            nb_workers = self.params.get('nb_workers', 1)

            for detail in WrapExecuteSequence(self._perform_once, self.iter_params, nb_workers, desc='experiments'):
                self.df_results = self.df_results.append(detail, ignore_index=True)
                logging.debug('partial results: %r', detail)
        else:
            logging.debug('perform single configuration')
            detail = self._perform_once({})
            self.df_results = pd.DataFrame([detail])

    def _estimate_atlas_weights(self, images, params):
        """ This is the method to be be over written by individual methods

        :param list(ndarray) images:
        :param dict params:
        :return tuple(ndarray,ndarray,dict):
        """
        del params
        atlas = np.zeros_like(self._images[0])
        weights = np.zeros((len(self._images), 0))
        return atlas, weights, None

    def _perform_once(self, d_params):
        """ perform single experiment

        :param dict d_params: used specific configuration
        :return dict: output statistic
        """
        detail = copy.deepcopy(self.params)
        detail.update(copy.deepcopy(d_params))
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
        except Exception:  # todo, optionally remove this try/catch
            logging.exception('FAIL estimate atlas for %r with %r', self.__class__, detail)
            atlas = np.zeros_like(self._images[0])
            weights = np.zeros((len(self._images), 0))
            extras = None
            detail['time'] = -0.

        logging.debug('estimated atlas of size %r and labels %r', atlas.shape, np.unique(atlas).tolist())

        weights_all = [weights_image_atlas_overlap_major(img, atlas) for img in self._images]
        weights_all = np.array(weights_all)

        logging.debug('estimated weights of size %r and summing %r', weights_all.shape, np.sum(weights_all, axis=0))

        self._export_atlas(atlas, suffix=detail['name_suffix'])
        self._export_coding(weights_all, suffix=detail['name_suffix'])
        self._export_extras(extras, suffix=detail['name_suffix'])

        detail.update(self._evaluate_base(atlas, weights_all))
        detail.update(self._evaluate_extras(atlas, weights, extras))

        return detail

    def _export_atlas(self, atlas, suffix=''):
        """ export estimated atlas

        :param ndarray atlas: np.array<height, width>
        :param str suffix:
        """
        n_img = NAME_ATLAS.format(suffix)
        export_image(self.params.get('path_exp'), atlas, n_img, stretch_range=False)
        path_atlas_rgb = os.path.join(self.params.get('path_exp'), n_img + '_rgb.png')
        logging.debug('exporting RGB atlas: %s', path_atlas_rgb)
        plt.imsave(path_atlas_rgb, atlas, cmap=plt.cm.jet)

    def _export_coding(self, weights, suffix=''):
        """ export estimated atlas

        :param ndarray weights:
        :param str suffix:
        """
        if not hasattr(self, '_image_names'):
            self._image_names = [str(i) for i in range(weights.shape[0])]
        df = format_table_weights(self._image_names, weights)

        path_csv = os.path.join(self.params.get('path_exp'), NAME_ENCODING.format(suffix))
        logging.debug('exporting encoding: %s', path_csv)
        df.to_csv(path_csv)

    def _export_extras(self, extras, suffix=''):
        """ export some extra parameters

        :param dict extras: dictionary with extra variables
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

        :param list(ndarray) images_rct: reconstructed images
        :return tuple(str,float):
        """
        assert images_rct is not None, 'missing any images to compare with'
        # error estimation from original reconstruction
        if im_type == 'GT' and hasattr(self, '_gt_images'):
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

    def _evaluate_base(self, atlas, weights):
        """ Compute the statistic for GT and estimated atlas and reconst. images

        :param ndarray atlas: np.array<height, width>
        :param list(ndarray) weights: np.array<nb_samples, nb_patterns>
        :return dict:
        """
        images_rct = reconstruct_samples(atlas, weights)
        tag, diff = self._evaluate_reconstruct(images_rct)
        stat = {'atlas ARS': self.__evaluate_atlas(atlas), 'reconst. diff %s' % tag: diff}
        return stat

    @classmethod
    def _evaluate_extras(self, atlas, weights, extras):
        """ some extra evaluation

        :param ndarray atlas: np.array<height, width>
        :param list(ndarray) weights: np.array<nb_samples, nb_patterns>
        :param dict extras:
        :return dict:
        """
        return {}

    def _summarise(self):
        """ summarise and export experiment results """
        logging.info('summarise the experiment')
        path_csv = os.path.join(self.params.get('path_exp'), RESULTS_CSV)
        logging.debug('save results: "%s"', path_csv)
        self.df_results.to_csv(path_csv)

        if hasattr(self, 'df_results') and not self.df_results.empty:
            df_stat = self.df_results.describe()
            # df_stat = df_stat[[c for c in EVAL_COLUMNS if c in df_stat.columns]]
            cols = [c for c in df_stat.columns if any([c.startswith(cc) for cc in EVAL_COLUMNS_START])]
            df_stat = df_stat[cols]
            with open(self._path_stat, 'a') as fp:
                fp.write('\n' * 3 + 'RESULTS: \n' + '=' * 9)
                fp.write('\n{}'.format(df_stat))
            logging.debug('statistic: \n%r', df_stat)


# =============================================================================


def simplify_params(dict_params):
    """ extract simple configuration dictionary

    :return dict:

    >>> params = simplify_params({'t': 'abc', 'n': [1, 2]})
    >>> pd.Series(params).sort_index()  #doctest: +NORMALIZE_WHITESPACE
    n      1
    t    abc
    dtype: object
    """
    d_params = {}
    for k in dict_params:
        d_params[k] = dict_params[k][0] if is_list_like(dict_params[k]) else dict_params[k]
    return d_params


def expand_params(dict_params, simple_config=None, skip_patterns=('--', '__')):
    """ extend parameters to a list

    :param dict dict_params: input dictionary with params
    :param dict simple_config: simple config dictionary
    :param list(str) skip_patterns: ignored configs
    :return:

    >>> param_range = {'t': ['abc'], 'n': [1, 2], 's': ('x', 'y'), 's--opts': ('a', 'b')}
    >>> params = expand_params(param_range)
    >>> df = pd.DataFrame(params)
    >>> df[sorted(df.columns)]# doctest: +NORMALIZE_WHITESPACE
       n    param_idx  s s--opts    t
    0  1  n-2#1_s-2#1  x       a  abc
    1  1  n-2#1_s-2#2  y       a  abc
    2  2  n-2#2_s-2#1  x       a  abc
    3  2  n-2#2_s-2#2  y       a  abc
    >>> params = expand_params({'s': ('x', 'y')}, {'old': 123.})
    >>> df = pd.DataFrame(params)
    >>> df[sorted(df.columns)]  # doctest: +NORMALIZE_WHITESPACE
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
        if any(ptn in k for ptn in skip_patterns):
            continue
        if not is_list_like(dict_params[k]) or len(dict_params[k]) <= 1:
            continue
        list_configs = extend_list_params(list_configs, k, dict_params[k])

    return list_configs


def parse_config_txt(path_config):
    """ open file with saved configuration and restore it

    :param str path_config:
    :return dict: {str: str}

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
    rec = re.compile(r'"(\S+)":\s+(.*)')
    dict_config = {n: convert_numerical(v) for n, v in rec.findall(text) if len(v) > 0}
    return dict_config


def activate_sigm(x, shift=0.12, slope=35.):
    """ transformation function for gene activations

    :param x: input values in range (0, 1)
    :param shift: shift the slope
    :param slope: steepness of the slope
    :return float: values in range (0, 1)

    >>> activate_sigm(0)
    0.0
    >>> activate_sigm(0.1)  # doctest: +ELLIPSIS
    0.32...
    >>> activate_sigm(1)
    1.0
    """
    sigm = lambda x, a, b: 1. / (1 + np.exp(b * (-x + a)))
    sigm_0, sigm_inf = sigm(0, shift, slope), sigm(1, shift, slope)
    val = (sigm(x, shift, slope) - sigm_0) / (sigm_inf - sigm_0)
    return val
