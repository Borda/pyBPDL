"""
The base class for all Atomic Pattern Dictionary methods
such as the stat of the art and our newly developed

Copyright (C) 2015-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
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
import traceback
import multiprocessing as mproc

# to suppress all visu, has to be on the beginning
import matplotlib
if os.environ.get('DISPLAY','') == '':
    logging.warning('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as data_utils
import bpdl.pattern_atlas as ptn_dict

FORMAT_DT = '%Y%m%d-%H%M%S'
CONFIG_JSON = 'config.json'
RESULTS_TXT = 'resultStat.txt'
RESULTS_CSV = 'results.csv'
FILE_LOGS = 'logging.txt'

# fixing ImportError: No module named 'copy_reg' for Python3
if sys.version_info.major == 2:
    import types
    import copy_reg

    def _reduce_method(m):
        """ REQURED FOR MPROC POOL
        ISSUE: cPickle.PicklingError:
          Can't pickle <type 'instancemethod'>: attribute lookup __builtin__.instancemethod failed
        http://stackoverflow.com/questions/25156768/cant-pickle-type-instancemethod-using-pythons-multiprocessing-pool-apply-a
        """
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    copy_reg.pickle(types.MethodType, _reduce_method)

NB_THREADS = int(mproc.cpu_count() * .75)
PATH_DATA_SYNTH = data_utils.update_path('data')
PATH_DATA_REAL = data_utils.update_path('data')
PATH_RESULTS = 'results'
DEFAULT_PARAMS = {
    'computer': os.uname(),
    'nb_samples': None,
    'tol': 1e-3,
    'init_tp': 'GT-deform   ',  # msc, rnd, msc2, grd
    'max_iter': 250,  # 250, 25
    'gc_regul': 1e-9,
    'nb_labels': data_utils.NB_BIN_PATTERNS + 1,
    'nb_runs': NB_THREADS,  # 500
    'gc_reinit': True,
    'ptn_split': False,
    'ptn_compact': False,
    'overlap_mj': True,
}

SYNTH_DATASET_NAME = data_utils.DIR_MANE_SYNTH_DATASET
SYNTH_PATH_APD = os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET_NAME)
SYNTH_SUB_DATASETS_BINARY = [
    'datasetBinary_raw',
    'datasetBinary_noise',
    'datasetBinary_deform',
    'datasetBinary_defNoise']
SYNTH_SUB_DATASETS_PROBA = [
    'datasetProb_raw',
    'datasetProb_noise',
    'datasetProb_deform',
    'datasetProb_defNoise']
SYNTH_SUB_DATASETS_PROBA_NOISE = [
    'datasetProb_raw_gauss-0.001',
    'datasetProb_raw_gauss-0.005',
    'datasetProb_raw_gauss-0.010',
    'datasetProb_raw_gauss-0.025',
    'datasetProb_raw_gauss-0.050',
    'datasetProb_raw_gauss-0.075',
    'datasetProb_raw_gauss-0.100',
    'datasetProb_raw_gauss-0.125',
    'datasetProb_raw_gauss-0.150',
    'datasetProb_raw_gauss-0.200']
SYNTH_PARAMS = DEFAULT_PARAMS.copy()
SYNTH_PARAMS.update({
    'path_in': SYNTH_PATH_APD,
    'dataset': SYNTH_SUB_DATASETS_PROBA,
    'path_out': PATH_RESULTS,
    'type': 'synth',
})
# SYNTH_RESULTS_NAME = 'experiments_APD'

REAL_PARAMS = DEFAULT_PARAMS.copy()
REAL_PARAMS.update({
    'path_in': os.path.join(PATH_DATA_REAL, ''),
    'dataset': [],
    'path_out': PATH_RESULTS,
    'max_iter': 50,
    'nb_runs': 10})
# PATH_OUTPUT = os.path.join('..','..','results')


def create_args_parser(dict_params):
    """ create simple arg parser with default values (input, output, dataset)

    :param {str: ...} dict_params:
    :return: object argparse<...>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--path_in', type=str, required=True,
                        help='path to the input image dataset',
                        default=dict_params['path_in'])
    parser.add_argument('-out', '--path_out', type=str, required=True,
                        help='path to the output with experiment results',
                        default=dict_params['path_out'])
    parser.add_argument('-t', '--type', type=str, required=False,
                        help='switch between real and synth. images',
                        default='synth', choices=['real', 'synth'])
    parser.add_argument('-n', '--name', type=str, required=False,
                        help='specific name', default=None)
    parser.add_argument('--dataset', type=str, required=False,
                        nargs='+', default=None,
                        help='name of dataset to be used')
    parser.add_argument('-ptn', '--nb_patterns', type=int, required=False,
                        default=[2], nargs='+',
                        help='number of patterns to be estimated')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        default=NB_THREADS,
                        help='number of processes running in parallel')
    parser.add_argument('--method', type=str, required=False, nargs='+',
                        default=None,
                        help='possible APD methods',
                        choices=['PCA', 'ICA', 'DL', 'NMF', 'APDL'])
    parser.add_argument('-imgs', '--list_images', type=str, required=False,
                        default=None,
                        help='csv file with list of selected images')
    return parser


def parse_arg_params(parser):
    """ parse basic args and return as dictionary

    :param parser: argparse
    :return: {str: ...}
    """
    args = vars(parser.parse_args())
    # remove not filled parameters
    args = {k: args[k] for k in args if args[k] is not None}
    for n in (k for k in args if k.startswith('path_') and args[k] is not None):
        args[n] = data_utils.update_path(args[n])
        assert os.path.exists(args[n]), '%s' % args[n]
    if not isinstance(args['nb_patterns'], list):
        args['nb_patterns'] = [args['nb_patterns']]
    return args


def parse_params(default_params):
    """ parse arguments from command line

    :param {str: ...} default_params:
    :return: {str: ...}
    """
    parser = create_args_parser(default_params)
    params = copy.deepcopy(default_params)
    arg_params = parse_arg_params(parser)
    params.update(arg_params)
    return params


def load_list_img_names(name_csv, path_in=''):
    """

    :param str name_csv:
    :param str path_in:
    :return [str]:
    """
    # in case it is just name assume that it is in the input folder
    if not os.path.exists(name_csv):
        name_csv = os.path.abspath(os.path.join(path_in,
                                                os.path.expanduser(name_csv)))
    assert os.path.exists(name_csv), '%s' % name_csv
    df = pd.DataFrame.from_csv(name_csv, index_col=False, header=None)
    assert len(df.columns) == 1  # assume just single column
    list_names = df.as_matrix()[:, 0].tolist()
    # if the input path was set and the list are just names, no complete paths
    if os.path.exists(path_in) and not all(os.path.exists(p) for p in list_names):
        # to each image name add the input path
        list_names = [os.path.join(path_in, p) for p in list_names]
    return list_names


def create_experiment_folder(params, dir_name, stamp_unique=True, skip_load=True):
    """ create the experiment folder and iterate while there is no available

    :param {str: any} params:
    :param str dir_name:
    :param bool stamp_unique:
    :param bool skip_load:
    :return {str: any}:

    >>> p = {'path_out': '.'}
    >>> p = create_experiment_folder(p, 'my_test', False, skip_load=True)
    >>> 'computer' in p
    True
    >>> p['path_exp']
    './my_test_EXAMPLE'
    >>> shutil.rmtree(p['path_exp'])
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
    params.update({'computer': os.uname(),
                   'path_exp': path_expt})
    if os.path.exists(path_config) and not skip_load:
        logging.debug('loading saved params from file "%s"', CONFIG_JSON)
        with open(path_config, 'r') as fp:
            params = json.load(fp)
        params.update({'computer': os.uname(),
                       'path_exp': path_expt})
    logging.debug('saving params to file "%s"', CONFIG_JSON)
    with open(path_config, 'w') as f:
        json.dump(params, f)
    return params


def set_experiment_logger(path_out, file_name=FILE_LOGS, reset=True):
    """ set the logger to file

    :param path_out:
    :param file_name:
    :param reset:
    :return:
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


def string_dict(d, offset=30, desc='DICTIONARY'):
    """ transform dictionary to a formatted string

    :param {} d:
    :param int offset: length between name and value
    :param str desc: dictionary title
    :return str:

    >>> string_dict({'abc': 123})  #doctest: +NORMALIZE_WHITESPACE
    \'DICTIONARY: \\n"abc": 123\'
    """
    s = desc + ': \n'
    tmp_name = '{:' + str(offset) + 's} {}'
    rows = [tmp_name.format('"{}":'.format(n), d[n]) for n in sorted(d)]
    s += '\n'.join(rows)
    return str(s)


class ExperimentAPD(object):
    """
    main_train class for APD experiments State-of-the-Art and ALPE
    """

    def __init__(self, dict_params):
        """ initialise class and set the experiment parameters

        :param {str: ...} dict_params:
        """
        if not 'name' in dict_params:
            dataset_name = dict_params['dataset']
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[0]
            dict_params['name'] = '{}_{}_{}'.format(
                dict_params['type'], os.path.basename(dict_params['path_in']),
                dataset_name)
        if not os.path.exists(dict_params['path_out']):
            os.mkdir(dict_params['path_out'])
        self.params = copy.deepcopy(dict_params)
        self.params['class'] = self.__class__.__name__
        self.__check_exist_path()
        self.__create_folder()
        set_experiment_logger(self.params['path_exp'])
        logging.info('PARAMS: \n%s', string_dict(self.params))
        self.df_stat = pd.DataFrame()
        self.path_stat = os.path.join(self.params.get('path_exp'), RESULTS_TXT)
        self.list_img_paths = None
        # self.params.export_as(self.path_stat)
        str_params = 'PARAMETERS: \n' + \
                     '\n'.join(['"{}": \t {}'.format(k, self.params[k])
                                for k in self.params])
        logging.info(str_params)
        with open(self.path_stat, 'w') as fp:
            fp.write(str_params)

    def __check_exist_path(self):
        for p in [self.params[n] for n in self.params
                  if 'dir' in n.lower() or 'path' in n.lower()]:
            if not os.path.exists(p):
                raise Exception('given folder "%s" does not exist!' % p)
        for p in [self.params[n] for n in self.params if 'file' in n.lower()]:
            if not os.path.exists(p):
                raise Exception('given folder "%s" does not exist!' % p)

    def __create_folder(self):
        """ create the experiment folder and iterate while there is no available
        """
        # create results folder for experiments
        if not os.path.exists(self.params.get('path_out')):
            logging.error('no results folder "%s"' % self.p.get('path_out'))
            self.params['path_exp'] = ''
            return
        self.params = create_experiment_folder(self.params,
                                               self.__class__.__name__)

    def _load_data_ground_truth(self):
        """ loading all GT suh as atlas and reconstructed images from GT encoding

        :param params: {str: ...}, parameter settings
        """
        self.gt_atlas = data_utils.dataset_compose_atlas(self.params.get('path_in'))
        if self.list_img_paths is not None:
            img_names = [os.path.splitext(os.path.basename(p))[0]
                         for p in self.list_img_paths]
            gt_encoding = data_utils.dataset_load_weights(self.params.get('path_in'),
                                                          img_names=img_names)
        else:
            gt_encoding = data_utils.dataset_load_weights(self.params.get('path_in'))
        self.gt_img_rct = ptn_dict.reconstruct_samples(self.gt_atlas, gt_encoding)

    def _load_data(self, gt=True):
        """ load all required data for APD and also ground-truth if required

        :param bool gt:
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
            self.list_img_paths = load_list_img_names(path_csv, path_in=self.path_data)
        self._load_images()
        if gt:
            self._load_data_ground_truth()
            assert len(self.imgs) == len(self.gt_img_rct)
        logging.debug('loaded %i images', len(self.imgs))
        # self.imgs = [im.astype(np.uint8, copy=False) for im in self.imgs]

    def _load_images(self):
        """ load image data """
        self.imgs, self._im_names = data_utils.dataset_load_images(
            self.path_data, path_imgs=self.list_img_paths,
            nb_jobs=self.params.get('nb_jobs', 1))

    def run(self, gt=True, iter_var='case', iter_vals=range(1)):
        """ the main_real procedure that load, perform and evaluete experiment

        :param bool gt:
        :param str iter_var: name of variable to be iterated in the experiment
        :param [] iter_vals: list of possible values for :param iter_var:
        """
        logging.info('perform the complete experiment')
        self.iter_var_name = iter_var
        self.iter_values = iter_vals
        self._load_data(gt)
        self._perform()
        self._evaluate()
        self._summarise()
        logging.getLogger().handlers = []

    def _perform(self):
        """ perform experiment as sequence of iterated configurations """
        self._perform_sequence()

    def _perform_sequence(self):
        """ iteratively change a single experiment parameter with the same data
        """
        logging.info('perform_sequence in single thread')
        self.l_stat = []
        tqdm_bar = tqdm.tqdm(total=len(self.iter_values))
        for v in self.iter_values:
            self.params[self.iter_var_name] = v
            logging.debug(' -> set iterable "%s" on %s', self.iter_var_name,
                         repr(self.params[self.iter_var_name]))
            t = time.time()
            stat = self._perform_once(v)
            stat['time'] = time.time() - t
            self.l_stat.append(stat)
            logging.info('partial results: %s', repr(stat))
            # just partial export
            self._evaluate()
            tqdm_bar.update(1)

    def _perform_once(self, v):
        """ perform single experiment

        :param any v:
        :return {str: val}:
        """
        stat = {self.iter_var_name: v}
        return stat

    def _export_atlas(self, posix=''):
        """ export estimated atlas

        :param np.array<height, width> atlas:
        :param str posix:
        """
        assert hasattr(self, 'atlas')
        n_img = 'atlas{}'.format(posix)
        data_utils.export_image(self.params.get('path_exp'), self.atlas, n_img)

    def _export_coding(self, posix=''):
        """ export estimated atlas

        :param np.array<height, width> atlas:
        :param str posix:
        """
        assert hasattr(self, 'w_bins')
        n_csv = 'encoding{}.csv'.format(posix)
        path_csv = os.path.join(self.params.get('path_exp'), n_csv)
        if not hasattr(self, '_im_names'):
            self._im_names = [str(i) for i in range(self.w_bins.shape[0])]
        df = pd.DataFrame(data=self.w_bins, index=self._im_names[:len(self.w_bins)])
        df.columns = ['ptn {:02d}'.format(lb + 1) for lb in df.columns]
        df.index.name = 'image'
        df.to_csv(path_csv)

    def _compute_statistic_gt(self, imgs_rct=None):
        """ compute the statistic gor GT and estimated atlas and reconstructed images

        :param np.array<height, width> atlas:
        :param [np.array<height, width>] imgs_rct:
        :return {str: float, }:
        """
        stat = {}
        logging.debug('compute static - %s', hasattr(self, 'gt_atlas'))
        if hasattr(self, 'gt_atlas') and hasattr(self, 'atlas'):
            if self.gt_atlas.shape == self.atlas.shape:
                stat['atlas_ARS'] = metrics.adjusted_rand_score(self.gt_atlas.ravel(),
                                                                self.atlas.ravel())
        logging.debug('compute reconstruction - %s', hasattr(self, 'gt_img_rct'))
        # error estimation from original reconstruction
        if hasattr(self, 'gt_img_rct') and imgs_rct is not None:
            # imgs_rct = ptn_dict.reconstruct_samples(self.atlas, self.w_bins)
            # imgs_rct = self._binarize_img_reconstruction(imgs_rct)
            imgs_gt = self.gt_img_rct[:len(imgs_rct)]
            diff = np.asarray(imgs_gt) - np.asarray(imgs_rct)
            stat['reconstruct_diff'] = np.sum(abs(diff)) / float(np.prod(diff.shape))
        elif hasattr(self, 'imgs') and imgs_rct is not None:
            imgs = self.imgs[:len(imgs_rct)]
            diff = np.asarray(imgs) - np.asarray(imgs_rct)
            stat['reconstruct_diff'] = np.sum(abs(diff)) / float(np.prod(diff.shape))
        return stat

    def _evaluate(self):
        """ evaluate experiment with given GT """
        self.df_stat = pd.DataFrame()
        for stat in self.l_stat:
            self.df_stat = self.df_stat.append(stat, ignore_index=True)
        if self.iter_var_name in stat:
            self.df_stat.set_index(self.iter_var_name, inplace=True)
        path_csv = os.path.join(self.params.get('path_exp'), RESULTS_CSV)
        logging.debug('save results: "%s"', path_csv)
        self.df_stat.to_csv(path_csv)

    def _summarise(self):
        """ summarise and export experiment results """
        logging.info('summarise the experiment')
        if hasattr(self, 'df_stat') and not self.df_stat.empty:
            with open(self.path_stat, 'a') as fp:
                fp.write('\n' * 3 + 'RESULTS: \n' + '=' * 9)
                fp.write('\n{}'.format(self.df_stat.describe()))
            logging.debug('statistic: \n%s', repr(self.df_stat.describe()))


class ExperimentAPD_parallel(ExperimentAPD):
    """
    run the experiment in multiple threads
    """

    def __init__(self, dict_params, nb_jobs=NB_THREADS):
        """ initialise parameters and nb jobs in parallel

        :param {str: ...} dict_params:
        :param int nb_jobs:
        """
        super(ExperimentAPD_parallel, self).__init__(dict_params)
        self.nb_jobs = nb_jobs

    def _load_images(self):
        """ load image data """
        self.imgs, self._im_names = data_utils.dataset_load_images(
            self.path_data, path_imgs=self.list_img_paths, nb_jobs=self.nb_jobs)

    def _warp_perform_once(self, v):
        try:
            self.params[self.iter_var_name] = v
            logging.debug(' -> set iterable "%s" on %s', self.iter_var_name,
                          repr(self.params[self.iter_var_name]))
            t = time.time()
            # stat = super(ExperimentAPD_mp, self)._perform_once(v)
            stat = self._perform_once(v)
            stat['time'] = time.time() - t
            logging.info('partial results: %s', repr(stat))
        except:
            # fixme, optionally remove the try/catch
            logging.error(traceback.format_exc())
            exit(1)
        return stat

    # def _perform_once(self, v):
    #     """ perform single experiment
    #
    #     :param v: value
    #     :return: {str: val}
    #     """
    #     t = time.time()
    #     self.params[self.iter_var_name] = v
    #     stat = super(ExperimentAPD_mp, self)._perform_once(v)
    #     stat['time'] = time.time() - t
    #     return stat

    def _perform_sequence(self):
        """ perform sequence in multiprocessing pool """
        logging.debug('perform_sequence in %i threads for %i values',
                      self.nb_jobs, len(self.iter_values))
        # ISSUE with passing large date to processes so the images are saved
        # and loaded in particular process again
        # p_imgs = os.path.join(self.params.get('path_exp'), 'input_images.npz')
        # np.savez(open(p_imgs, 'w'), imgs=self.imgs)

        self.l_stat = []
        # tqdm_bar = tqdm.tqdm(total=len(self.iter_values))
        mproc_pool = mproc.Pool(self.nb_jobs)
        for stat in mproc_pool.map(self._warp_perform_once, self.iter_values):
            self.l_stat.append(stat)
            self._evaluate()
            # tqdm_bar.update(1)
        mproc_pool.close()
        mproc_pool.join()

        # remove temporary image file
        # os.remove(p_imgs)


def extend_list_params(list_params, name_param, list_options):
    """ extend the parameter list by all sub-datasets

    :param [{str: ...}] list_params:
    :param str name_param:
    :param [] list_options:
    :return [{str: ...}]:

    >>> params = extend_list_params([{'a': 1}], 'a', [3, 4])
    >>> pd.DataFrame(params)  # doctest: +NORMALIZE_WHITESPACE
       a param_idx
    0  3     a_1/2
    1  4     a_2/2
    """
    if not isinstance(list_options, list):
        list_options = [list_options]
    list_params_new = []
    for p in list_params:
        for i, v in enumerate(list_options):
            p_new = p.copy()
            p_new.update({name_param: v})
            p_new['param_idx'] = \
                '%s_%i/%i' % (name_param, i + 1, len(list_options))
            list_params_new.append(p_new)
    return list_params_new
