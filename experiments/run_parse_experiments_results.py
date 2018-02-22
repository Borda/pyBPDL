"""
walk over all experiment folders and extract configurations as DataFrame
per ech folder and add final statistic

EXAMPLES:
>> python run_parse_experiments_results.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth \
    --fname_results results.csv --fname_config config.json --func_stat mean

>> python run_parse_experiments_results.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth \
    --fname_results results_NEW.csv --fname_config config.json --func_stat none

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import re
import glob
import json
import argparse
import traceback
import logging
import multiprocessing as mproc
from functools import partial

import tqdm
import numpy as np
import pandas as pd

NAME_INPUT_CONFIG = 'resultStat.txt'
NAME_INPUT_RESULT = 'results.csv'
NAME_OVERALL_RESULT = '%s_OVERALL.csv'
NB_THREADS = int(mproc.cpu_count() * 0.9)

DICT_STATISTIC_FUNC = {
    'mean': np.mean,
    'median': np.median,
    'min': np.min,
    'max': np.max,
    'std': np.std
}


def create_args_parser():
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='path to set of experiments')
    parser.add_argument('--fname_config', type=str, required=False,
                        help='config file name', default=NAME_INPUT_CONFIG)
    parser.add_argument('--fname_results', type=str, required=False, nargs='*',
                        help='result file name', default=[NAME_INPUT_RESULT])
    # parser.add_argument('-t', '--type', type=str, required=False, default='synth',
    #                     help='type of experiment data', choices=['synth', 'real'])
    parser.add_argument('--res_cols', type=str, required=False, default=None,
                        nargs='*', help='important columns from results')
    parser.add_argument('--func_stat', type=str, required=False,
                        help='type od stat over results', default='none')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        help='number running in parallel', default=NB_THREADS)
    return parser


def parse_arg_params(parser):
    """ parse basic args and return as dictionary

    :param parser: argparse
    :return: {str: ...}
    """
    args = vars(parser.parse_args())
    args['path'] = os.path.abspath(os.path.expanduser(args['path']))
    return args


def parse_config_txt(path_config):
    """ open file with saved configuration and restore it

    :param str path_config:
    :return {str: str}:
    """
    if not os.path.exists(path_config):
        logging.error('config file "%s" does not exist!', path_config)
        return {}
    with open(path_config, 'r') as f:
        text = ''.join(f.readlines())
    rec = re.compile('"(\S+)":\s+(.*)\n')
    dict_config = {n: v for n, v in rec.findall(text)}
    return dict_config


def parse_results_csv_summary(path_result, cols_sel, func_stat):
    """ load file with results and over specific cols aor all do an statistic

    :param str path_result:
    :param [str] cols_sel:
    :param func_stat:
    :return {str: any}:
    """
    dict_result = {}
    df_res = load_results_csv(path_result, cols_sel)
    if df_res is None:
        return dict_result
    for col in df_res.columns:
        if df_res[col].dtype == int or df_res[col].dtype == float:
            dict_result[col] = np.round(func_stat(df_res[col]), 6)
    dict_result['nb_res'] = len(df_res)
    return dict_result


def load_results_csv(path_result, cols_select):
    """ load file with results and over specific cols aor all do an statistic

    :param str path_result:
    :param [str] cols_select:
    :param fn_stat:
    :return: pd.DataFrame
    """
    if not os.path.exists(path_result):
        logging.warning('result file "%s" does not exist!', path_result)
        return None
    assert path_result.endswith('.csv'), '%s' % path_result
    df_res = pd.DataFrame().from_csv(path_result, index_col=None)
    if cols_select is not None:
        cols_select = [c for c in df_res.columns]
        df_res = df_res[cols_select]
    return df_res


def load_multiple_results(path_expt, func_stat, params):
    df_results = pd.DataFrame()
    for name_results in params['fname_results']:
        path_results = os.path.join(path_expt, name_results)
        if func_stat is not None:
            dict_res = parse_results_csv_summary(path_results,
                                                 params['res_cols'],
                                                 func_stat)
            df_res = pd.DataFrame().from_dict(dict_res, orient='index').T
            df_results = df_results.join(df_res, how='outer')
        else:
            df_res = load_results_csv(path_results, params['res_cols'])
            df_results = pd.concat([df_results, df_res])
    return df_results


def parse_experiment_folder(path_expt, params):
    """ parse experiment folder, get configuration and results

    :param str path_expt: path to experiment folder
    :param {str: any} params:
    :return {str: any}:
    """
    assert os.path.isdir(path_expt), 'missing %s' % path_expt
    path_config = os.path.join(path_expt, params['fname_config'])
    assert os.path.exists(path_config), 'missing %s' % path_config
    if path_config.endswith('json'):
        dict_info = json.load(open(path_config, 'r'))
    else:
        dict_info = parse_config_txt(path_config)
    logging.debug(' -> loaded params: %s', repr(dict_info.keys()))

    dict_info.update(count_folders_subfolders(path_expt))
    df_info = pd.DataFrame().from_dict(dict_info, orient='index').T
    try:
        func_stat = DICT_STATISTIC_FUNC.get(params['func_stat'], None)
        df_results = load_multiple_results(path_expt, func_stat, params)
    except:
        logging.error(traceback.format_exc())
        df_results = pd.DataFrame()

    if len(df_results) == 0:
        return df_results

    logging.debug('  -> results params: %s', repr(df_results.columns.tolist()))
    list_cols = [c for c in df_info.columns if c not in df_results.columns]
    df_infos = pd.concat([df_info[list_cols]] * len(df_results), ignore_index=True)
    df_results = pd.concat([df_results, df_infos], axis=1)
    # df_results.fillna(method='pad', inplace=True)
    return df_results


def try_parse_experiment_folder(path_expt, params):
    """

    :param str path_expt:
    :param params: {str: ...}
    :return:
    """
    try:
        df_folder = parse_experiment_folder(path_expt, params)
    except:
        df_folder = None
        logging.warning(traceback.format_exc())
    return df_folder


def parse_experiments(params, nb_jobs=NB_THREADS):
    """ with specific input parameters wal over result folder and parse it

    :param {str: any} params:
    :return: DF<nb_experiments, nb_info>
    """
    logging.info('running parse Experiments results')
    logging.info('ARGUMENTS: \n%s', '\n'.join('"{}": \t {}'.format(k, params[k])
                                              for k in params))
    assert os.path.exists(params['path']), 'path to expt "%s"' % params['path']

    df_all = pd.DataFrame()
    path_dirs = [p for p in glob.glob(os.path.join(params['path'], '*'))
                 if os.path.isdir(p)]
    logging.info('found experiments: %i', len(path_dirs))
    wrapper_parse_folder = partial(try_parse_experiment_folder, params=params)
    tqdm_bar = tqdm.tqdm(total=len(path_dirs))

    if nb_jobs > 1:
        logging.debug('perform_sequence in %i threads', nb_jobs)
        mproc_pool = mproc.Pool(nb_jobs)
        for df_folder in mproc_pool.imap_unordered(wrapper_parse_folder,
                                                   path_dirs):
            try:
                # df = pd.concat([df, df_folder], ignore_index=True)
                df_all = df_all.append(df_folder, ignore_index=True)
            except:
                logging.error(traceback.format_exc())
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for path_expt in path_dirs:
            logging.debug('folder %s', path_expt)
            try:
                df_folder = parse_experiment_folder(path_expt, params)
                # df = pd.concat([df, df_folder], axis=1, ignore_index=True)
                df_all = df_all.append(df_folder, ignore_index=True)
            except:
                logging.warning('no data extracted from folder %s',
                                os.path.basename(path_expt))
                logging.error(traceback.format_exc())
            tqdm_bar.update()

    if isinstance(params['fname_results'], list):
        fname_results = '_'.join(os.path.splitext(n)[0]
                                 for n in params['fname_results'])
    else:
        fname_results = os.path.splitext(params['fname_results'])[0]
    csv_name = NAME_OVERALL_RESULT % fname_results
    logging.info('export results as %s', os.path.join(params['path'], csv_name))
    df_all.to_csv(os.path.join(params['path'], csv_name), index=False)
    return df_all


def count_folders_subfolders(path_expt):
    """ count number and files in sub-folders

    :param str path_expt:
    :return {str: int}:
    """
    list_dirs = [p for p in glob.glob(os.path.join(path_expt, '*'))
                 if os.path.isdir(p)]
    list_sub_dirs = [len(glob.glob(os.path.join(p, '*'))) for p in list_dirs]
    dict_counts = {'folders': len(list_dirs),
                   'files @dir': np.mean(list_sub_dirs)}
    return dict_counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    parser = create_args_parser()
    params = parse_arg_params(parser)
    parse_experiments(params, nb_jobs=params['nb_jobs'])

    logging.info('DONE')
