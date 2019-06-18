"""
walk over all experiment folders and extract configurations as DataFrame
per ech folder and add final statistic

EXAMPLES:
>> python run_parse_experiments_result.py \
    -i ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth \
    --name_results results.csv --name_config config.yaml --func_stat mean

>> python run_parse_experiments_result.py \
    -i ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth \
    --name_results results_NEW.csv --name_config config.yaml --func_stat none

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import argparse
import logging
import multiprocessing as mproc
from functools import partial

import numpy as np
import pandas as pd
from imsegm.utilities.data_io import update_path
from imsegm.utilities.experiments import (
    WrapExecuteSequence, try_decorator, string_dict, load_config_yaml)

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from experiments.experiment_general import parse_config_txt

NAME_INPUT_CONFIG = 'resultStat.txt'
NAME_INPUT_RESULT = 'results.csv'
TEMPLATE_NAME_OVERALL_RESULT = '%s_OVERALL.csv'
NB_WORKERS = int(mproc.cpu_count() * 0.9)

DICT_STATISTIC_FUNC = {
    'mean': np.mean,
    'median': np.median,
    'min': np.min,
    'max': np.max,
    'std': np.std
}

PARAMS = {
    'name_config': NAME_INPUT_CONFIG,
    'name_results': [NAME_INPUT_RESULT],
}


def parse_arg_params(params):
    """ create simple arg parser with default values (input, results, dataset)

    :return: argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path', type=str, required=True,
                        help='path to set of experiments')
    parser.add_argument('-c', '--name_config', type=str, required=False,
                        help='config file name', default=params['name_config'])
    parser.add_argument('-r', '--name_results', type=str, required=False,
                        nargs='*', default=params['name_results'], help='result file name')
    parser.add_argument('--result_columns', type=str, required=False,
                        default=None, nargs='*', help='important columns from results')
    parser.add_argument('-f', '--func_stat', type=str, required=False,
                        help='type od stat over results', default='none')
    parser.add_argument('--nb_workers', type=int, required=False,
                        default=NB_WORKERS, help='number of jobs running in parallel')

    args = vars(parser.parse_args())
    args['path'] = update_path(args['path'])
    assert os.path.isdir(args['path']), 'missing: %s' % args['path']
    return args


def parse_results_csv_summary(path_result, cols_sel, func_stat):
    """ load file with results and over specific cols aor all do an statistic

    :param str path_result:
    :param list(str) cols_sel:
    :param func_stat:
    :return dict:
    """
    dict_result = {}
    df_res = load_results_csv(path_result, cols_sel)
    if df_res is None or df_res.empty:
        return dict_result
    for col in df_res.columns:
        if df_res[col].dtype == int or df_res[col].dtype == float:
            dict_result[col] = np.round(func_stat(df_res[col]), 6)
    dict_result['nb_results'] = len(df_res)
    return dict_result


def load_results_csv(path_result, cols_select=None):
    """ load file with results and over specific cols aor all do an statistic

    :param str path_result:
    :param list(str) cols_select:
    :return: pd.DataFrame
    """
    if not os.path.exists(path_result):
        logging.warning('result file "%s" does not exist!', path_result)
        return None
    assert path_result.endswith('.csv'), '%s' % path_result
    df_res = pd.read_csv(path_result, index_col=None)
    df_res.drop([c for c in df_res.columns if c.startswith('Unnamed:')],
                axis=1, inplace=True)
    if cols_select is not None:
        cols_select = [c for c in df_res.columns]
        df_res = df_res[cols_select]
    return df_res


def load_multiple_results(path_expt, func_stat, params):
    df_results = pd.DataFrame()
    result_cols = params.get('result_columns', None)
    for name_results in params['name_results']:
        path_results = os.path.join(path_expt, name_results)
        if func_stat is not None:
            dict_res = parse_results_csv_summary(path_results,
                                                 result_cols,
                                                 func_stat)
            df_res = pd.DataFrame().from_dict(dict_res, orient='index').T
            df_results = df_results.join(df_res, how='outer')
        else:
            df_res = load_results_csv(path_results, result_cols)
            df_results = pd.concat([df_results, df_res])
    return df_results


@try_decorator
def parse_experiment_folder(path_expt, params):
    """ parse experiment folder, get configuration and results

    :param str path_expt: path to experiment folder
    :param dict params:
    :return dict:
    """
    assert os.path.isdir(path_expt), 'missing %s' % path_expt
    path_config = os.path.join(path_expt, params['name_config'])
    assert os.path.exists(path_config), 'missing %s' % path_config
    if any(path_config.endswith(ext) for ext in ['.yaml', '.yml']):
        dict_info = load_config_yaml(path_config)
    else:
        dict_info = parse_config_txt(path_config)
    logging.debug(' -> loaded params: %r', dict_info.keys())

    dict_info.update(count_folders_subdirs(path_expt))
    df_info = pd.DataFrame().from_dict(dict_info, orient='index').T
    try:
        func_stat = DICT_STATISTIC_FUNC.get(params['func_stat'], None)
        df_results = load_multiple_results(path_expt, func_stat, params)
    except Exception:
        logging.exception('load_multiple_results: %s', path_expt)
        df_results = pd.DataFrame()

    if len(df_results) == 0:
        return df_results

    logging.debug('  -> results params: %r', df_results.columns.tolist())
    list_cols = [c for c in df_info.columns
                 if c not in df_results.columns]
    df_infos = pd.concat([df_info[list_cols]] * len(df_results),
                         ignore_index=True)
    df_results = pd.concat([df_results, df_infos], axis=1)
    # df_results.fillna(method='pad', inplace=True)
    return df_results


def append_df_folder(df_all, df_folder):
    try:
        # df = pd.concat([df, df_folder], ignore_index=True)
        df_all = df_all.append(df_folder, ignore_index=True)
    except Exception:
        logging.exception('appending fail for DataFrame...')
    return df_all


def parse_experiments(params):
    """ with specific input parameters wal over result folder and parse it

    :param dict params:
    :return: DF<nb_experiments, nb_info>
    """
    logging.info('running parse Experiments results')
    logging.info(string_dict(params, desc='ARGUMENTS:'))
    assert os.path.isdir(params['path']), 'missing "%s"' % params['path']
    nb_workers = params.get('nb_workers', NB_WORKERS)

    df_all = pd.DataFrame()
    path_dirs = [p for p in glob.glob(os.path.join(params['path'], '*'))
                 if os.path.isdir(p)]
    logging.info('found experiments: %i', len(path_dirs))

    _wrapper_parse_folder = partial(parse_experiment_folder, params=params)
    for df_folder in WrapExecuteSequence(_wrapper_parse_folder, path_dirs, nb_workers):
        df_all = append_df_folder(df_all, df_folder)

    if isinstance(params['name_results'], list):
        name_results = '_'.join(os.path.splitext(n)[0]
                                for n in params['name_results'])
    else:
        name_results = os.path.splitext(params['name_results'])[0]

    df_all.reset_index(inplace=True)
    path_csv = os.path.join(params['path'],
                            TEMPLATE_NAME_OVERALL_RESULT % name_results)
    logging.info('export results as %s', path_csv)
    df_all.to_csv(path_csv, index=False)
    return df_all


def count_folders_subdirs(path_expt):
    """ count number and files in sub-folders

    :param str path_expt:
    :return dict: {str: int}
    """
    list_dirs = [p for p in glob.glob(os.path.join(path_expt, '*'))
                 if os.path.isdir(p)]
    list_sub_dirs = [len(glob.glob(os.path.join(p, '*'))) for p in list_dirs]
    dict_counts = {
        'folders': len(list_dirs),
        'files @dir': np.mean(list_sub_dirs) if len(list_sub_dirs) > 0 else 0
    }
    return dict_counts


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = parse_arg_params(PARAMS)
    parse_experiments(params)

    logging.info('DONE')
