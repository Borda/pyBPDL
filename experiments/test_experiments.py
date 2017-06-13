"""
run experiments tests

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import copy
import logging
import glob

# to suppress all visual, has to be on the beginning
import matplotlib
matplotlib.use('Agg')

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import apdl.dataset_utils as tl_datset
import experiments.experiment_apdl as expt_apdl
import experiments.run_experiments_all as r_all
import experiments.run_experiments_bpdl as r_bpdl
import experiments.run_parse_experiments_results as r_parse
import experiments.run_recompute_experiments_results as r_recomp


def test_experiments_soa(params=r_all.SYNTH_PARAMS):
    """ simple test of the experiments

    :param {str: value} dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)

    params.update({
        'type': 'synth',
        'dataset': tl_datset.DEFAULT_NAME_DATASET,
        'max_iter': 15,
        'nb_runs': 2,
        'nb_samples': 0.5,
    })

    for n, cls_expt in r_all.METHODS.iteritems():
        logging.info('testing %s by %s', n, cls_expt.__class__)
        expt = cls_expt(params)
        expt.run(iter_var='case', iter_vals=range(params['nb_runs']))
        del expt


def test_experiments_bpdl_inits(dict_params=r_bpdl.SYNTH_PARAMS):
    """  simple test of the experiments

    :param {str: any} dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)
    # experiment_pipeline_alpe_showcase()
    params = copy.deepcopy(dict_params)
    params.update({
        'type': 'synth',
        'dataset': tl_datset.DEFAULT_NAME_DATASET,
        'max_iter': 15,
        'nb_runs': 1,
    })

    for tp in r_bpdl.DICT_ATLAS_INIT.keys():
        params.update({'init_tp': tp})
        logging.info('RUN: ExperimentAPDL_raw')
        expt = r_bpdl.ExperimentAPDL_base(params)
        expt.run(iter_var='case', iter_vals=[0])
        del expt


def test_experiments_bpdl(dict_params=r_bpdl.SYNTH_PARAMS):
    """  simple test of the experiments

    :param {str: any} dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)
    # experiment_pipeline_alpe_showcase()
    params = copy.deepcopy(dict_params)
    params.update({
        'type': 'synth',
        'dataset': tl_datset.DEFAULT_NAME_DATASET,
        'max_iter': 15,
        'nb_runs': 2,
    })

    logging.info('RUN: ExperimentAPDL_raw')
    expt = r_bpdl.ExperimentAPDL_base(params)
    expt.run(iter_var='case', iter_vals=range(params['nb_runs']))
    del expt

    # negate default params
    params.update({
        'init_tp': 'random',
        'gc_reinit': not params['gc_reinit'],
        'ptn_split': not params['ptn_split'],
        'ptn_compact': not params['ptn_compact'],
        'nb_runs': 3,
    })

    logging.info('RUN: ExperimentAPDL_mp')
    expt_p = r_bpdl.ExperimentAPDL(params)
    expt_p.run(iter_var='case', iter_vals=range(params['nb_runs']))
    del expt_p


def test_experiments_postprocessing():
    params = {'res_cols': None, 'func_stat': 'none', 'type': 'synth',
              'fname_results': [expt_apdl.RESULTS_CSV],
              'fname_config': expt_apdl.CONFIG_JSON,
              'path': tl_datset.exist_path_bubble_up('results')}

    dir_expts = glob.glob(os.path.join(params['path'], '*'))
    # in case the the posporcesing is called before experiment themselves
    if len([p for p in dir_expts if os.path.isdir(p)]) == 0:
        test_experiments_soa()

    r_parse.parse_experiments(params)

    params.update({'fname_results': expt_apdl.RESULTS_CSV})
    r_recomp.parse_experiments(params, nb_jobs=2)

    name_res = os.path.splitext(expt_apdl.RESULTS_CSV)[0]
    params.update({'fname_results': [name_res + '_NEW.csv']})
    r_parse.parse_experiments(params, nb_jobs=2)


def main():
    """ main_real entry point """
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    test_experiments_bpdl()
    test_experiments_bpdl_inits()

    test_experiments_soa()

    test_experiments_postprocessing()

    logging.info('DONE')
    # plt.show()


if __name__ == '__main__':
    main()
