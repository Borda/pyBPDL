"""
run experiments tests

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import copy
import logging
import glob

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_data
import experiments.experiment_apdl as expt_apdl
import experiments.run_experiments_all as r_all
import experiments.run_experiments_bpdl as r_bpdl
import experiments.run_parse_experiments_results as r_parse
import experiments.run_recompute_experiments_results as r_recomp


def test_experiments_soa_synth(params=r_all.SYNTH_PARAMS):
    """ simple test of State-of-the-Art methods on Synthetic dataset

    :param {str: value} dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)

    params.update({
        'dataset': tl_data.DEFAULT_NAME_DATASET,
        'max_iter': 15,
        'nb_runs': 2,
        'nb_samples': 0.5,
    })

    for n in r_all.METHODS:
        cls_expt = r_all.METHODS[n]
        logging.info('testing %s by %s', n, cls_expt.__class__)
        expt = cls_expt(params)
        expt.run(gt=True, iter_params={'run': range(params['nb_runs'])})
        del expt


def test_experiments_soa_real(params=r_all.REAL_PARAMS):
    """ simple test of State-of-the-Art methods on Real images

    :param {str: value} dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)

    params.update({
        'dataset': 'gene_small',
        'max_iter': 15,
        'nb_runs': 1,
    })

    for n in r_all.METHODS_BASE:
        cls_expt = r_all.METHODS_BASE[n]
        logging.info('testing %s by %s', n, cls_expt.__class__)
        expt = cls_expt(params)
        expt.run(gt=False, iter_params={'nb_labels': [3, 5, 9]})
        del expt


def test_experiments_bpdl_inits(dict_params=r_bpdl.SYNTH_PARAMS):
    """  simple test of various BPDL initializations

    :param {str: any} dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)
    # experiment_pipeline_alpe_showcase()
    params = expt_apdl.simplify_params(dict_params)
    params.update({
        'dataset': tl_data.DEFAULT_NAME_DATASET,
        'max_iter': 15,
        'nb_runs': 1,
    })

    for tp in r_bpdl.DICT_ATLAS_INIT.keys():
        params.update({'init_tp': tp})
        logging.info('RUN: ExperimentBPDL-base, init: %s', tp)
        expt = r_bpdl.ExperimentBPDL_base(params)
        expt.run(gt=True, iter_params={'run': range(params['nb_runs'])})
        del expt


def test_experiments_bpdl(dict_params=r_bpdl.SYNTH_PARAMS):
    """  simple & parallel test of BPDL

    :param {str: any} dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)
    # experiment_pipeline_alpe_showcase()
    params = expt_apdl.simplify_params(dict_params)
    params.update({
        'dataset': tl_data.DEFAULT_NAME_DATASET,
        'max_iter': 15,
        'nb_runs': 2,
    })

    logging.info('RUN: ExperimentBPDL-base')
    expt = r_bpdl.ExperimentBPDL_base(params)
    expt.run(gt=True, iter_params={'run': range(params['nb_runs'])})
    del expt

    # negate default params
    params = expt_apdl.simplify_params(dict_params)
    params.update({
        'init_tp': 'random',
        'gc_reinit': not params['gc_reinit'],
        'ptn_split': not params['ptn_split'],
        'ptn_compact': not params['ptn_compact'],
        'nb_runs': 3,
    })

    logging.info('RUN: ExperimentBPDL-parallel')
    expt_p = r_bpdl.ExperimentBPDL(params)
    expt_p.run(gt=True, iter_params={'run': range(params['nb_runs'])})
    del expt_p


def test_experiments_postprocessing():
    """ testing experiment postprocessing """
    params = {'res_cols': None, 'func_stat': 'none', 'type': 'synth',
              'fname_results': [expt_apdl.RESULTS_CSV],
              'fname_config': expt_apdl.CONFIG_JSON,
              'path': tl_data.update_path('results')}

    dir_expts = glob.glob(os.path.join(params['path'], '*'))
    # in case the the posporcesing is called before experiment themselves
    if len([p for p in dir_expts if os.path.isdir(p)]) == 0:
        test_experiments_soa_synth()

    r_parse.parse_experiments(params)

    params.update({'fname_results': expt_apdl.RESULTS_CSV})
    r_recomp.parse_experiments(params, nb_jobs=2)

    name_res = os.path.splitext(expt_apdl.RESULTS_CSV)[0]
    params.update({'fname_results': [name_res + '_NEW.csv']})
    r_parse.parse_experiments(params, nb_jobs=2)


def main():
    """ main entry point """
    logging.info('test experiments BPDL')
    test_experiments_bpdl()
    logging.info('test experiments BPDL inits')
    test_experiments_bpdl_inits()

    logging.info('test experiments S-o-A synth')
    test_experiments_soa_synth()

    logging.info('test experiments S-o-A real')
    test_experiments_soa_real()

    logging.info('test_experiments_postprocessing')
    test_experiments_postprocessing()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    main()

    logging.info('DONE')
    # plt.show()
