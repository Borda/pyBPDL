"""
run experiments tests

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import glob

from imsegm.utilities.data_io import update_path

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from experiments.experiment_general import NB_WORKERS, RESULTS_CSV, CONFIG_YAML, simplify_params
from experiments.experiment_methods import DICT_ATLAS_INIT, ExperimentBPDL
from experiments.run_experiments import SYNTH_PARAMS, REAL_PARAMS, METHODS
import experiments.run_parse_experiments_result as r_parse
import experiments.run_recompute_experiments_result as r_recomp

PARAMS_TEST_SYNTH_UPDATE = {
    # 'dataset': tl_data.DEFAULT_NAME_DATASET,
    'max_iter': 5,
    'nb_workers': NB_WORKERS,
}


def test_experiments_soa_synth(params=SYNTH_PARAMS):
    """ simple test of State-of-the-Art methods on Synthetic dataset

    :param dict dict_params:
    """
    logging.getLogger().setLevel(logging.INFO)
    params.update(PARAMS_TEST_SYNTH_UPDATE)

    for n in METHODS:
        cls_expt = METHODS[n]
        logging.info('testing %s by %s', n, str(cls_expt))
        expt = cls_expt(params)
        expt.run(gt=True)
        del expt


def test_experiments_soa_real(params=REAL_PARAMS):
    """ simple test of State-of-the-Art methods on Real images

    :param dict dict_params:
    """
    logging.getLogger().setLevel(logging.INFO)
    params.update({
        'dataset': 'gene_small',
        'max_iter': 15,
    })

    for n in METHODS:
        cls_expt = METHODS[n]
        logging.info('testing %s by %s', n, str(cls_expt))
        expt = cls_expt(params)
        expt.run(gt=False, iter_params={'nb_labels': [4, 7]})
        del expt


def test_experiments_bpdl_initials(dict_params=SYNTH_PARAMS):
    """  simple test of various BPDL initializations

    :param dict dict_params:
    """
    logging.getLogger().setLevel(logging.INFO)
    # experiment_pipeline_alpe_showcase()
    params = simplify_params(dict_params)
    params.update(PARAMS_TEST_SYNTH_UPDATE)

    for tp in DICT_ATLAS_INIT.keys():
        params.update({'init_tp': tp})
        logging.info('RUN: ExperimentBPDL-base, init: %s', tp)
        expt = ExperimentBPDL(params)
        expt.run(gt=True)
        del expt


def test_experiments_bpdl(dict_params=SYNTH_PARAMS):
    """  simple & parallel test of BPDL and w. w/o deformation

    :param dict dict_params:
    """
    logging.getLogger().setLevel(logging.DEBUG)
    # experiment_pipeline_alpe_showcase()
    params = simplify_params(dict_params)
    params.update(PARAMS_TEST_SYNTH_UPDATE)

    logging.info('RUN: ExperimentBPDL-base')
    expt = ExperimentBPDL(params)
    expt.run(gt=True, iter_params={'deform_coef': [None, 0.15, 1]})
    del expt

    # negate default params
    params = simplify_params(dict_params)
    params.update({
        'init_tp': 'random',
        'gc_reinit': not params['gc_reinit'],
        'ptn_compact': not params['ptn_compact'],
        'nb_workers': 1,
    })

    logging.info('RUN: ExperimentBPDL-parallel')
    expt_p = ExperimentBPDL(params)
    expt_p.run(gt=True, iter_params={'run': [0, 1]})
    del expt_p


def test_experiments_postprocessing():
    """ testing experiment postprocessing """
    logging.getLogger().setLevel(logging.INFO)
    params = {
        'res_cols': None,
        'func_stat': 'none',
        'type': 'synth',
        'name_results': [RESULTS_CSV],
        'name_config': CONFIG_YAML,
        'nb_workers': 2,
        'path': update_path('results')
    }

    dir_expts = glob.glob(os.path.join(params['path'], '*'))
    # in case the the postporcesing is called before experiment themselves
    if not [p for p in dir_expts if os.path.isdir(p)]:
        test_experiments_soa_synth()

    r_parse.parse_experiments(params)

    params.update({'name_results': RESULTS_CSV})
    r_recomp.parse_experiments(params)

    name_res = os.path.splitext(RESULTS_CSV)[0]
    params.update({'name_results': [name_res + '_NEW.csv'], 'nb_workers': 1})
    r_parse.parse_experiments(params)


def main():
    """ main entry point """
    logging.info('test experiments BPDL')
    test_experiments_bpdl()
    logging.info('test experiments BPDL initials')
    test_experiments_bpdl_initials()

    logging.info('test experiments S-o-A synth')
    test_experiments_soa_synth()

    logging.info('test experiments S-o-A real')
    test_experiments_soa_real()

    logging.info('test_experiments_postprocessing')
    test_experiments_postprocessing()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')

    main()

    logging.info('DONE')
    # plt.show()
