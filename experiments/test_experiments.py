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
import experiments.experiment_general as e_gen
import experiments.experiment_methods as e_mthd
import experiments.run_experiments as r_expt
import experiments.run_parse_experiments_result as r_parse
import experiments.run_recompute_experiments_result as r_recomp

PARAMS_TEST_SYNTH_UPDATE = {
    # 'dataset': tl_data.DEFAULT_NAME_DATASET,
    'max_iter': 5,
}


def test_experiments_soa_synth(params=r_expt.SYNTH_PARAMS):
    """ simple test of State-of-the-Art methods on Synthetic dataset

    :param {str: value} dict_params:
    """
    logging.getLogger().setLevel(logging.INFO)
    params.update(PARAMS_TEST_SYNTH_UPDATE)

    for n in r_expt.METHODS:
        cls_expt = r_expt.METHODS[n]
        logging.info('testing %s by %s', n, str(cls_expt))
        expt = cls_expt(params)
        expt.run(gt=True)
        del expt


def test_experiments_soa_real(params=r_expt.REAL_PARAMS):
    """ simple test of State-of-the-Art methods on Real images

    :param {str: value} dict_params:
    """
    logging.getLogger().setLevel(logging.INFO)
    params.update({
        'dataset': 'gene_small',
        'max_iter': 15,
    })

    for n in r_expt.METHODS_BASE:
        cls_expt = r_expt.METHODS_BASE[n]
        logging.info('testing %s by %s', n, str(cls_expt))
        expt = cls_expt(params)
        expt.run(gt=False, iter_params={'nb_labels': [4, 7]})
        del expt


def test_experiments_bpdl_initials(dict_params=r_expt.SYNTH_PARAMS):
    """  simple test of various BPDL initializations

    :param {str: any} dict_params:
    """
    logging.getLogger().setLevel(logging.INFO)
    # experiment_pipeline_alpe_showcase()
    params = e_gen.simplify_params(dict_params)
    params.update(PARAMS_TEST_SYNTH_UPDATE)

    for tp in e_mthd.DICT_ATLAS_INIT.keys():
        params.update({'init_tp': tp})
        logging.info('RUN: ExperimentBPDL-base, init: %s', tp)
        expt = e_mthd.ExperimentBPDL_base(params)
        expt.run(gt=True)
        del expt


def test_experiments_bpdl(dict_params=r_expt.SYNTH_PARAMS):
    """  simple & parallel test of BPDL and w. w/o deformation

    :param {str: any} dict_params:
    """
    logging.getLogger().setLevel(logging.INFO)
    # experiment_pipeline_alpe_showcase()
    params = e_gen.simplify_params(dict_params)
    params.update(PARAMS_TEST_SYNTH_UPDATE)

    logging.info('RUN: ExperimentBPDL-base')
    expt = e_mthd.ExperimentBPDL_base(params)
    expt.run(gt=True, iter_params={'deform_coef': [None, 1]})
    del expt

    # negate default params
    params = e_gen.simplify_params(dict_params)
    params.update({
        'init_tp': 'random',
        'gc_reinit': not params['gc_reinit'],
        'ptn_split': not params['ptn_split'],
        'ptn_compact': not params['ptn_compact'],
    })

    logging.info('RUN: ExperimentBPDL-parallel')
    expt_p = e_mthd.ExperimentBPDL(params)
    expt_p.run(gt=True, iter_params={'run': [0, 1]})
    del expt_p


def test_experiments_postprocessing():
    """ testing experiment postprocessing """
    logging.getLogger().setLevel(logging.INFO)
    params = {'res_cols': None, 'func_stat': 'none', 'type': 'synth',
              'name_results': [e_gen.RESULTS_CSV],
              'name_config': e_gen.CONFIG_JSON, 'nb_jobs': 2,
              'path': tl_data.update_path('results')}

    dir_expts = glob.glob(os.path.join(params['path'], '*'))
    # in case the the postporcesing is called before experiment themselves
    if len([p for p in dir_expts if os.path.isdir(p)]) == 0:
        test_experiments_soa_synth()

    r_parse.parse_experiments(params)

    params.update({'name_results': e_gen.RESULTS_CSV})
    r_recomp.parse_experiments(params)

    name_res = os.path.splitext(e_gen.RESULTS_CSV)[0]
    params.update({'name_results': [name_res + '_NEW.csv'], 'nb_jobs': 1})
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
