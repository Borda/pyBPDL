"""
run experiments tests

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import copy
import logging

import apdl.dataset_utils as tl_datset
import run_experiment_apd_all
import run_experiment_apd_apdl


def test_experiments_soa(params=run_experiment_apd_all.SYNTH_PARAMS):
    """ simple test of the experiments

    :param {str: value} dict_params:
    """
    logging.basicConfig(level=logging.DEBUG)

    params['type'] = 'synth'
    params['nb_runs'] = 2
    params['nb_samples'] = 0.5
    params['dataset'] = tl_datset.DEFAULT_NAME_DATASET

    for n, cls_expt in run_experiment_apd_all.METHODS.iteritems():
        logging.info('testing %s by %s', n, cls_expt.__class__)
        expt = cls_expt(params)
        expt.run(iter_var='case', iter_vals=range(params['nb_runs']))


def test_experiments_apdl(dict_params=run_experiment_apd_apdl.SYNTH_PARAMS):
    """  simple test of the experiments

    :param {str: any} dict_params:
    """
    logging.basicConfig(level=logging.DEBUG)
    # experiment_pipeline_alpe_showcase()
    params = copy.deepcopy(dict_params)
    params['nb_runs'] = 3
    params['dataset'] = tl_datset.DEFAULT_NAME_DATASET

    logging.info('RUN: ExperimentAPDL_raw')
    expt = run_experiment_apd_apdl.ExperimentAPDL_base(params)
    expt.run(iter_var='case', iter_vals=range(params['nb_runs']))

    logging.info('RUN: ExperimentAPDL_mp')
    expt_p = run_experiment_apd_apdl.ExperimentAPDL(params)
    expt_p.run(iter_var='case', iter_vals=range(params['nb_runs']))


def main():
    """ main_real entry point """
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    test_experiments_apdl()

    test_experiments_soa()

    logging.info('DONE')
    # plt.show()


if __name__ == '__main__':
    main()
