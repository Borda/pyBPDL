"""
run experiments with Stat-of-the-art methods

Example run:

>> python run_experiments.py --type synth \
    -i /mnt/F464B42264B3E590/TEMP/apdDataset_00 \
    -o /mnt/F464B42264B3E590/TEMP/experiments_APD \
    --nb_jobs 1 

>> python run_experiments.py --type synth \
    -i ~/Medical-drosophila/synthetic_data/apdDataset_v1 \
    -o ~/Medical-drosophila/TEMPORARY/experiments_APD

>> python run_experiments.py --type synth  --method BPDL \
    -i ~/Medical-drosophila/synthetic_data/apdDataset_v1 \
    -o ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth2 \

>> python run_experiments.py --type real \
    -i ~/Medical-drosophila/TEMPORARY/type_1_segm_reg_binary \
    -o ~/Medical-drosophila/TEMPORARY/experiments_APD_real \
    --dataset gene_ssmall

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import gc, time
import logging

import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.data_utils as tl_data
import bpdl.dictionary_learning as dl
import bpdl.pattern_atlas as ptn_dict
import experiments.experiment_general as expt_gen
import experiments.experiment_methods as e_methods

# standard multiprocessing version
METHODS = {
    'sPCA': e_methods.ExperimentSparsePCA,
    'fICA': e_methods.ExperimentFastICA,
    'DL': e_methods.ExperimentDictLearn,
    'NMF': e_methods.ExperimentNMF,
    'SC': e_methods.ExperimentSpectClust,
    'cICA': e_methods.ExperimentCanICA,
    'MSDL': e_methods.ExperimentMSDL,
    'BPDL': e_methods.ExperimentBPDL,
}
LIST_METHODS = sorted(list(METHODS.keys()))

SYNTH_PARAMS = expt_gen.SYNTH_PARAMS
SYNTH_PARAMS.update({
    'dataset': ['datasetFuzzy_raw'],
    'method': LIST_METHODS,
    'max_iter': 25,  # 250, 150
})

REAL_PARAMS = expt_gen.REAL_PARAMS
REAL_PARAMS.update({
    'method': LIST_METHODS,
    'max_iter': 25,  # 250, 150
})

# INIT_TYPES = ['OWS', 'OWSr', 'GWS']
INIT_TYPES_ALL = sorted(e_methods.DICT_ATLAS_INIT.keys())
INIT_TYPES_NORM = [t for t in INIT_TYPES_ALL if 'tune' not in t]
INIT_TYPES_NORM_REAL = [t for t in INIT_TYPES_NORM if not t.startswith('GT')]
GRAPHCUT_REGUL = [0., 1e-9, 1e-3]
SPECIAL_EXPT_PARAMS = ['dataset', 'method', 'OPTIONS']


def experiment_pipeline_alpe_showcase(path_out):
    """ an simple show case to prove that the particular steps are computed

    :param path_out: str
    :return (ndarray, ndarray):
    """
    path_atlas = os.path.join(expt_gen.SYNTH_PATH_APD,
                              tl_data.DIR_NAME_DICTIONARY)
    atlas = tl_data.dataset_compose_atlas(path_atlas)
    # plt.imshow(atlas)

    path_in = os.path.join(expt_gen.SYNTH_PATH_APD, tl_data.DEFAULT_NAME_DATASET)
    path_imgs = tl_data.find_images(path_in)
    imgs, _ = tl_data.dataset_load_images(path_imgs)
    # imgs = tl_data.dataset_load_images('datasetBinary_defNoise',
    #                                    path_base=SYNTH_PATH_APD)

    # init_atlas_org = ptn_atlas.init_atlas_deform_original(atlas)
    # init_atlas_rnd = ptn_atlas.init_atlas_random(atlas.shape, np.max(atlas))
    init_atlas_msc = ptn_dict.init_atlas_mosaic(atlas.shape, np.max(atlas))
    # init_encode_rnd = ptn_weigth.initialise_weights_random(len(imgs),
    #                                                        np.max(atlas))

    atlas, weights, deforms = dl.bpdl_pipeline(imgs, out_prefix='mosaic',
                                               init_atlas=init_atlas_msc,
                                               max_iter=9, out_dir=path_out)
    return atlas, weights


def experiment_iterate(params, iter_params, user_gt):
    if not expt_gen.is_list_like(params['dataset']):
        params['dataset'] = [params['dataset']]

    # tqdm_bar = tqdm.tqdm(total=len(l_params))
    for method, data in [(m, d) for m in params['method']
                         for d in params['dataset']]:
        params['dataset'] = data
        params['method'] = method
        cls_expt = METHODS.get(method, None)
        assert cls_expt is not None, 'not existing experiment "%s"' % method

        expt = cls_expt(params, time_stamp=params.get('unique', True))
        expt.run(gt=user_gt, iter_params=iter_params)
        del expt
        gc.collect(), time.sleep(1)


def filter_iterable_params(params):
    d_iter = {k: params[k] for k in params
              if expt_gen.is_iterable(params[k])
              and not any(x in k for x in SPECIAL_EXPT_PARAMS)}
    return d_iter


def experiments_synthetic(params=SYNTH_PARAMS):
    """ run all experiments

    :param {str: ...} params:
    """
    params = expt_gen.parse_params(params, LIST_METHODS)
    logging.info(expt_gen.string_dict(params, desc='PARAMETERS'))

    iter_params = filter_iterable_params(params)
    # iter_params = {
    #     'init_tp': INIT_TYPES_NORM,
    #     'ptn_compact': [True, False],
    #     'gc_regul': GRAPHCUT_REGUL,
    #     'nb_labels': params['nb_labels']
    # }

    experiment_iterate(params, iter_params, user_gt=True)


def experiments_real(params=REAL_PARAMS):
    """ run all experiments

    :param {str: ...} params:
    """
    params = expt_gen.parse_params(params, LIST_METHODS)
    logging.info(expt_gen.string_dict(params, desc='PARAMETERS'))

    iter_params = filter_iterable_params(params)
    # iter_params = {
    #     'init_tp': INIT_TYPES_NORM_REAL,
    #     'ptn_compact': [True, False],
    #     'gc_regul': GRAPHCUT_REGUL,
    #     'nb_labels': params['nb_labels']
    # }

    experiment_iterate(params, iter_params, user_gt=False)


def main(params):
    """ main entry point

    :param {} params:
    """
    # swap according dataset type
    if params['type'] == 'synth':
        experiments_synthetic()
    elif params['type'] == 'real':
        experiments_real()
    # plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    arg_params = expt_gen.parse_params({}, LIST_METHODS)
    if arg_params['debug']:
        logging.getLogger().setLevel(logging.DEBUG)

    main(arg_params)

    logging.info('DONE')
