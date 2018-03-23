"""
run experiments with Stat-of-the-art methods

Example run:

>> python run_experiment_apd_all.py \
    -in /mnt/F464B42264B3E590/TEMP/apdDataset_00 \
    -out /mnt/F464B42264B3E590/TEMP/experiments_APD \
    --nb_jobs 1 

>> python run_experiment_apd_all.py \
    -in ~/Medical-drosophila/synthetic_data/apdDataset_v1 \
    -out ~/Medical-drosophila/TEMPORARY/experiments_APD

>> python run_experiment_apd_all.py \
    -in ~/Medical-drosophila/synthetic_data/apdDataset_v1 \
    -out ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth2 \
    --method BPDL

>> python run_experiment_apd_all.py --type real \
    -in ~/Medical-drosophila/TEMPORARY/type_1_segm_reg_binary \
    -out ~/Medical-drosophila/TEMPORARY/experiments_APD_real \
    --dataset gene_ssmall

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import gc, time
import logging
import traceback

import numpy as np
from sklearn.decomposition import SparsePCA, FastICA, DictionaryLearning, NMF
from skimage import segmentation
import tqdm

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_data
import bpdl.dictionary_learning as dl
import bpdl.pattern_atlas as ptn_dict
import experiments.experiment_general as expt_gen
import experiments.experiment_methods as e_methods

# standard multiprocessing version
METHODS = {
    'PCA': e_methods.ExperimentFastICA,
    'ICA': e_methods.ExperimentSparsePCA,
    'DL': e_methods.ExperimentDictLearn,
    'NMF': e_methods.ExperimentNMF,
    'BPDL': e_methods.ExperimentBPDL,
}

# working jut in single thread for pasiisng to image data to prtial jobs
METHODS_BASE = {
    'PCA': e_methods.ExperimentFastICA_base,
    'ICA': e_methods.ExperimentSparsePCA_base,
    'DL': e_methods.ExperimentDictLearn_base,
    'NMF': e_methods.ExperimentNMF_base,
    'BPDL': e_methods.ExperimentBPDL_base,
}

SYNTH_PARAMS = expt_gen.SYNTH_PARAMS
SYNTH_PARAMS.update({
    'dataset': ['datasetFuzzy_raw'],
    'method': list(METHODS.keys()),
    'max_iter': 25,  # 250, 25
})

REAL_PARAMS = expt_gen.REAL_PARAMS
REAL_PARAMS.update({
    'method': list(METHODS.keys()),
    'max_iter': 25,  # 250, 25
})

# INIT_TYPES = ['OWS', 'OWSr', 'GWS']
INIT_TYPES_ALL = sorted(e_methods.DICT_ATLAS_INIT.keys())
INIT_TYPES_NORM = [t for t in INIT_TYPES_ALL if 'tune' not in t]
INIT_TYPES_NORM_REAL = [t for t in INIT_TYPES_NORM if not t.startswith('GT')]
GRAPHCUT_REGUL = [0., 1e-9, 1e-3]


def experiment_pipeline_alpe_showcase(path_out):
    """ an simple show case to prove that the particular steps are computed

    :param path_in: str
    :param path_out: str
    :return:
    """
    path_atlas = os.path.join(expt_gen.SYNTH_PATH_APD,
                              tl_data.DIR_NAME_DICTIONARY)
    atlas = tl_data.dataset_compose_atlas(path_atlas)
    # plt.imshow(atlas)

    path_in = os.path.join(expt_gen.SYNTH_PATH_APD, tl_data.DEFAULT_NAME_DATASET)
    path_imgs = tl_data.find_images(path_in)
    imgs, _ = tl_data.dataset_load_images(path_imgs)
    # imgs = tl_data.dataset_load_images('datasetBinary_defNoise',
    #                                       path_base=SYNTH_PATH_APD)

    # init_atlas_org = ptn_dict.init_atlas_deform_original(atlas)
    # init_atlas_rnd = ptn_dict.init_atlas_random(atlas.shape, np.max(atlas))
    init_atlas_msc = ptn_dict.init_atlas_mosaic(atlas.shape, np.max(atlas))
    # init_encode_rnd = ptn_weigth.initialise_weights_random(len(imgs), np.max(atlas))

    atlas, weights, deforms = dl.bpdl_pipeline(
                        imgs, out_prefix='mosaic', init_atlas=init_atlas_msc,
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
        if params['nb_jobs'] <= 1:
            cls_expt = METHODS_BASE.get(method, None)
        else:
            cls_expt = METHODS.get(method, None)
        assert cls_expt is not None, 'not existing experiment "%s"' % method

        expt = cls_expt(params)
        expt.run(gt=user_gt, iter_params=iter_params)
        del expt
        gc.collect(), time.sleep(1)


def experiments_synthetic(params=SYNTH_PARAMS):
    """ run all experiments

    :param {str: ...} params:
    """
    params = expt_gen.parse_params(params)
    logging.info(expt_gen.string_dict(params, desc='PARAMETERS'))

    iter_params = {
        'nb_labels': params['nb_labels']
    }
    # iter_params = {
    #     'init_tp': INIT_TYPES_NORM,
    #     # 'ptn_split': [True, False],
    #     'ptn_compact': [True, False],
    #     'gc_regul': GRAPHCUT_REGUL,
    #     'nb_labels': params['nb_labels']
    # }

    experiment_iterate(params, iter_params, user_gt=True)


def experiments_real(params=REAL_PARAMS):
    """ run all experiments

    :param {str: ...} params:
    """
    params = expt_gen.parse_params(params)
    logging.info(expt_gen.string_dict(params, desc='PARAMETERS'))

    iter_params = {
        'nb_labels': params['nb_labels']
    }
    # iter_params = {
    #     'init_tp': INIT_TYPES_NORM_REAL,
    #     'ptn_compact': [True, False],
    #     'gc_regul': GRAPHCUT_REGUL,
    #     'nb_labels': params['nb_labels']
    # }

    experiment_iterate(params, iter_params, user_gt=False)


def main(arg_params):
    """ main entry point """
    # swap according dataset type
    if arg_params['type'] == 'synth':
        experiments_synthetic()
    elif arg_params['type'] == 'real':
        experiments_real()
    # plt.show()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    arg_params = expt_gen.parse_params({})
    main(arg_params)

    logging.info('DONE')
