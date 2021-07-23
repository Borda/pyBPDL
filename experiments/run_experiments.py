"""
run experiments with Stat-of-the-art methods

Example run:

>> python run_experiments.py --type synth \
    -i /mnt/F464B42264B3E590/TEMP/apdDataset_00 \
    -o /mnt/F464B42264B3E590/TEMP/experiments_APD \
    --nb_workers 1

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

import gc
import logging
import os
import sys
import time

import numpy as np
from imsegm.utilities.experiments import string_dict

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bpdl.data_utils import (
    dataset_compose_atlas,
    dataset_load_images,
    DEFAULT_NAME_DATASET,
    DIR_NAME_DICTIONARY,
    find_images,
)
from bpdl.dictionary_learning import bpdl_pipeline
from bpdl.pattern_atlas import init_atlas_mosaic
from bpdl.utilities import is_iterable, is_list_like
from experiments.experiment_general import parse_params, REAL_PARAMS, SYNTH_PARAMS, SYNTH_PATH_APD
from experiments.experiment_methods import (
    DICT_ATLAS_INIT,
    ExperimentBPDL,
    ExperimentCanICA,
    ExperimentDictLearn,
    ExperimentFastICA,
    ExperimentMSDL,
    ExperimentNMF,
    ExperimentSparsePCA,
    ExperimentSpectClust,
)

# standard multiprocessing version
METHODS = {
    'sPCA': ExperimentSparsePCA,
    'fICA': ExperimentFastICA,
    'DL': ExperimentDictLearn,
    'NMF': ExperimentNMF,
    'SC': ExperimentSpectClust,
    'cICA': ExperimentCanICA,
    'MSDL': ExperimentMSDL,
    'BPDL': ExperimentBPDL,
}
LIST_METHODS = sorted(list(METHODS.keys()))

SYNTH_PARAMS.update({
    'dataset': ['datasetFuzzy_raw'],
    'method': LIST_METHODS,
    'max_iter': 25,  # 250, 150
})

REAL_PARAMS.update({
    'method': LIST_METHODS,
    'max_iter': 25,  # 250, 150
})

# INIT_TYPES = ['OWS', 'OWSr', 'GWS']
INIT_TYPES_ALL = sorted(DICT_ATLAS_INIT.keys())
INIT_TYPES_NORM = [t for t in INIT_TYPES_ALL if 'tune' not in t]
INIT_TYPES_NORM_REAL = [t for t in INIT_TYPES_NORM if not t.startswith('GT')]
GRAPHCUT_REGUL = [0., 1e-9, 1e-3]
SPECIAL_EXPT_PARAMS = ['dataset', 'method', 'OPTIONS']


def experiment_pipeline_alpe_showcase(path_out):
    """ an simple show case to prove that the particular steps are computed

    :param path_out: str
    :return tuple(ndarray,ndarray):
    """
    path_atlas = os.path.join(SYNTH_PATH_APD, DIR_NAME_DICTIONARY)
    atlas = dataset_compose_atlas(path_atlas)
    # plt.imshow(atlas)

    path_in = os.path.join(SYNTH_PATH_APD, DEFAULT_NAME_DATASET)
    path_imgs = find_images(path_in)
    imgs, _ = dataset_load_images(path_imgs)
    # imgs = tl_data.dataset_load_images('datasetBinary_defNoise',
    #                                    path_base=SYNTH_PATH_APD)

    # init_atlas_org = ptn_atlas.init_atlas_deform_original(atlas)
    # init_atlas_rnd = ptn_atlas.init_atlas_random(atlas.shape, np.max(atlas))
    init_atlas_msc = init_atlas_mosaic(atlas.shape, np.max(atlas))
    # init_encode_rnd = ptn_weigth.initialise_weights_random(len(imgs),
    #                                                        np.max(atlas))

    atlas, weights, _ = bpdl_pipeline(
        imgs, out_prefix='mosaic', init_atlas=init_atlas_msc, max_iter=9, out_dir=path_out
    )
    return atlas, weights


def experiment_iterate(params, iter_params, user_gt):
    if not is_list_like(params['dataset']):
        params['dataset'] = [params['dataset']]

    # tqdm_bar = tqdm.tqdm(total=len(l_params))
    for method, data in [(m, d) for m in params['method'] for d in params['dataset']]:
        params['dataset'] = data
        params['method'] = method
        cls_expt = METHODS.get(method, None)
        assert cls_expt is not None, 'not existing experiment "%s"' % method

        expt = cls_expt(params, time_stamp=params.get('unique', True))
        expt.run(gt=user_gt, iter_params=iter_params)
        del expt
        gc.collect()
        time.sleep(1)


def filter_iterable_params(params):
    _any_special = lambda k: any(x in k for x in SPECIAL_EXPT_PARAMS)
    d_iter = {k: params[k] for k in params if is_iterable(params[k]) and not _any_special(k)}
    return d_iter


def experiments_synthetic(params=SYNTH_PARAMS):
    """ run all experiments

    :param dict params:
    """
    params = parse_params(params, LIST_METHODS)
    logging.info(string_dict(params, desc='PARAMETERS'))

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

    :param dict params:
    """
    params = parse_params(params, LIST_METHODS)
    logging.info(string_dict(params, desc='PARAMETERS'))

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

    :param dict params:
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

    arg_params = parse_params({}, LIST_METHODS)
    if arg_params['debug']:
        logging.getLogger().setLevel(logging.DEBUG)

    main(arg_params)

    logging.info('DONE')
