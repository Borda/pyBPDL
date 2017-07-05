"""
run experiments with Atomic Learning Pattern Encoding

Example run:
>> python run_experiment_apd_bpdl.py \
    -in /datagrid/Medical/microscopy/drosophila/synthetic_data/apdDataset_v1 \
    -out /datagrid/Medical/microscopy/drosophila/TEMPORARY/experiments_APDL_synth

>> python run_experiment_apd_bpdl.py --type real \
    -in /datagrid/Medical/microscopy/drosophila/TEMPORARY/type_1_segm_reg_binary \
    -out /datagrid/Medical/microscopy/drosophila/TEMPORARY/experiments_APDL_real \
    --dataset gene_ssmall

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

# to suppress all visual, has to be on the beginning

import copy
import gc
import logging
import os
import sys
import time
import traceback
from functools import partial

import tqdm
import numpy as np

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_data
import bpdl.dictionary_learning as dl
import bpdl.pattern_atlas as ptn_dict
import experiments.experiment_apdl as expt_apd


NB_THREADS = expt_apd.NB_THREADS
DEFAULT_PARAMS = expt_apd.DEFAULT_PARAMS
SYNTH_PARAMS = expt_apd.SYNTH_PARAMS
SYNTH_PARAMS.update({
    'method': 'BPDL',
    'dataset': ['datasetProb_raw'],
    'max_iter': 25,  # 250, 25
})

REAL_PARAMS = expt_apd.REAL_PARAMS
REAL_PARAMS.update({
    'method': 'BPDL',
    'max_iter': 25,  # 250, 25
})

DICT_ATLAS_INIT = {
    'random-grid': ptn_dict.initialise_atlas_grid,
    'random-mosaic': ptn_dict.initialise_atlas_mosaic,
    'random-mosaic-1.5': partial(ptn_dict.initialise_atlas_mosaic, coef=1.5),
    'random-mosaic-2': partial(ptn_dict.initialise_atlas_mosaic, coef=2),
    'random': ptn_dict.initialise_atlas_random,
    'greedy-GausWS': ptn_dict.initialise_atlas_gauss_watershed_2d,
    'greedy-OtsuWS': ptn_dict.initialise_atlas_otsu_watershed_2d,
    'greedy-OtsuWS-rand': partial(ptn_dict.initialise_atlas_otsu_watershed_2d,
                                  bg_type='rand'),
    'GT': None,  # init by Ground Truth, require GT atlas
    'GT-deform': None,  # init by deformed Ground Truth, require GT atlas
    'soa-init-NFM': partial(ptn_dict.initialise_atlas_nmf, nb_iter=5),
    'soa-init-ICA': partial(ptn_dict.initialise_atlas_fast_ica, nb_iter=15),
    'soa-init-PCA': partial(ptn_dict.initialise_atlas_sparse_pca, nb_iter=5),
    'soa-init-DL': partial(ptn_dict.initialise_atlas_dict_learn, nb_iter=5),
    'soa-tune-NFM': partial(ptn_dict.initialise_atlas_nmf, nb_iter=150),
    'soa-tune-ICA': partial(ptn_dict.initialise_atlas_fast_ica, nb_iter=150),
    'soa-tune-PCA': partial(ptn_dict.initialise_atlas_sparse_pca, nb_iter=150),
    'soa-tune-DL': partial(ptn_dict.initialise_atlas_dict_learn, nb_iter=150),
}

# SIMPLE RUN
# INIT_TYPES = ['OWS', 'OWSr', 'GWS']
INIT_TYPES_ALL = sorted(DICT_ATLAS_INIT.keys())
INIT_TYPES_NORM = [t for t in INIT_TYPES_ALL if 'tune' not in t]
INIT_TYPES_NORM_REAL = [t for t in INIT_TYPES_NORM if not t.startswith('GT')]
GRAPHCUT_REGUL = [0., 1e-9, 1e-3]
# COMPLEX RUN
# GRAPHCUT_REGUL = [0., 0e-12, 1e-9, 1e-6, 1e-3, 1e-1]


def experiment_pipeline_alpe_showcase(path_out):
    """ an simple show case to prove that the particular steps are computed

    :param path_in: str
    :param path_out: str
    :return:
    """
    path_atlas = os.path.join(expt_apd.SYNTH_PATH_APD,
                              tl_data.DIR_NAME_DICTIONARY)
    atlas = tl_data.dataset_compose_atlas(path_atlas)
    # plt.imshow(atlas)

    path_in = os.path.join(expt_apd.SYNTH_PATH_APD, tl_data.DEFAULT_NAME_DATASET)
    path_imgs = tl_data.find_images(path_in)
    imgs, _ = tl_data.dataset_load_images(path_imgs)
    # imgs = tl_data.dataset_load_images('datasetBinary_defNoise',
    #                                       path_base=SYNTH_PATH_APD)

    # init_atlas_org = ptn_dict.initialise_atlas_deform_original(atlas)
    # init_atlas_rnd = ptn_dict.initialise_atlas_random(atlas.shape, np.max(atlas))
    init_atlas_msc = ptn_dict.initialise_atlas_mosaic(atlas.shape, np.max(atlas))
    # init_encode_rnd = ptn_weigth.initialise_weights_random(len(imgs), np.max(atlas))

    atlas, w_bins = dl.bpdl_pipe_atlas_learning_ptn_weights(
                        imgs, out_prefix='mosaic', init_atlas=init_atlas_msc,
                        max_iter=9, out_dir=path_out)
    return atlas, w_bins


class ExperimentBPDL_base(expt_apd.ExperimentAPD):
    """ the main_train real experiment or our Atlas Learning Pattern Encoding
    """

    def _init_atlas(self, nb_patterns, init_type, imgs):
        """ init atlas according an param

        :param int nb_labels:
        :param str init_type:
        :return: np.array<w, h>
        """
        im_size = self.imgs[0].shape
        logging.debug('INIT atlas - nb labels: %s and type: %s',
                      nb_patterns, init_type)
        if init_type.startswith('greedy'):
            assert init_type in DICT_ATLAS_INIT
            fn_init_atlas = DICT_ATLAS_INIT[init_type]
            init_atlas = fn_init_atlas(imgs, nb_patterns)
        elif init_type.startswith('random'):
            assert init_type in DICT_ATLAS_INIT
            fn_init_atlas = DICT_ATLAS_INIT[init_type]
            init_atlas = fn_init_atlas(im_size, nb_patterns)
        elif init_type.startswith('soa'):
            assert init_type in DICT_ATLAS_INIT
            fn_init_atlas = DICT_ATLAS_INIT[init_type]
            init_atlas = fn_init_atlas(imgs, nb_patterns)
        elif init_type.startswith('GT'):
            assert hasattr(self, 'gt_atlas')
            init_atlas = np.remainder(self.gt_atlas, nb_patterns)
            if init_type == 'GT-deform':
                init_atlas = ptn_dict.initialise_atlas_deform_original(init_atlas)
            init_atlas = init_atlas.astype(int)
        else:
            logging.error('not supported atlas init "%s"', init_type)
            raise NotImplemented()

        assert np.max(init_atlas) <= nb_patterns, \
            'init. atlas max=%i and nb labels=%i' % \
            (int(np.max(init_atlas)), nb_patterns)
        assert init_atlas.shape == im_size, \
            'init atlas: %s & img size: %s' % \
            (repr(init_atlas.shape), repr(im_size))
        assert init_atlas.dtype == np.int, 'type: %s' % init_atlas.dtype
        return init_atlas

    def _estimate_atlas(self, posix=''):
        """ set all params and run the atlas estimation in try mode

        :param int i: index of try
        :param init_atlas: np.array<w, h>
        :return: np.array, np.array
        """
        logging.debug(' -> estimate atlas...')
        logging.debug(expt_apd.string_dict(self.params, desc='PARAMETERS'))
        init_atlas = self._init_atlas(self.params['nb_labels'] - 1,
                                      self.params['init_tp'], self.imgs)
        # prefix = 'expt_{}'.format(p['init_tp'])
        path_out = os.path.join(self.params['path_exp'], 'debug' + posix)
        if isinstance(self.params['nb_samples'], float):
            self.params['nb_samples'] = int(len(self.imgs) * self.params['nb_samples'])
        try:
            atlas, w_bins = dl.bpdl_pipe_atlas_learning_ptn_weights(
                                        self.imgs[:self.params['nb_samples']],
                                        init_atlas=init_atlas,
                                        tol=self.params['tol'],
                                        gc_reinit=self.params['gc_reinit'],
                                        gc_coef=self.params['gc_regul'],
                                        max_iter=self.params['max_iter'],
                                        ptn_split=self.params['ptn_split'],
                                        ptn_compact=self.params['ptn_compact'],
                                        overlap_major=self.params['overlap_mj'],
                                        out_dir=path_out)  # , out_prefix=prefix
        except:
            logging.error('FAILED, no atlas estimated!')
            logging.error(traceback.format_exc())
            atlas = np.zeros_like(self.imgs[0])
            w_bins = np.zeros((len(self.imgs), 0))
        assert atlas.max() == w_bins.shape[1], \
            'atlas max=%i and dict=%i' % (int(atlas.max()), w_bins.shape[1])
        self.atlas = atlas
        self.w_bins = w_bins

    def _perform_once(self, d_params):
        """ perform single experiment

        :param dict d_params: update of used parameters
        :return: {str: ...}
        """
        self.params.update(d_params)
        name_posix = '_' + '_'.join('{}={}'.format(k, d_params[k])
                                    for k in sorted(d_params) if k != 'param_idx')
        logging.info('perform single experiment...')
        self._estimate_atlas(posix=name_posix)
        logging.debug('atlas of size %s and labels %s', repr(self.atlas.shape),
                      repr(np.unique(self.atlas).tolist()))
        logging.debug('weights of size %s and summing %s',
                      repr(self.w_bins.shape), repr(np.sum(self.w_bins, axis=0)))
        self._export_atlas(name_posix)
        self._export_coding(name_posix)
        img_rct = ptn_dict.reconstruct_samples(self.atlas, self.w_bins)
        stat = self._compute_statistic_gt(img_rct)
        stat.update(d_params)
        return stat


class ExperimentBPDL(ExperimentBPDL_base, expt_apd.ExperimentAPD_parallel):
    """
    parallel version of APDL
    """
    pass


def experiment_iterate(params, iter_params, user_gt):
    if not expt_apd.is_list_like(params['dataset']):
        params['dataset'] = [params['dataset']]

    for dataset in params['dataset']:
        params['dataset'] = dataset
        if params['nb_jobs'] > 1:
            expt = ExperimentBPDL(params, params['nb_jobs'])
        else:
            expt = ExperimentBPDL_base(params)
        expt.run(gt=user_gt, iter_params=iter_params)
        del expt
        gc.collect(), time.sleep(1)
    # expt.run(iter_var='nb_labels', iter_vals=ptn_range)


def experiments_synthetic(params=SYNTH_PARAMS):
    """ run all experiments

    :param {str: any} params:
    """
    params = expt_apd.parse_params(params)
    logging.info(expt_apd.string_dict(params, desc='PARAMETERS'))
    # params.update({'max_iter': 25})

    iter_params = {
        'init_tp': INIT_TYPES_NORM,
        # 'ptn_split': [True, False],
        'ptn_compact': [True, False],
        'gc_regul': GRAPHCUT_REGUL,
        'nb_labels': params['nb_labels']
    }

    experiment_iterate(params, iter_params, user_gt=True)


def experiments_real(params=REAL_PARAMS):
    """ run all experiments

    :param {str: any} params:
    """
    params = expt_apd.parse_params(params)
    logging.info(expt_apd.string_dict(params, desc='PARAMETERS'))

    iter_params = {
        'init_tp': INIT_TYPES_NORM_REAL,
        # 'ptn_compact': [True, False],
        # 'gc_regul': GRAPHCUT_REGUL,
        'nb_labels': params['nb_labels']
    }

    experiment_iterate(params, iter_params, user_gt=False)


def main(arg_params):
    # test_encoding(atlas, imgs, encoding)
    # test_atlasLearning(atlas, imgs, encoding)
    # experiments_test()
    # plt.show()
    if arg_params['type'] == 'synth':
        experiments_synthetic()
    elif arg_params['type'] == 'real':
        experiments_real()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')

    arg_params = expt_apd.parse_params({})

    main(arg_params)

    logging.info('DONE')
