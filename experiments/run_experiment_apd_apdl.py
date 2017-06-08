"""
run experiments with Atomic Learning Pattern Encoding

Example run:
>> python run_experiment_apd_apdl.py \
    -in /datagrid/Medical/microscopy/drosophila/synthetic_data/atomicPatternDictionary_v1 \
    -out /datagrid/Medical/microscopy/drosophila/TEMPORARY/experiments_APDL_synth

>> python run_experiment_apd_apdl.py --type real \
    -in /datagrid/Medical/microscopy/drosophila/TEMPORARY/type_1_segm_reg_binary \
    -out /datagrid/Medical/microscopy/drosophila/TEMPORARY/experiments_APDL_real \
    --dataset gene_ssmall

Copyright (C) 2015-2016 Jiri Borovec <jiri.borovec@fel.cvut.cz>
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

import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import tqdm

sys.path.append(os.path.abspath(os.path.join('..','..')))  # Add path to root
from apdl import dataset_utils as gen_data
import apdl.dictionary_learning as dl
import apdl.pattern_disctionary as ptn_dict
import apdl.pattern_weights as ptn_weigth
import experiment_apd as expt_apd


NB_THREADS = expt_apd.NB_THREADS
SYNTH_PARAMS = expt_apd.SYNTH_PARAMS
SYNTH_SUB_DATASETS = expt_apd.SYNTH_SUB_DATASETS_PROBA
SYNTH_PTN_RANGE = expt_apd.SYNTH_PTN_RANGE
REAL_PARAMS = expt_apd.REAL_PARAMS
NB_PATTERNS_REAL = expt_apd.NB_PATTERNS_REAL

DICT_ATLAS_INIT = {
    'msc': ptn_dict.initialise_atlas_mosaic,
    'msc1': partial(ptn_dict.initialise_atlas_mosaic, coef=1.5),
    'msc2': partial(ptn_dict.initialise_atlas_mosaic, coef=2),
    'rnd': ptn_dict.initialise_atlas_random,
    'OWS': ptn_dict.initialise_atlas_otsu_watershed_2d,
    'OWSr': partial(ptn_dict.initialise_atlas_otsu_watershed_2d, bg='rand'),
    'GWS': ptn_dict.initialise_atlas_gauss_watershed_2d,
    'GT': None,  # init by Ground Truth, require GT atlas
    'GTd': None,  # init by deformed Ground Truth, require GT atlas
}

# SIMPLE RUN
INIT_TYPES = ['OWS', 'OWSr', 'GWS']
GRAPHCUT_REGUL = [0., 1e-9, 1e-3]
# COMPLEX RUN
# INIT_TYPES = DICT_ATLAS_INIT.keys()
# GRAPHCUT_REGUL = [0., 0e-12, 1e-9, 1e-6, 1e-3, 1e-1]


def test_simple_show_case():
    """   """
    # implement simple case just with 2 images and 2/3 classes in atlas
    atlas = gen_data.create_simple_atlas()
    # atlas2 = atlas.copy()
    # atlas2[atlas2>2] = 0
    imgs = gen_data.create_sample_images(atlas)
    l_ws = [([1,0,0], [0,1,0], [0,0,1]),
            ([1,0,1], [0,1,1], [0,0,1])]
    for j, ws in enumerate(l_ws):
        plt.figure()
        plt.title('w: {}'.format(repr(ws)))
        gs = gridspec.GridSpec(2, len(imgs) + 2)
        plt.subplot(gs[0, 0]), plt.title('atlas')
        cm = plt.cm.get_cmap('jet', len(np.unique(atlas)))
        plt.imshow(atlas, cmap=cm, interpolation='nearest'), plt.colorbar()
        for i, (img, w) in enumerate(zip(imgs, ws)):
            plt.subplot(gs[0, i + 1]), plt.title('w:{}'.format(w))
            plt.imshow(img, cmap='gray', interpolation='nearest')
        t = time.time()
        uc = dl.compute_relative_penaly_images_weights(imgs, np.array(ws))
        logging.debug('elapsed TIME: %s', repr(time.time() - t))
        res = dl.estimate_atlas_graphcut_general(imgs, np.array(ws), 0.)
        plt.subplot(gs[0, -1]), plt.title('result')
        plt.imshow(res, cmap=cm, interpolation='nearest'), plt.colorbar()
        uc = uc.reshape(atlas.shape+uc.shape[2:])
        # logging.debug(ws)
        for i in range(uc.shape[2]):
            plt.subplot(gs[1, i])
            plt.imshow(uc[:,:,i], vmin=0, vmax=1, interpolation='nearest')
            plt.title('cost lb #{}'.format(i)), plt.colorbar()
        # logging.debug(uc)


def experiment_pipeline_alpe_showcase(path_out):
    """ an simple show case to prove that the particular steps are computed

    :param path_in: str
    :param path_out: str
    :return:
    """
    atlas = gen_data.dataset_compose_atlas(expt_apd.SYNTH_PATH_APD)
    # plt.imshow(atlas)

    path_in = os.path.join(expt_apd.SYNTH_PATH_APD, gen_data.DEFAULT_NAME_DATASET)
    imgs, _ = gen_data.dataset_load_images(path_in)
    # imgs = gen_data.dataset_load_images('datasetBinary_defNoise',
    #                                     path_base=SYNTH_PATH_APD)

    # init_atlas_org = ptn_dict.initialise_atlas_deform_original(atlas)
    # init_atlas_rnd = ptn_dict.initialise_atlas_random(atlas.shape, np.max(atlas))
    init_atlas_msc = ptn_dict.initialise_atlas_mosaic(atlas.shape, np.max(atlas))
    # init_encode_rnd = ptn_weigth.initialise_weights_random(len(imgs), np.max(atlas))

    atlas, w_bins = dl.apdl_pipe_atlas_learning_ptn_weights(
                        imgs, out_prefix='mosaic', init_atlas=init_atlas_msc,
                        max_iter=9, out_dir=path_out)
    return atlas, w_bins


class ExperimentAPDL_base(expt_apd.ExperimentAPD):
    """
    the main_train real experiment or our Atlas Learning Pattern Encoding
    """

    def _init_atlas(self, nb_labels, init_type, imgs):
        """ init atlas according an param

        :param int nb_labels:
        :param str init_type:
        :return: np.array<w, h>
        """
        im_size = self.imgs[0].shape
        logging.debug('INIT atlas - nb labels: %s and type: %s',
                      nb_labels, init_type)
        if init_type.startswith('OWS') or init_type == 'GWS':
            assert init_type in DICT_ATLAS_INIT
            fn_init_atlas = DICT_ATLAS_INIT[init_type]
            init_atlas = fn_init_atlas(imgs, nb_labels)
        elif init_type.startswith('msc') or init_type == 'rnd':
            assert init_type in DICT_ATLAS_INIT
            fn_init_atlas = DICT_ATLAS_INIT[init_type]
            init_atlas = fn_init_atlas(im_size, nb_labels)
        elif init_type == 'GT':
            assert hasattr(self, 'gt_atlas')
            init_atlas = np.remainder(self.gt_atlas, nb_labels)
        elif init_type == 'GTd':
            assert hasattr(self, 'gt_atlas')
            init_atlas = np.remainder(self.gt_atlas, nb_labels)
            init_atlas = ptn_dict.initialise_atlas_deform_original(init_atlas)

        assert init_atlas.max() <= nb_labels, \
            'init. atlas max=%i and nb labels=%i' % \
            (int(init_atlas.max()), nb_labels)
        assert init_atlas.shape == im_size, \
            'init atlas: %s & img size: %s' % \
            (repr(init_atlas.shape) , repr(im_size))
        assert init_atlas.dtype == np.int, 'type: %s' % init_atlas.dtype
        return init_atlas

    def _estimate_atlas(self, v):
        """ set all params and run the atlas estimation in try mode

        :param int i: index of try
        :param init_atlas: np.array<w, h>
        :return: np.array, np.array
        """
        logging.debug(' -> estimate atlas...')
        self.params[self.iter_var_name] = v
        logging.debug('PARAMS: %s', repr(self.params))
        init_atlas = self._init_atlas(self.params['nb_labels'], self.params['init_tp'], self.imgs)
        # prefix = 'expt_{}'.format(p['init_tp'])
        path_out = os.path.join(self.params['path_exp'],
                                'debug_{}_{}'.format(self.iter_var_name, v))
        if isinstance(self.params['nb_samples'], float):
            self.params['nb_samples'] = int(len(self.imgs) * self.params['nb_samples'])
        try:
            atlas, w_bins = dl.apdl_pipe_atlas_learning_ptn_weights(
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

    def _perform_once(self, v):
        """ perform single experiment

        :param v: value
        :return: {str: ...}
        """
        logging.info('perform single experiment...')
        self._estimate_atlas(v)
        logging.debug('atlas of size %s and labels %s', repr(self.atlas.shape),
                      repr(np.unique(self.atlas).tolist()))
        logging.debug('weights of size %s and summing %s',
                      repr(self.w_bins.shape), repr(np.sum(self.w_bins, axis=0)))
        name_posix = '_{}_{}'.format(self.iter_var_name, v)
        self._export_atlas(name_posix)
        self._export_coding(name_posix)
        img_rct = ptn_dict.reconstruct_samples(self.atlas, self.w_bins)
        stat = self._compute_statistic_gt(img_rct)
        stat[self.iter_var_name] = v
        return stat


class ExperimentAPDL(ExperimentAPDL_base, expt_apd.ExperimentAPD_parallel):
    """
    parallel version of APDL
    """
    pass


def experiments_synthetic(params=SYNTH_PARAMS):
    """ run all experiments

    :param {str: any} params:
    """
    arg_params = expt_apd.parse_params(params)
    logging.info('PARAMS: \n%s', '\n'.join(['"{}": \n\t {}'.format(k, v)
                                            for k, v in arg_params.iteritems()]))
    params.update(arg_params)
    params.update({'max_iter': 25})

    l_params = [params]
    if isinstance(params['dataset'], list):
        l_params = expt_apd.extend_list_params(l_params, 'dataset', params['dataset'])
    l_params = expt_apd.extend_list_params(l_params, 'init_tp', INIT_TYPES)
    l_params = expt_apd.extend_list_params(l_params, 'ptn_split', [True, False])
    l_params = expt_apd.extend_list_params(l_params, 'ptn_compact', [True, False])
    l_params = expt_apd.extend_list_params(l_params, 'gc_regul', GRAPHCUT_REGUL)
    ptn_range = SYNTH_PTN_RANGE[os.path.basename(params['path_in'])]
    l_params = expt_apd.extend_list_params(l_params, 'nb_labels', ptn_range)

    logging.debug('list params: %i', len(l_params))

    tqdm_bar = tqdm.tqdm(total=len(l_params))
    for params in l_params:
        try:
            if params['nb_jobs'] > 1:
                expt = ExperimentAPDL(params, params['nb_jobs'])
            else:
                expt = ExperimentAPDL_base(params)
            expt.run(iter_var='case', iter_vals=range(params['nb_runs']))
            # exp.run(iter_var='nb_labels', iter_vals=ptn_range)
        except:
            logging.error(traceback.format_exc())
        del expt
        tqdm_bar.update(1)
        gc.collect(), time.sleep(1)


def experiments_real(params=REAL_PARAMS):
    """ run all experiments

    :param {str: any} params:
    """
    arg_params = expt_apd.parse_params(params)
    logging.info('PARAMS: \n%s', '\n'.join(['"{}": \n\t {}'.format(k, v)
                                            for k, v in arg_params.iteritems()]))
    params.update(arg_params)

    l_params = [copy.deepcopy(params)]
    if isinstance(params['dataset'], list):
        l_params = expt_apd.extend_list_params(l_params, 'dataset', params['dataset'])
    l_params = expt_apd.extend_list_params(l_params, 'init_tp', INIT_TYPES)
    # l_params = expt_apd.extend_list_params(l_params, 'ptn_split', [True, False])
    l_params = expt_apd.extend_list_params(l_params, 'ptn_compact', [True, False])
    l_params = expt_apd.extend_list_params(l_params, 'gc_regul', GRAPHCUT_REGUL)
    # l_params = expt_apd.extend_list_params(l_params, 'nb_labels',
    #                                           [5, 9, 12, 15, 20, 25, 30, 40])
    logging.debug('list params: %i', len(l_params))

    tqdm_bar = tqdm.tqdm(total=len(l_params))
    for params in l_params:
        if params['nb_jobs'] > 1:
            expt = ExperimentAPDL(params, params['nb_jobs'])
        else:
            expt = ExperimentAPDL_base(params)
        # exp.run(gt=False, iter_var='case', iter_values=range(params['nb_runs']))
        expt.run(gt=False, iter_var='nb_labels', iter_vals=NB_PATTERNS_REAL)
        del expt
        tqdm_bar.update(1)
        gc.collect(), time.sleep(1)


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')

    # test_encoding(atlas, imgs, encoding)
    # test_atlasLearning(atlas, imgs, encoding)
    # experiments_test()
    # plt.show()

    arg_params = expt_apd.parse_params(SYNTH_PARAMS)
    if arg_params['type'] == 'synth':
        experiments_synthetic()
    elif arg_params['type'] == 'real':
        experiments_real()

    logging.info('DONE')


if __name__ == "__main__":
    main()
