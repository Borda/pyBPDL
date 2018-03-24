"""
run experiments with Stat-of-the-art methods

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
import traceback
from functools import partial

import numpy as np
from sklearn.decomposition import SparsePCA, FastICA, DictionaryLearning, NMF
from skimage import segmentation

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.pattern_atlas as ptn_dict
import bpdl.dictionary_learning as dl
import bpdl.pattern_weights as ptn_weight
import experiments.experiment_general as expt_gen


def estim_atlas_as_argmax(atlas_components, fit_result, bg_threshold=0.1):
    """ take max pattern with max value

    :param [ndarray] atlas_components:
    :param float bg_threshold: setting the backround
    :return: np.array<height, width>
    """
    ptn_used = np.sum(np.abs(fit_result), axis=0) > 0
    # filter just used patterns
    atlas_components = atlas_components[ptn_used, :]
    # take the maximal component
    atlas_mean = np.mean(np.abs(atlas_components), axis=0)
    atlas = np.argmax(atlas_components, axis=0)  # + 1
    # filter small values
    atlas[atlas_mean < bg_threshold] = 0

    assert atlas.shape == atlas_components[0].shape, \
        'dimension mix - atlas: %s atlas_ptns: %s' \
        % (atlas.shape, atlas_components.shape)

    return atlas


def estim_atlas_as_unique_sum(atlas_ptns):
    """

    :param [] atlas_ptns:
    :return: np.array<height, width>
    """
    atlas = np.sum(np.abs(atlas_ptns), axis=0)
    atlas /= np.max(atlas)
    atlas = np.array(atlas * len(np.unique(atlas)), dtype=np.int)
    return atlas


def binarize_img_reconstruction(img_rct, thr=0.5):
    """ binarise the reconstructed images to be sure again binary

    :param img_rct: np.array<nb_spl, w, h>
    :param float thr:
    :return:
    """
    img_rct_bin = [None] * img_rct.shape[0]
    for i, im in enumerate(img_rct.tolist()):
        img_rct_bin[i] = np.array(np.asarray(im) > thr, dtype=np.int)
    return img_rct_bin


class ExperimentLinearCombineBase(expt_gen.Experiment):
    """
    State-of-te-Art methods that are based on Linear Combination
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        """ perform the estimation of LinComb and set the estimator,
        results and patterns

        :param ndarray imgs_vec: np.array<nb_imgs, height*width>
        :return:
        """
        estimator, components, fit_result = None, np.array([0]), np.array([0])
        return estimator, components, fit_result

    def __perform_linear_combination(self, imgs_vec, params):
        """ perform the linear combination and reformulate the outputs

        :param imgs_vec: np.array<nb_imgs, height*width>
        :return:
        """
        try:
            estimator, components, fit_result = \
                self._estimate_linear_combination(imgs_vec, params)
            atlas_ptns = components.reshape((-1, ) + self._images[0].shape)
            # rct_vec = np.dot(fit_result, components)
        except Exception:
            logging.warning('CRASH in "__perform_linear_combination" in %s',
                            self.__class__.__name__)
            logging.warning(traceback.format_exc())
            atlas_ptns = np.array([np.zeros(self._images[0].shape)])
            fit_result = np.zeros((len(imgs_vec), 1))
            # rct_vec = np.zeros(imgs_vec.shape)
        atlas = estim_atlas_as_argmax(atlas_ptns, fit_result)
        return atlas

    def _estimate_atlas_weights(self, images, params):

        imgs_vec = np.nan_to_num(np.array([np.ravel(im) for im in images]))

        atlas = self.__perform_linear_combination(imgs_vec, params)
        # atlas = self._estim_atlas_as_unique_sum(atlas_ptns)
        atlas = segmentation.relabel_sequential(atlas)[0]

        weights = [ptn_weight.weights_image_atlas_overlap_major(img, atlas)
                   for img in self._images]
        weights = np.array(weights)

        return atlas, weights, None


class ExperimentFastICA_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = FastICA(n_components=params.get('nb_labels'),
                            max_iter=params.get('max_iter'),
                            algorithm='deflation',
                            whiten=True)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.mixing_.T
        return estimator, components, fit_result


class ExperimentFastICA(ExperimentFastICA_base, expt_gen.ExperimentParallel):
    pass


class ExperimentSparsePCA_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = SparsePCA(n_components=params.get('nb_labels'),
                              max_iter=params.get('max_iter'),
                              n_jobs=1)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_
        return estimator, components, fit_result


class ExperimentSparsePCA(ExperimentSparsePCA_base, expt_gen.ExperimentParallel):
    pass


class ExperimentDictLearn_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = DictionaryLearning(n_components=params.get('nb_labels'),
                                       max_iter=params.get('max_iter'),
                                       fit_algorithm='lars',
                                       transform_algorithm='omp',
                                       split_sign=False,
                                       n_jobs=1)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_
        return estimator, components, fit_result


class ExperimentDictLearn(ExperimentDictLearn_base, expt_gen.ExperimentParallel):
    pass


class ExperimentNMF_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = NMF(n_components=params.get('nb_labels'),
                             max_iter=params.get('max_iter'),
                             init='random')
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_
        return estimator, components, fit_result


class ExperimentNMF(ExperimentNMF_base, expt_gen.ExperimentParallel):
    pass


DICT_ATLAS_INIT = {
    'random-grid': ptn_dict.init_atlas_grid,
    'random-mosaic': ptn_dict.init_atlas_mosaic,
    'random-mosaic-1.5': partial(ptn_dict.init_atlas_mosaic, coef=1.5),
    'random-mosaic-2': partial(ptn_dict.init_atlas_mosaic, coef=2),
    'random': ptn_dict.init_atlas_random,
    'greedy-GausWS': ptn_dict.init_atlas_gauss_watershed_2d,
    'greedy-OtsuWS': ptn_dict.init_atlas_otsu_watershed_2d,
    'greedy-OtsuWS-rand': partial(ptn_dict.init_atlas_otsu_watershed_2d,
                                  bg_type='rand'),
    'GT': None,  # init by Ground Truth, require GT atlas
    'GT-deform': None,  # init by deformed Ground Truth, require GT atlas
    'soa-init-NFM': partial(ptn_dict.init_atlas_nmf, nb_iter=5),
    'soa-init-ICA': partial(ptn_dict.init_atlas_fast_ica, nb_iter=15),
    'soa-init-PCA': partial(ptn_dict.init_atlas_sparse_pca, nb_iter=5),
    'soa-init-DL': partial(ptn_dict.init_atlas_dict_learn, nb_iter=5),
    'soa-tune-NFM': partial(ptn_dict.init_atlas_nmf, nb_iter=150),
    'soa-tune-ICA': partial(ptn_dict.init_atlas_fast_ica, nb_iter=150),
    'soa-tune-PCA': partial(ptn_dict.init_atlas_sparse_pca, nb_iter=150),
    'soa-tune-DL': partial(ptn_dict.init_atlas_dict_learn, nb_iter=150),
}


class ExperimentBPDL_base(expt_gen.Experiment):
    """ the main_train real experiment or our Atlas Learning Pattern Encoding
    """

    def _init_atlas(self, nb_patterns, init_type, imgs):
        """ init atlas according an param

        :param int nb_labels:
        :param str init_type:
        :return: np.array<w, h>
        """
        im_size = self._images[0].shape
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
            assert hasattr(self, '_gt_atlas'), 'missing GT atlas'
            init_atlas = np.remainder(self._gt_atlas, nb_patterns)
            if init_type == 'GT-deform':
                init_atlas = ptn_dict.init_atlas_deform_original(init_atlas)
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

    def _estimate_atlas_weights(self, images, params):
        """ set all params and run the atlas estimation in try mode

        :param int i: index of try
        :param init_atlas: np.array<w, h>
        :return: np.array, np.array
        """
        logging.debug(' -> estimate atlas...')
        logging.debug(expt_gen.string_dict(params, desc='PARAMETERS'))
        init_atlas = self._init_atlas(params['nb_labels'] - 1,
                                      params['init_tp'], self._images)
        # prefix = 'expt_{}'.format(p['init_tp'])
        path_out = os.path.join(params['path_exp'],
                                'debug' + params['name_suffix'])

        atlas, weights, deforms = dl.bpdl_pipeline(
                                    images,
                                    init_atlas=init_atlas,
                                    tol=params['tol'],
                                    gc_reinit=params['gc_reinit'],
                                    gc_coef=params['gc_regul'],
                                    max_iter=params['max_iter'],
                                    ptn_split=params['ptn_split'],
                                    ptn_compact=params['ptn_compact'],
                                    overlap_major=params['overlap_mj'],
                                    deform_coef=params.get('deform_coef', None),
                                    out_dir=path_out)  # , out_prefix=prefix

        assert atlas.max() == weights.shape[1], \
            'atlas max=%i and dict=%i' % (int(atlas.max()), weights.shape[1])
        extras = {'deforms': deforms}

        return atlas, weights, extras

    def _export_extras(self, extras, suffix=''):
        """ export some extra parameters

        :param {} extras: dictionary with extra variables
        """
        # todo, export deform
        pass

    def _evaluate_extras(self, atlas, weights, extras):
        """ some extra evaluation

        :param ndarray atlas: np.array<height, width>
        :param [ndarray] weights: np.array<nb_samples, nb_patterns>
        :param {} extras:
        :return {}:
        """
        # todo, evaluate deform
        return {}


class ExperimentBPDL(ExperimentBPDL_base, expt_gen.ExperimentParallel):
    """
    parallel version of APDL
    """
    pass
