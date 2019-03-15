"""
experiments with Stat-of-the-Art methods

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import logging
from functools import partial

import numpy as np
import nibabel as nib
from sklearn.decomposition import SparsePCA, FastICA, DictionaryLearning, NMF
from sklearn.cluster import SpectralClustering
from nilearn.decomposition import CanICA, DictLearning
from skimage import segmentation

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bpdl.data_utils import relabel_boundary_background
from bpdl.pattern_atlas import (
    init_atlas_grid, init_atlas_mosaic, init_atlas_random, init_atlas_gauss_watershed_2d,
    init_atlas_otsu_watershed_2d, init_atlas_nmf, init_atlas_fast_ica, init_atlas_sparse_pca,
    init_atlas_dict_learn, init_atlas_deform_original, reconstruct_samples)
from bpdl.dictionary_learning import reset_atlas_background, bpdl_pipeline
from bpdl.pattern_weights import weights_image_atlas_overlap_major
from bpdl.registration import warp2d_images_deformations
from experiments.experiment_general import Experiment, string_dict

NAME_DEFORMS = 'deformations{}.npz'


def estim_atlas_as_argmax(atlas_components, fit_results, force_bg=False,
                          max_bg_ration=0.9):
    """ take pattern index with max value and suppress some background

    :param [ndarray] atlas_components:
    :param [ndarray] fit_results:
    :param float max_bg_ration: reset BG threshold if the background is larger
    :param bool force_bg: force too small components as background
    :return ndarray : np.array<height, width>
    """
    ptn_used = np.sum(np.abs(fit_results), axis=0) > 0
    # filter just used patterns
    atlas_components = atlas_components[ptn_used, :]
    # take the maximal component
    atlas_mean = np.mean(np.abs(atlas_components), axis=0)
    atlas = np.argmax(atlas_components, axis=0)  # + 1

    # filter small values
    if force_bg:
        atlas = reset_atlas_background(atlas, atlas_mean, max_bg_ration)

    assert atlas.shape == atlas_components[0].shape, \
        'dimension mix - atlas: %s atlas_patterns: %s' \
        % (atlas.shape, atlas_components.shape)

    return atlas


def estim_atlas_as_unique_sum(atlas_ptns):
    """

    :param [] atlas_ptns:
    :return ndarray: np.array<height, width>
    """
    atlas = np.sum(np.abs(atlas_ptns), axis=0)
    atlas /= np.max(atlas)
    atlas = np.array(atlas * len(np.unique(atlas)), dtype=np.int)
    return atlas


def binarize_img_reconstruction(img_rct, thr=0.5):
    """ binarise the reconstructed images to be sure again binary

    :param img_rct: np.array<nb_spl, w, h>
    :param float thr:
    :return ndarray:
    """
    img_rct_bin = [None] * img_rct.shape[0]
    for i, im in enumerate(img_rct.tolist()):
        img_rct_bin[i] = np.array(np.asarray(im) > thr, dtype=np.int)
    return img_rct_bin


class ExperimentSpectClust(Experiment):
    """
    State-of-te-Art methods that are based on Spectral Clustering
    """

    def _estimate_atlas_weights(self, images, params):

        imgs_vec = np.nan_to_num(np.array([np.ravel(im) for im in images]))
        bg_offset = 1 if params.get('force_bg', False) else 1

        sc = SpectralClustering(n_clusters=params.get('nb_labels') - bg_offset,
                                affinity='nearest_neighbors',
                                eigen_tol=params.get('tol'),
                                # assign_labels='discretize',
                                degree=0,
                                n_jobs=1)
        sc.fit(imgs_vec.T)
        atlas = sc.labels_.reshape(images[0].shape)

        atlas = relabel_boundary_background(atlas, bg_val=0) + bg_offset
        atlas = segmentation.relabel_sequential(atlas)[0]

        weights = [weights_image_atlas_overlap_major(img, atlas) for img in self._images]
        weights = np.array(weights)

        return atlas, weights, None


def convert_images_nifti(images):
    mask = np.ones(images[0].shape, dtype=np.int8)
    if images[0].ndim == 2:
        images = [np.expand_dims(img, axis=0) for img in images]
        mask = np.expand_dims(mask, axis=0)

    nii_images = [nib.Nifti1Image(img.astype(np.float32), affine=np.eye(4))
                  for img in images]
    nii_mask = nib.Nifti1Image(mask, affine=np.eye(4))
    return nii_images, nii_mask


class ExperimentCanICA(Experiment):
    """
    State-of-te-Art methods that are based on CanICA,
    CanICA is an ICA method for group-level analysis
    """

    def _estimate_atlas_weights(self, images, params):

        nii_images, nii_mask = convert_images_nifti(images)
        bg_offset = 1 if params.get('force_bg', False) else 1

        canica = CanICA(mask=nii_mask,
                        n_components=params.get('nb_labels') - bg_offset,  # - 1
                        mask_strategy='background',
                        threshold='auto',
                        n_init=5,
                        n_jobs=1,
                        verbose=0)
        canica.fit(nii_images)
        components = np.argmax(canica.components_, axis=0) + bg_offset  # + 1
        atlas = components.reshape(images[0].shape)

        atlas = segmentation.relabel_sequential(atlas)[0]

        weights = [weights_image_atlas_overlap_major(img, atlas) for img in self._images]
        weights = np.array(weights)

        return atlas, weights, None


class ExperimentMSDL(Experiment):
    """
    State-of-te-Art methods that are based on MSDL,
    Multi Subject Dictionary Learning
    """

    def _estimate_atlas_weights(self, images, params):

        nii_images, nii_mask = convert_images_nifti(images)
        bg_offset = 1 if params.get('force_bg', False) else 1

        dict_learn = DictLearning(mask=nii_mask,
                                  n_components=params.get('nb_labels') - bg_offset,
                                  mask_strategy='background',
                                  # method='lars',
                                  n_epochs=10,
                                  n_jobs=1,
                                  verbose=0)
        dict_learn.fit(nii_images)
        components = np.argmax(dict_learn.components_, axis=0) + bg_offset
        atlas = components.reshape(images[0].shape)

        # atlas = segmentation.relabel_sequential(atlas)[0]

        weights = [weights_image_atlas_overlap_major(img, atlas) for img in self._images]
        weights = np.array(weights)

        return atlas, weights, None


class ExperimentLinearCombineBase(Experiment):
    """
    State-of-te-Art methods that are based on Linear Combination
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        """ perform the estimation of LinComb and set the estimator,
        results and patterns

        :param ndarray imgs_vec: np.array<nb_imgs, height*width>
        :return (obj, ndarray, ndarray):
        """
        estimator, components, fit_result = None, np.array([0]), np.array([0])
        return estimator, components, fit_result

    def __perform_linear_combination(self, imgs_vec, params):
        """ perform the linear combination and reformulate the outputs

        :param imgs_vec: np.array<nb_imgs, height*width>
        :return (obj, ndarray, ndarray):
        """
        try:
            estimator, components, fit_result = \
                self._estimate_linear_combination(imgs_vec, params)
            atlas_ptns = components.reshape((-1, ) + self._images[0].shape)
            # rct_vec = np.dot(fit_result, components)
        except Exception:
            logging.exception('CRASH in "__perform_linear_combination" in %s',
                              self.__class__.__name__)
            atlas_ptns = np.array([np.zeros(self._images[0].shape)])
            fit_result = np.zeros((len(imgs_vec), 1))
            # rct_vec = np.zeros(imgs_vec.shape)

        atlas = estim_atlas_as_argmax(atlas_ptns, fit_result,
                                      force_bg=params.get('force_bg', False))
        return atlas

    def _estimate_atlas_weights(self, images, params):

        imgs_vec = np.nan_to_num(np.array([im.ravel() for im in images]))

        atlas = self.__perform_linear_combination(imgs_vec, params)
        # atlas = self._estim_atlas_as_unique_sum(atlas_ptns)
        atlas = segmentation.relabel_sequential(atlas)[0]

        weights = [weights_image_atlas_overlap_major(img, atlas) for img in self._images]
        weights = np.array(weights)

        return atlas, weights, None


class ExperimentFastICA(ExperimentLinearCombineBase):
    """ Fast ICA
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = FastICA(n_components=params.get('nb_labels'),
                            max_iter=params.get('max_iter'),
                            algorithm='deflation',
                            tol=params.get('tol'),
                            whiten=True)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.mixing_.T
        return estimator, components, fit_result


class ExperimentSparsePCA(ExperimentLinearCombineBase):
    """ Sparse PCA
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = SparsePCA(n_components=params.get('nb_labels'),
                              max_iter=params.get('max_iter'),
                              tol=params.get('tol'),
                              n_jobs=1)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_
        return estimator, components, fit_result


class ExperimentDictLearn(ExperimentLinearCombineBase):
    """ Dictionary Learning
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = DictionaryLearning(n_components=params.get('nb_labels'),
                                       max_iter=params.get('max_iter'),
                                       fit_algorithm='lars',
                                       transform_algorithm='omp',
                                       split_sign=False,
                                       tol=params.get('tol'),
                                       n_jobs=1)
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_
        return estimator, components, fit_result


class ExperimentNMF(ExperimentLinearCombineBase):
    """ NMF
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    """

    def _estimate_linear_combination(self, imgs_vec, params):
        estimator = NMF(n_components=params.get('nb_labels'),
                        max_iter=params.get('max_iter'),
                        tol=params.get('tol'),
                        init='random')
        fit_result = estimator.fit_transform(imgs_vec)
        components = estimator.components_
        return estimator, components, fit_result


DICT_ATLAS_INIT = {
    'random-grid': init_atlas_grid,
    'random-mosaic': init_atlas_mosaic,
    'random-mosaic-1.5': partial(init_atlas_mosaic, coef=1.5),
    'random-mosaic-2': partial(init_atlas_mosaic, coef=2),
    'random': init_atlas_random,
    'greedy-GaussWS': init_atlas_gauss_watershed_2d,
    'greedy-OtsuWS': init_atlas_otsu_watershed_2d,
    'greedy-OtsuWS-rand': partial(init_atlas_otsu_watershed_2d, bg_type='rand'),
    'GT': None,  # init by Ground Truth, require GT atlas
    'GT-deform': None,  # init by deformed Ground Truth, require GT atlas
    'soa-init-NFM': partial(init_atlas_nmf, nb_iter=5),
    'soa-init-ICA': partial(init_atlas_fast_ica, nb_iter=15),
    'soa-init-PCA': partial(init_atlas_sparse_pca, nb_iter=5),
    'soa-init-DL': partial(init_atlas_dict_learn, nb_iter=5),
    'soa-tune-NFM': partial(init_atlas_nmf, nb_iter=150),
    'soa-tune-ICA': partial(init_atlas_fast_ica, nb_iter=150),
    'soa-tune-PCA': partial(init_atlas_sparse_pca, nb_iter=150),
    'soa-tune-DL': partial(init_atlas_dict_learn, nb_iter=150),
}
LIST_BPDL_PARAMS = ['tol', 'gc_reinit', 'gc_regul', 'max_iter',
                    'ptn_compact', 'overlap_major']


class ExperimentBPDL(Experiment):
    """ the main_train real experiment or our Atlas Learning Pattern Encoding
    """

    def _init_atlas(self, nb_patterns, init_type, imgs):
        """ init atlas according an param

        :param int nb_labels:
        :param str init_type:
        :return ndarray: np.array<w, h>
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
                init_atlas = init_atlas_deform_original(init_atlas)
            init_atlas = init_atlas.astype(int)
        else:
            logging.error('not supported atlas init "%s"', init_type)
            raise NotImplementedError()

        assert np.max(init_atlas) <= nb_patterns, \
            'init. atlas max=%i and nb labels=%i' % \
            (int(np.max(init_atlas)), nb_patterns)
        assert init_atlas.shape == im_size, \
            'init atlas: %r & img size: %r' % (init_atlas.shape, im_size)
        assert init_atlas.dtype == np.int, 'type: %s' % init_atlas.dtype
        if len(np.unique(init_atlas)) == 1:
            logging.warning('atlas init type "%s" failed '
                            'to estimate an atlas', init_type)
        # just to secure the maximal number of patters
        init_atlas[0, 0, ...] = nb_patterns
        return init_atlas

    def _estimate_atlas_weights(self, images, params):
        """ set all params and run the atlas estimation in try mode

        :param ndarray images: np.array<w, h>
        :param {str: ...} params:
        :return (ndarray, ndarray, {}):
        """
        logging.debug(' -> estimate atlas...')
        logging.debug(string_dict(params, desc='PARAMETERS'))
        init_atlas = self._init_atlas(params['nb_labels'] - 1,
                                      params['init_tp'], self._images)
        # prefix = 'expt_{}'.format(p['init_tp'])
        path_out = os.path.join(params['path_exp'],
                                'debug' + params['name_suffix'])

        bpdl_params = {k: params[k] for k in params if k in LIST_BPDL_PARAMS}
        bpdl_params['deform_coef'] = params.get('deform_coef', None)
        atlas, weights, deforms = bpdl_pipeline(images,
                                                init_atlas=init_atlas,
                                                out_dir=path_out,
                                                **bpdl_params)

        assert atlas.max() == weights.shape[1], \
            'atlas max=%i and dict=%i' % (int(atlas.max()), weights.shape[1])
        extras = {'deforms': deforms}

        return atlas, weights, extras

    def _export_extras(self, extras, suffix=''):
        """ export some extra parameters

        :param {} extras: dictionary with extra variables
        """
        if extras is None:  # in case that there are no extras...
            return
        if extras.get('deforms', None) is not None:
            dict_deforms = dict(zip(self._image_names[:len(extras['deforms'])],
                                    extras['deforms']))
            path_npz = os.path.join(self.params.get('path_exp'),
                                    NAME_DEFORMS.format(suffix))
            logging.debug('exporting deformations: %s', path_npz)
            # np.savez(open(path_npz, 'w'), **dict_deforms)
            np.savez_compressed(open(path_npz, 'wb'), **dict_deforms)

    def _evaluate_extras(self, atlas, weights, extras):
        """ some extra evaluation

        :param ndarray atlas: np.array<height, width>
        :param [ndarray] weights: np.array<nb_samples, nb_patterns>
        :param {} extras:
        :return {}:
        """
        stat = {}
        if extras is None:  # in case that there are no extras...
            return {}
        if extras.get('deforms', None) is not None:
            deforms = extras['deforms']
            images_rct = reconstruct_samples(atlas, weights)
            # images = self._images[:len(weights)]
            assert len(images_rct) == len(deforms), \
                'nb reconst. images (%i) and deformations (%i) should match' \
                % (len(images_rct), len(deforms))
            # apply the estimated deformation
            images_rct = warp2d_images_deformations(images_rct, deforms,
                                                           method='nearest',
                                                           inverse=True)
            tag, diff = self._evaluate_reconstruct(images_rct, im_type='input')
            stat['reconst. diff %s deform' % tag] = diff
        return stat
