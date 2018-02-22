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
import bpdl.pattern_atlas as ptn_dict
import bpdl.pattern_weights as ptn_weight
import experiments.experiment_apdl as expt_apd
import experiments.run_experiments_bpdl as run_bpdl


class ExperimentLinearCombineBase(expt_apd.ExperimentAPD):
    """
    State-of-te-Art methods that are based on Linear Combination
    """

    def _estimate_linear_combination(self, imgs_vec):
        """ perform the estimation of LinComb and set the estimator,
        results and patterns

        :param imgs_vec: np.array<nb_imgs, height*width>
        :return:
        """
        pass

    def _perform_linear_combination(self, imgs_vec):
        """ perform the linear combination and reformulate the outputs

        :param imgs_vec: np.array<nb_imgs, height*width>
        :return:
        """
        try:
            self._estimate_linear_combination(imgs_vec)
            logging.debug('fitting parameters: %s',
                          repr(self.estimator.get_params()))
            logging.debug('number of iteration: %i', self.estimator.n_iter_)

            atlas_ptns = self.components.reshape((-1, ) + self.imgs[0].shape)
            rct_vec = np.dot(self.fit_result, self.components)
        except:
            logging.warning('crash in "_perform_linear_combination" in %s',
                            self.__class__.__name__)
            logging.warning(traceback.format_exc())
            atlas_ptns = np.array([np.zeros(self.imgs[0].shape)])
            rct_vec = np.zeros(imgs_vec.shape)
        return atlas_ptns, rct_vec

    def estim_atlas_as_argmax(self, atlas_components, bg_threshold=0.1):
        """ take max pattern with max value

        :param [] atlas_components:
        :return: np.array<height, width>
        """
        # in case the method crash before and the attribute doe not exst
        if not hasattr(self, 'fit_result'):
            atlas = np.zeros(atlas_components[0].shape)
            return atlas
        ptn_used = np.sum(np.abs(self.fit_result), axis=0) > 0
        # filter just used patterns
        atlas_components = atlas_components[ptn_used, :]
        # take the maximal component
        atlas_mean = np.mean(np.abs(atlas_components), axis=0)
        atlas = np.argmax(atlas_components, axis=0) # + 1
        # filter small values
        atlas[atlas_mean < bg_threshold] = 0
        assert atlas.shape == atlas_components[0].shape, \
            'dimension mix - atlas: %s atlas_ptns: %s' \
            % (atlas.shape, atlas_components.shape)
        return atlas

    def estim_atlas_as_unique_sum(self, atlas_ptns):
        """

        :param [] atlas_ptns:
        :return: np.array<height, width>
        """
        atlas = np.sum(np.abs(atlas_ptns), axis=0)
        atlas /= np.max(atlas)
        atlas = np.array(atlas * len(np.unique(atlas)), dtype=np.int)
        return atlas

    def _convert_patterns_to_atlas(self, atlas_ptns):
        """ convert the estimated patterns into a reasonable atlas

        :param atlas_ptns: np.array<nb_patterns, w*h>
        :return: np.array<height, width>
        """
        atlas = self.estim_atlas_as_argmax(atlas_ptns)
        # atlas = self.estim_atlas_as_unique_sum(atlas_ptns)
        self.atlas = segmentation.relabel_sequential(atlas)[0]

    def _binarize_img_reconstruction(self, img_rct, thr=0.5):
        """ binarise the reconstructed images to be sure again binary

        :param img_rct: np.array<nb_spl, w, h>
        :param float thr:
        :return:
        """
        img_rct_bin = [None] * img_rct.shape[0]
        for i, im in enumerate(img_rct.tolist()):
            img_rct_bin[i] = np.array(np.asarray(im) > thr, dtype=np.int)
        return img_rct_bin

    def _perform_once(self, d_params):
        """ perform one experiment

        :param dict d_params: update of used parameters
        :return:
        """
        self.params.update(d_params)
        name_posix = '_' + '_'.join('{}={}'.format(k, d_params[k])
                                    for k in sorted(d_params) if k != 'param_idx')
        if isinstance(self.params['nb_samples'], float):
            self.params['nb_samples'] = int(len(self.imgs) * self.params['nb_samples'])
        imgs_vec = np.array([np.ravel(im) for im in self.imgs[:self.params['nb_samples']]])
        atlas_ptns, rct_vec = self._perform_linear_combination(imgs_vec)
        # img_rct = rct_vec.reshape(np.asarray(self.imgs[:self.params['nb_samples']]).shape)
        self._convert_patterns_to_atlas(atlas_ptns)
        self._export_atlas(name_posix)
        w_bins = [ptn_weight.weights_image_atlas_overlap_major(img, self.atlas)
                  for img in self.imgs[:self.params['nb_samples']]]
        self.w_bins = np.array(w_bins)
        self._export_coding(name_posix)
        img_rct = ptn_dict.reconstruct_samples(self.atlas, self.w_bins)
        stat = self._compute_statistic_gt(img_rct)
        stat.update(d_params)
        return stat


class ExperimentFastICA_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    """

    def _estimate_linear_combination(self, imgs_vec):
        self.estimator = FastICA(n_components=self.params.get('nb_labels'),
                                 max_iter=self.params.get('max_iter'),
                                 algorithm='deflation',
                                 whiten=True)
        self.fit_result = self.estimator.fit_transform(imgs_vec)
        self.components = self.estimator.mixing_.T


class ExperimentSparsePCA_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
    """

    def _estimate_linear_combination(self, imgs_vec):
        self.estimator = SparsePCA(n_components=self.params.get('nb_labels'),
                                   max_iter=self.params.get('max_iter'),
                                   n_jobs=1)
        self.fit_result = self.estimator.fit_transform(imgs_vec)
        self.components = self.estimator.components_


class ExperimentDictLearn_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    """

    def _estimate_linear_combination(self, imgs_vec):
        self.estimator = DictionaryLearning(n_components=self.params.get('nb_labels'),
                                            max_iter=self.params.get('max_iter'),
                                            fit_algorithm='lars',
                                            transform_algorithm='omp',
                                            split_sign=False,
                                            n_jobs=1)
        self.fit_result = self.estimator.fit_transform(imgs_vec)
        self.components = self.estimator.components_


class ExperimentNMF_base(ExperimentLinearCombineBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    """

    def _estimate_linear_combination(self, imgs_vec):
        self.estimator = NMF(n_components=self.params.get('nb_labels'),
                             max_iter=self.params.get('max_iter'),
                             init='random')
        self.fit_result = self.estimator.fit_transform(imgs_vec)
        self.components = self.estimator.components_


class ExperimentFastICA(ExperimentFastICA_base, expt_apd.ExperimentAPD_parallel):
    pass


class ExperimentSparsePCA(ExperimentSparsePCA_base, expt_apd.ExperimentAPD_parallel):
    pass


class ExperimentDictLearn(ExperimentDictLearn_base, expt_apd.ExperimentAPD_parallel):
    pass


class ExperimentNMF(ExperimentNMF_base, expt_apd.ExperimentAPD_parallel):
    pass


# standard multiprocessing version
METHODS = {
    'PCA': ExperimentFastICA,
    'ICA': ExperimentSparsePCA,
    'DL': ExperimentDictLearn,
    'NMF': ExperimentNMF,
    'BPDL': run_bpdl.ExperimentBPDL,
}

# working jut in single thread for pasiisng to image data to prtial jobs
METHODS_BASE = {
    'PCA': ExperimentFastICA_base,
    'ICA': ExperimentSparsePCA_base,
    'DL': ExperimentDictLearn_base,
    'NMF': ExperimentNMF_base,
    'BPDL': run_bpdl.ExperimentBPDL_base,
}

SYNTH_PARAMS = expt_apd.SYNTH_PARAMS
SYNTH_PARAMS.update({
    'dataset': ['datasetProb_raw'],
    'method': list(METHODS.keys()),
    'max_iter': 25,  # 250, 25
})

REAL_PARAMS = expt_apd.REAL_PARAMS
REAL_PARAMS.update({
    'method': list(METHODS.keys()),
    'max_iter': 25,  # 250, 25
})


def experiment_iterate(params, iter_params, user_gt):
    if not expt_apd.is_list_like(params['dataset']):
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
        assert cls_expt is not None

        expt = cls_expt(params)
        expt.run(gt=user_gt, iter_params=iter_params)
        del expt
        gc.collect(), time.sleep(1)


def experiments_synthetic(params=SYNTH_PARAMS):
    """ run all experiments

    :param {str: value} params:
    """
    params = expt_apd.parse_params(params)
    logging.info(expt_apd.string_dict(params, desc='PARAMETERS'))

    iter_params = {
        'nb_labels': params['nb_labels']
    }

    experiment_iterate(params, iter_params, user_gt=True)


def experiments_real(params=REAL_PARAMS):
    """ run all experiments

    :param {str: value} params:
    """
    params = expt_apd.parse_params(params)
    logging.info(expt_apd.string_dict(params, desc='PARAMETERS'))

    iter_params = {
        'nb_labels': params['nb_labels']
    }

    experiment_iterate(params, iter_params, user_gt=False)


def main(arg_params):
    # swap according dataset type
    if arg_params['type'] == 'synth':
        experiments_synthetic()
    elif arg_params['type'] == 'real':
        experiments_real()
    # plt.show()


if __name__ == "__main__":
    """ main entry point """
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.DEBUG)

    arg_params = expt_apd.parse_params({})
    main(arg_params)

    logging.info('DONE')
