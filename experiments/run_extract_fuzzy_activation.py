"""
Extracting the gene activation in case it is separate image channel


>>  python run_extract_fuzzy_activation.py \
    -i "../data_images/ovary_stage-2/image/*.png" \
    -o ../data_images/ovary_stage-2/gene

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import glob
import logging
import multiprocessing as mproc
import os
import sys
from functools import partial

import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import numpy as np
from imsegm.utilities.data_io import io_imread, update_path
from imsegm.utilities.experiments import WrapExecuteSequence
# from sklearn.mixture import GaussianMixture
from scipy import ndimage
from skimage import filters

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bpdl.data_utils import export_image
from experiments.run_cut_minimal_images import args_parse_params

NB_WORKERS = int(mproc.cpu_count() * .75)
PARAMS = {
    'path_in': os.path.join(update_path('data_images/ovary_stage-3/image'), '*.png'),
    'path_out': update_path('data_images/ovary_stage-3/gene'),
}


def extract_activation(path_img, path_out):
    name = os.path.splitext(os.path.basename(path_img))[0]
    img = io_imread(path_img)
    mask = img[:, :, 0] > 5

    im_struc = img[:, :, 0]
    im_struc_gauss = ndimage.gaussian_filter(im_struc, 1)
    im_gene = img[:, :, 1]
    im_gene_gauss = ndimage.gaussian_filter(im_gene, 1)

    mask_struc = im_struc_gauss > filters.threshold_otsu(im_struc_gauss[mask])
    ms = np.median(im_struc_gauss[mask_struc])
    mask_gene = im_gene_gauss > filters.threshold_otsu(im_gene_gauss[mask])
    mg = np.median(im_gene_gauss[mask_gene])

    ration_gene = np.sum(mask_gene) / float(np.sum(mask))
    coef = (ms / mg * 2.5) if ration_gene > 0.3 else (ms / mg * 5)
    im_mean = np.max(np.array([im_gene_gauss + (im_struc_gauss / coef)]), axis=0)

    otsu = filters.threshold_otsu(im_mean[mask])
    im_gene = im_mean.copy()
    im_gene[im_gene < otsu] = 0

    # gmm = GaussianMixture(n_components=3, n_init=10)
    # data = np.array([im_gene_gauss[mask].ravel(), im_struc_gauss[mask]]).T
    # gmm.fit(data)
    # id_max = np.argmax(gmm.means_[:, 0])
    # gm_mean = gmm.means_[id_max, 0]
    # gm_std = np.sqrt(gmm.covariances_[id_max, 0, 0])
    # im_gene[im_gene_gauss < (gm_mean - gm_std)] = 0

    # p_out = os.path.join(path_out, name)
    export_image(path_out, im_gene, name)


def main(path_pattern_in, path_out, nb_workers=NB_WORKERS):
    assert os.path.isdir(os.path.dirname(path_pattern_in)), 'missing: %s' % path_pattern_in
    assert os.path.isdir(os.path.dirname(path_out)), 'missing: %s' % os.path.dirname(path_out)

    if not os.path.isdir(path_out):
        logging.info('create dir: %s', path_out)
        os.mkdir(path_out)

    list_img_paths = glob.glob(path_pattern_in)
    logging.info('found images: %i', len(list_img_paths))

    _wrapper_extract = partial(extract_activation, path_out=path_out)
    list(WrapExecuteSequence(_wrapper_extract, list_img_paths, nb_workers))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = args_parse_params(PARAMS)
    main(params['path_in'], params['path_out'], nb_workers=params['nb_workers'])

    logging.info('DONE')
