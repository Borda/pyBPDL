"""
Extracting the gene activation in case it is separate image channel


>>  python run_extract_fuzzy_activation.py \
    -in "images/ovary_stage-2/image/*.png" \
    -out images/ovary_stage-2/gene

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""


import os
import sys
import glob
import argparse
import logging
import multiprocessing as mproc
from functools import partial

import tqdm
import numpy as np
from skimage import io, morphology, filters
from sklearn.mixture import GaussianMixture
from scipy import ndimage

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_data

NB_THREADS = int(mproc.cpu_count() * .75)
PATH_IN = os.path.join(tl_data.update_path('images/ovary_stage-3/image'), '*.png')
PATH_OUT = tl_data.update_path('images/ovary_stage-3/gene')


def args_parse_params():
    """ create simple arg parser with default values (input, output)

    :param {str: ...} dict_params:
    :return: object argparse<...>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--path_in', type=str, required=True, default=PATH_IN,
                        help='path to the folder with input image dataset')
    parser.add_argument('-out', '--path_out', type=str, required=True, default=PATH_OUT,
                        help='path to the output with experiment results')
    parser.add_argument('--nb_jobs', type=int, required=False,
                        default=NB_THREADS, help='number of parallel processes')

    args = vars(parser.parse_args())
    for k in (k for k in args if k.startswith('path_')):
        p = tl_data.update_path(os.path.dirname(args[k]))
        assert os.path.exists(p), 'missing: %s' % p
        args[k] = os.path.join(p, os.path.basename(args[k]))
    return args


def extract_activation(path_img, path_out):
    name = os.path.splitext(os.path.basename(path_img))[0]
    img = tl_data.io_imread(path_img)
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
    tl_data.export_image(path_out, im_gene, name)


def main(path_pattern_in, path_out, nb_jobs=NB_THREADS):
    if not os.path.isdir(path_out):
        logging.info('create dir: %s', path_out)
        os.mkdir(path_out)

    list_img_paths = glob.glob(path_pattern_in)
    logging.info('found images: %i', len(list_img_paths))
    wrapper_extract = partial(extract_activation, path_out=path_out)

    tqdm_bar = tqdm.tqdm(total=len(list_img_paths))
    mproc_pool = mproc.Pool(nb_jobs)
    for _ in mproc_pool.map(wrapper_extract, list_img_paths):
        tqdm_bar.update()
    mproc_pool.close()
    mproc_pool.join()

    logging.info('DONE')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('running...')
    params = args_parse_params()
    main(params['path_in'], params['path_out'],
         nb_jobs=params['nb_jobs'])
