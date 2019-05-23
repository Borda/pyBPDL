"""
script that take th csv files with encoding with the proposed atlas
and does the back reconstruction of each image. As sub-step it compute
the reconstruction error to evaluate the parameters and export visualisation

EXAMPLE:
>> python run_reconstruction.py \
    -e ../results/ExperimentBPDL_synth_datasetAPDL_v0_datasetFuzzy_deform \
    --nb_workers 2 --visual

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import time
import argparse
import gc
import glob
import logging
import multiprocessing as mproc
from functools import partial

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.segmentation as sk_segm
from imsegm.utilities.experiments import WrapExecuteSequence, string_dict, load_config_yaml
from imsegm.utilities.data_io import io_imread, update_path

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
from bpdl.data_utils import dataset_load_images, export_image
from bpdl.registration import warp2d_apply_deform_field

NB_THREADS = int(mproc.cpu_count() * .9)
FIGURE_SIZE = (20, 8)
NAME_CONFIG = 'config.yml'
FIELDS_PATH_IMAGES = ['path_in', 'dataset']
BASE_NAME_ATLAS = 'atlas'
BASE_NAME_ENCODE = 'encoding'
BASE_NAME_DEFORM = 'deformations'
BASE_NAME_RECONST = 'reconstruct'
BASE_NAME_VISUAL = 'reconstruct_visual'
CSV_RECONT_DIFF = 'reconstruction_diff.csv'
IMAGE_EXTENSION = '.png'


def parse_arg_params():
    """ parse the input parameters

    :return dict:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--path_expt', type=str, required=True,
                        help='path to the input experiment')
    parser.add_argument('-i', '--path_images', type=str, required=False,
                        help='path to the input images', default=None)
    parser.add_argument('-n', '--name_expt', type=str, required=False,
                        default='*', help='name of experiment')
    parser.add_argument('--nb_workers', type=int, required=False,
                        help='number of processes running in parallel',
                        default=NB_THREADS)
    parser.add_argument('--visual', required=False, action='store_true',
                        help='visualise results', default=False)

    args = vars(parser.parse_args())
    logging.debug('ARG PARAMETERS: \n %r', args)
    args['path_expt'] = update_path(args['path_expt'])
    assert os.path.exists(args['path_expt']), 'missing: %s' % args['path_expt']
    return args


def list_experiments(path, name_pattern):
    path_pattern = os.path.join(path, BASE_NAME_ENCODE + name_pattern)
    l_atlas = [os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(path_pattern)]
    l_expt = [n.replace(BASE_NAME_ENCODE, '') for n in l_atlas]
    return l_expt


def get_path_dataset(path, path_imgs=None):
    if path_imgs is not None and os.path.isdir(path_imgs):
        return path_imgs
    path_config = os.path.join(path, NAME_CONFIG)
    path_imgs = None
    if os.path.isfile(path_config):
        config = load_config_yaml(path_config)
        if all(k in config for k in FIELDS_PATH_IMAGES):
            path_imgs = os.path.join(config[FIELDS_PATH_IMAGES[0]],
                                     config[FIELDS_PATH_IMAGES[1]])
    return path_imgs


def load_images(path_images, names, nb_workers=NB_THREADS):
    if path_images is None or not os.path.isdir(path_images):
        return None
    _name = lambda p: os.path.splitext(os.path.basename(p))[0]
    list_img_paths = glob.glob(os.path.join(path_images, '*'))
    list_img_paths = [p for p in list_img_paths
                      if _name(p) in names]
    logging.debug('found images: %i', len(list_img_paths))
    images, im_names = dataset_load_images(list_img_paths, nb_workers=nb_workers)
    assert all(names == im_names), \
        'image names from weights and loaded images does not match'
    return images


def load_experiment(path_expt, name, path_dataset=None, path_images=None,
                    nb_workers=NB_THREADS):
    path_atlas = os.path.join(path_expt, BASE_NAME_ATLAS + name + '.png')
    atlas = io_imread(path_atlas)
    if (atlas.max() == 255 or atlas.max() == 1.) and len(np.unique(atlas)) < 128:
        # assume it is scratched image
        atlas = sk_segm.relabel_sequential(atlas)[0]

    path_csv = os.path.join(path_expt, BASE_NAME_ENCODE + name + '.csv')
    df_weights = pd.read_csv(path_csv, index_col=None)

    path_npz = os.path.join(path_expt, BASE_NAME_DEFORM + name + '.npz')
    if os.path.isfile(path_npz):
        dict_deforms = dict(np.load(open(path_npz, 'rb')))
        assert len(df_weights) == len(dict_deforms), \
            'unresistant weights (%i) and (%i)' \
            % (len(df_weights), len(dict_deforms))
    else:
        dict_deforms = None

    segms = load_images(path_dataset, df_weights['image'].values, nb_workers)
    images = load_images(path_images, df_weights['image'].values, nb_workers)

    return atlas, df_weights, dict_deforms, segms, images


def draw_reconstruction(atlas, segm_reconst, segm_orig=None, img_rgb=None,
                        fig_size=FIGURE_SIZE):
    """ visualise reconstruction together with the original segmentation

    :param ndarray atlas: np.array<height, width>
    :param ndarray segm_orig: np.array<height, width>
    :param ndarray segm_reconst: np.array<height, width>
    :param ndarray img_rgb: np.array<height, width, 3>
    """
    atlas = atlas.astype(int)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=fig_size)
    ax[0].set_title('original (input)')
    if img_rgb is not None:
        ax[0].imshow(img_rgb, alpha=0.9)
    elif segm_orig is not None:
        segm_orig = segm_orig.astype(np.float32)
        ax[0].imshow(segm_orig, cmap='Greys_r', alpha=0.7)
    ax[0].imshow(atlas, alpha=0.2)
    # ax[0].contour(atlas > 0, levels=np.unique(atlas > 0),
    #               linewidths=2, cmap=plt.cm.jet)
    ax[0].contour(atlas, levels=np.unique(atlas),
                  linewidths=2, cmap=plt.cm.jet)

    ax[1].set_title('reconstructed segmentation')
    atlas_levels = np.arange(atlas.max() + 1, dtype=float) / atlas.max()
    lut_colors = plt.cm.get_cmap('jet')(atlas_levels)
    select_ptns = lut_colors[atlas]
    select_ptns[segm_reconst == 0, :] = 1.
    ax[1].imshow(select_ptns)
    if segm_orig is not None:
        segm = (segm_orig > 0.5)
        ax[1].contour(segm, levels=np.unique(segm), linewidths=2, colors='k')

    ax[2].set_title('selected vs non-selected patterns.')
    active = np.zeros(segm_reconst.shape)
    active[np.logical_and(atlas > 0, segm_reconst > 0.5)] = 1  # green
    active[np.logical_and(atlas > 0, segm_reconst < 0.5)] = -1  # red
    ax[2].imshow(active, alpha=0.7, cmap=plt.cm.RdYlGn)
    ax[2].contour(atlas, levels=np.unique(atlas), linewidths=1, colors='w')
    if segm_orig is not None:
        segm = (segm_orig > 0.5)
        ax[2].contour(segm, levels=np.unique(segm), linewidths=3, colors='k')

    for i in range(3):
        ax[i].axes.get_xaxis().set_ticklabels([])
        ax[i].axes.get_yaxis().set_ticklabels([])
        # ax[i].axis('off')
    return fig


def perform_reconstruction(set_variables, atlas, path_out, path_visu=None):
    name, w_bins, segm, image, deform = set_variables
    if deform is not None:
        atlas = warp2d_apply_deform_field(atlas, deform, method='nearest')

    w_bin_ext = np.array([0] + w_bins.tolist())
    seg_reconst = np.asarray(w_bin_ext)[atlas].astype(atlas.dtype)
    export_image(path_out, seg_reconst, name)

    if path_visu is not None and os.path.isdir(path_visu):
        fig = draw_reconstruction(atlas, seg_reconst, segm, img_rgb=image)
        p_fig = os.path.join(path_visu, name + '.png')
        fig.savefig(p_fig, bbox_inches='tight')
        plt.close(fig)

    diff = np.sum(np.abs(image - seg_reconst)) if image is not None else None
    return name, diff


def process_expt_reconstruction(name_expt, path_expt, path_dataset=None,
                                path_imgs=None, nb_workers=NB_THREADS, visual=False):
    atlas, df_weights, dict_deforms, segms, images = load_experiment(
        path_expt, name_expt, path_dataset, path_imgs, nb_workers)
    df_weights.set_index('image', inplace=True)

    path_out = os.path.join(path_expt, BASE_NAME_RECONST + name_expt)
    if not os.path.isdir(path_out):
        logging.debug('create folder: %s', path_out)
        os.mkdir(path_out)

    if visual:
        path_visu = os.path.join(path_expt, BASE_NAME_VISUAL + name_expt)
        if not os.path.isdir(path_visu):
            logging.debug('create folder: %s', path_visu)
            os.mkdir(path_visu)
    else:
        path_visu = None

    if dict_deforms is not None:
        deforms = [dict_deforms[n] for n in df_weights.index]
    else:
        deforms = [None] * len(df_weights)
    segms = [None] * len(df_weights) if segms is None else segms
    images = [None] * len(df_weights) if images is None else images

    _reconst = partial(perform_reconstruction, atlas=atlas,
                       path_out=path_out, path_visu=path_visu)
    iterate = zip(df_weights.index, df_weights.values, segms, images, deforms)
    list_diffs = []
    for n, diff in WrapExecuteSequence(_reconst, iterate, nb_workers=nb_workers):
        list_diffs.append({'image': n, 'reconstruction diff.': diff})

    df_diff = pd.DataFrame(list_diffs)
    df_diff.set_index('image', inplace=True)
    df_diff.to_csv(os.path.join(path_out, CSV_RECONT_DIFF))


def main(params):
    """ process complete list of experiments """
    logging.info(string_dict(params, desc='PARAMETERS:'))
    list_expt = list_experiments(params['path_expt'], params['name_expt'])
    assert len(list_expt) > 0, 'No experiments found!'
    params['path_dataset'] = get_path_dataset(params['path_expt'])

    for name_expt in tqdm.tqdm(list_expt, desc='Experiments'):
        process_expt_reconstruction(name_expt, path_expt=params['path_expt'],
                                    path_dataset=params['path_dataset'],
                                    path_imgs=params['path_images'],
                                    nb_workers=params['nb_workers'],
                                    visual=params['visual'])
        gc.collect()
        time.sleep(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = parse_arg_params()
    main(params)

    logging.info('DONE')
