"""
walk over all experiment folders and ...

EXAMPLES:
>> python run_recompute_experiments_result.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
import sys
import glob
import json
import argparse
import traceback
import logging
import gc, time
import multiprocessing as mproc
from functools import partial

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from PIL import Image
from skimage.segmentation import relabel_sequential

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_data
import bpdl.metric_similarity as tl_metric
import experiments.experiment_apdl as expt_apd
import experiments.run_dataset_generate as r_data
import experiments.run_parse_experiments_result as r_parse

NAME_INPUT_CONFIG = r_data.NAME_CONFIG
NAME_INPUT_RESULT = 'results.csv'
NAME_OUTPUT_RESULT = 'results_NEW.csv'
SUB_PATH_GT_ATLAS = os.path.join('dictionary', 'atlas.png')
NAME_PATTERN_ATLAS = 'atlas%s.png'
NB_THREADS = int(mproc.cpu_count() * 0.9)
FIGURE_SIZE = 6

PARAMS = {
    'name_config': NAME_INPUT_CONFIG,
    'name_results': NAME_INPUT_RESULT,
}


def load_atlas(path_atlas):
    assert os.path.exists(path_atlas), 'missing: %s' % path_atlas
    img = Image.open(path_atlas)
    # atlas = np.array(img.convert('L'))
    atlas = np.array(img)
    atlas = relabel_sequential(atlas)[0]
    atlas -= np.min(atlas)
    return atlas


def export_atlas_overlap(atlas_gt, atlas, path_out_img, fig_size=FIGURE_SIZE):
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    ax.imshow(atlas, alpha=0.5, interpolation='nearest', cmap=plt.cm.jet)
    ax.contour(atlas_gt, levels=np.unique(atlas_gt), linewidth=2, cmap=plt.cm.jet)

    ax.axis('off')
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    path_out_img = os.path.splitext(path_out_img)[0] + '-overlap.png'
    fig.savefig(path_out_img)
    plt.close(fig)


def export_atlas_both(atlas_gt, atlas, path_out_img, fig_size=FIGURE_SIZE):
    fig, axarr = plt.subplots(ncols=2, figsize=(2 * fig_size, fig_size))
    max_label = max([np.max(atlas_gt), np.max(atlas)])
    discrete_cmap = plt.cm.get_cmap('jet', max_label + 1)

    axarr[0].set_title('GroundTruth atlas')
    im = axarr[0].imshow(atlas_gt, interpolation='nearest', cmap=discrete_cmap,
                         vmin=0, vmax=max_label)
    fig.colorbar(im, ax=axarr[0], ticks=np.arange(max_label + 1))

    axarr[1].set_title('estimated atlas')
    im = axarr[1].imshow(atlas, interpolation='nearest', cmap=discrete_cmap,
                         vmin=0, vmax=max_label)
    fig.colorbar(im, ax=axarr[1], ticks=np.arange(max_label + 1))

    for ax in axarr:
        ax.axis('off')
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.02, hspace=0.02)

    path_out_img = os.path.splitext(path_out_img)[0] + '-both.png'
    fig.savefig(path_out_img)
    plt.close(fig)


def parse_experiment_folder(path_expt, params):
    """ parse experiment folder, get configuration and results

    :param str path_expt: path to experiment folder
    :param {str: any} params:
    """
    assert os.path.isdir(path_expt), 'missing EXPERIMENT: %s' % path_expt

    path_config = os.path.join(path_expt, params['name_config'])
    assert path_config.endswith('.json'), '%s' % path_config
    assert os.path.exists(path_config), 'missing config: %s' % path_config
    dict_info = json.load(open(path_config, 'r'))
    logging.debug(' -> loaded params: %s', repr(dict_info.keys()))

    path_results = os.path.join(path_expt, params['name_results'])
    assert path_results.endswith('.csv'), '%s' % path_results
    assert os.path.exists(path_results), 'missing result: %s' % path_results
    df_res = pd.DataFrame().from_csv(path_results)
    index_name = df_res.index.name
    df_res[index_name] = df_res.index

    # load the GT atlas
    path_atlas = os.path.join(dict_info['path_in'],
                              tl_data.DIR_NAME_DICTIONARY)
    atlas_gt = tl_data.dataset_compose_atlas(path_atlas)
    path_atlas_gt = os.path.join(dict_info['path_in'], SUB_PATH_GT_ATLAS)
    atlas_name = str(os.path.splitext(os.path.basename(path_atlas_gt))[0])
    tl_data.export_image(os.path.dirname(path_atlas_gt), atlas_gt, atlas_name)
    plt.imsave(os.path.splitext(path_atlas_gt)[0] + '_visual.png', atlas_gt)

    df_res_new = pd.DataFrame()
    for idx, row in df_res.iterrows():
        dict_row = dict(row)
        if not isinstance(idx, str) and idx - int(idx) == 0:
            idx = int(idx)
        atlas_name = NAME_PATTERN_ATLAS % dict_row['name_suffix']
        atlas = load_atlas(os.path.join(path_expt, atlas_name))
        # try to find the mest match among patterns / labels
        atlas = tl_metric.relabel_max_overlap_unique(atlas_gt, atlas)
        # recompute the similarity measure
        dict_measure = tl_metric.compute_classif_metrics(atlas_gt.ravel(), atlas.ravel())
        dict_measure = {'atlas %s' % k: dict_measure[k] for k in dict_measure}
        dict_row.update(dict_measure)
        df_res_new = df_res_new.append(dict_row, ignore_index=True)
        # visualise atlas
        atlas_name_visu = os.path.splitext(atlas_name)[0] + '_visual.png'
        path_visu = os.path.join(path_expt, atlas_name_visu)
        export_atlas_overlap(atlas_gt, atlas, path_visu)
        export_atlas_both(atlas_gt, atlas, path_visu)

    df_res_new.set_index([index_name], inplace=True)
    path_results = os.path.join(path_expt, NAME_OUTPUT_RESULT)
    df_res_new.to_csv(path_results)
    # just to let it releases memory
    gc.collect(), time.sleep(1)


def try_parse_folder(path_expt, params):
    """ just a wrapper to cover all paring as try  """
    try:
        parse_experiment_folder(path_expt, params)
    except:
        logging.warning(traceback.format_exc())


def parse_experiments(params):
    """ with specific input parameters wal over result folder and parse it

    :param params: {str: ...}
    """
    logging.info('running recompute Experiments results')
    logging.info(expt_apd.string_dict(params, desc='ARGUMENTS:'))
    assert os.path.exists(params['path']), 'missing "%s"' % params['path']

    path_dirs = [p for p in glob.glob(os.path.join(params['path'], '*'))
                 if os.path.isdir(p)]
    logging.info('found experiments: %i', len(path_dirs))
    wrapper_parse_folder = partial(try_parse_folder,
                                   params=params)
    tqdm_bar = tqdm.tqdm(total=len(path_dirs))

    if params['nb_jobs'] > 1:
        logging.debug('perform_sequence in %i threads', params['nb_jobs'])
        mproc_pool = mproc.Pool(params['nb_jobs'])
        for _ in mproc_pool.imap_unordered(wrapper_parse_folder,
                                                   path_dirs):
            tqdm_bar.update()
        mproc_pool.close()
        mproc_pool.join()
    else:
        for path_expt in path_dirs:
            logging.debug('folder %s', path_expt)
            try_parse_folder(path_expt, params)
            tqdm_bar.update()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('running...')

    params = r_parse.parse_arg_params(PARAMS)
    parse_experiments(params)

    logging.info('DONE')