import os, sys, glob
import logging

import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pylab as plt
# from collections import Counter
from IPython.html import widgets
from IPython.display import display
from IPython.html.widgets import ToggleButtonsWidget as w_tb
from IPython.html.widgets import IntSliderWidget as w_is
from IPython.html.widgets import DropdownWidget as w_s

sys.path += [os.path.abspath('.'), os.path.abspath('..')]  # Add path to root
import bpdl.dataset_utils as tl_data

PATH_DATA_SYNTH = tl_data.update_path('images')
SYNTH_DATASET = 'syntheticDataset_vX'
PATH_DATA_SYNTH = '/mnt/30C0201EC01FE8BC/TEMP'
SYNTH_DATASET = 'atomicPatternDictionary_v0'
DEFAULT_PATH = os.path.join(PATH_DATA_SYNTH, SYNTH_DATASET)
SYNTH_DATASETS_BINARY = ['datasetBinary_raw',
                         'datasetBinary_deform',
                         'datasetBinary_noise',
                         'datasetBinary_defNoise']
SYNTH_DATASETS_PROB = ['datasetProb_raw',
                       'datasetProb_deform',
                       'datasetProb_noise',
                       'datasetProb_defNoise']
DEFAULT_IMG_POSIX = '.png'
TEMP_ATLAS_NAME = 'APDL_expt_msc_atlas_iter_'
DEFAULT_APDL_GRAPHS = ('atlas_ARS', 'reconstruct_diff', 'time')


def load_dataset(path_dataset):
    list_imgs = []
    paths_imgs = glob.glob(os.path.join(path_dataset, '*' + DEFAULT_IMG_POSIX))
    reporting = [int((i + 1) * len(paths_imgs) / 5.) for i in range(5)]
    for path_im in paths_imgs:
        im = io.imread(path_im)
        list_imgs.append(im / float(np.max(im)))
        if i in reporting:
            logging.debug(' -> loaded \t{:3.0%}'.format(i/float(len(paths_imgs))))
    return list_imgs


def show_sample_data_as_imgs(imgs, im_shape, nb_rows=5, nb_cols=3, bool_clr=False):
    nb_rows = min(nb_rows, np.ceil(len(imgs) / float(nb_cols)))
    plt.figure(figsize=(3 * nb_cols, 2.5 * nb_rows))
    nb_spls = min(nb_rows * nb_cols, len(imgs))
    for i in range(int(nb_spls)):
        im = imgs[i, :].reshape(im_shape)
        # u_px = Counter(im)
        unique_px = sorted(np.unique(im), reverse=True)
        plt.subplot(nb_rows, nb_cols, i + 1)
        if bool_clr:
            plt.imshow(im, interpolation='nearest'), plt.colorbar()
            # plt.title('; '.join(['{:.2f}'.format(float(v)) for v in unique_px[:2]]))
        else:
            plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
            # plt.title(repr(unique_px[:2]))
        plt.axis('off')
    # plt.tight_layout()


def bpdl_w_update_range(w_params, uq_range):
    for n in w_params:
        if n not in uq_range or len(uq_range[n]) == 0:
            w_params[n].visible = False
            w_params[n].max = 0
        else:
            w_params[n].visible = True
            if type(uq_range[n]) == int:
                w_params[n].min = min(uq_range[n])
                w_params[n].max = max(uq_range[n])
                w_params[n].value = w_params[n].min
            else:
                w_params[n].min = 0
                w_params[n].max = len(uq_range[n]) - 1


def bpdl_w_update_param(w_params, uq_params):
    for n in w_params:
        if n not in uq_params or len(uq_params[n]) == 0:
            w_params[n].visible = False
        else:
            try:
                float(uq_params[n][0])
                vals = ['v_'+str(v) for v in uq_params[n]]
            except:
                vals = uq_params[n]
            w_params[n].options = dict(zip(vals, uq_params[n]))
            if len(uq_params[n]) == 1:
                w_params[n].visible = False
            else:
                w_params[n].visible = True


def round_range_val(df_data, params, name):
    vals = filter_df_unique(df_data, params, [name])[name]
    if str(params[name]) not in vals and len(vals) > 0:
        val = params[name]
        dist = [abs(int(v) - int(val)) for v in vals]
        params[name] = vals[dist.index(min(dist))]
        logging.warning('no experiment with %s lbs -> nearest %s', val, params[name])
    return params


def bpdl_interact_results_iter_samples(df_data, dist_vars, tp):
    w_source = {n: w_s(options=dist_vars[n], description=n, )
              for n in ['dataset', 'sub_dataset']}
    w_param = {n: w_tb(options=dist_vars[n], description=n)
               for n in ['gc_reinit', 'init_tp', 'ptn_split', 'gc_regul']}
    w_range = {n: w_is(min=0, max=0, description=n)
               for n in ['nb_lbs', 'samples']}
    def colect_params():
        params = {n: w_source[n].value for n in w_source}
        params.update({n: w_param[n].value for n in w_param})
        params.update({n: w_range[n].value for n in w_range})
        round_range_val(df_data, params, 'nb_lbs')
        return params
    def show_results(**kwargs):
        params = colect_params()
        print 'params:', params
        # disable options with single value
        dict_source = {n: w_source[n].value for n in w_source}
        uq_param = filter_df_unique(df_data, dict_source, w_param)
        bpdl_w_update_param(w_param, uq_param)
        # collect all data from interact
        uq_range = filter_df_unique(df_data, params, ['nb_lbs'])
        # find desired experiment results
        filter_param = {n: params[n] for n in params if n not in ['samples']}
        df_filter, uq_range['samples'] = find_experiment(df_data, filter_param)
        bpdl_w_update_range(w_range, uq_range)
        bpdl_show_results(df_filter, uq_range['samples'], params['samples'], tp)
    # show the interact
    widgets.interact(show_results, w=w_source['dataset'])
    widgets.interact(show_results, w=w_source['sub_dataset'])
    widgets.interact(show_results, **w_param)
    widgets.interact(show_results, w=w_range['nb_lbs'])
    widgets.interact(show_results, w=w_range['samples'])


def bpdl_show_results(df_sel, path_imgs, idx=0, tp='gt', fig_size=(10, 5)):
    if len(df_sel) == 0:
        return
    res = df_sel.iloc[0]
    path_atlas_gt = os.path.join(res['in_path'], res['dataset'], 'dictionary', 'atlas.png')
    im_atlas = io.imread(path_imgs[idx])
    if os.path.exists(path_atlas_gt):
        fig, axarr = plt.subplots(1, 2, figsize=fig_size)
        im_atlas_gt = io.imread(path_atlas_gt)
        axarr[0].imshow(im_atlas_gt, interpolation='nearest')
        axarr[0].set_title('GT atlas')
        axarr[1].imshow(im_atlas)
        axarr[1].set_title('result'), axarr[1].axis('off')
    else:
        if tp == 'gt':
            logging.warning('no GT for tp=%s and path: "%s"', tp, path_atlas_gt)
        fig = plt.figure(figsize=fig_size)
        fig.gca().imshow(im_atlas, interpolation='nearest')
        fig.gca().set_title('result'), fig.gca().axis('off')
    display()


def filter_df_unique(df_data, dict_filter, var_unique):
    q = ' and '.join(['({} == "{}")'.format(n, dict_filter[n]) for n in dict_filter
                      if n in df_data.columns and n not in var_unique])
    df_filter = df_data.query(q, engine='python')
    dict_vars = {n: np.unique(df_filter[n]).tolist()
                 for n in var_unique if n in df_data.columns}
    return dict_vars


def find_experiment(df_data, params):
    q = ' and '.join(['({} == "{}")'.format(n, params[n])
                      for n in params if n in df_data.columns])
    df_filter = df_data.query(q, engine='python')
    path_atlas = []
    if len(df_filter) == 0:
        logging.warning('no such experiment')
    elif len(df_filter) > 0:
        assert len(df_filter) > 0
        if len(df_filter) > 1:
            logging.warning('multiple suitable experiments')
        assert os.path.exists(df_filter['res_path'].values[-1])
        path_res = os.path.join(df_filter['res_path'].values[-1], '*')
        path_dirs = [p for p in glob.glob(path_res) if os.path.isdir(p)]
        name_pattern = TEMP_ATLAS_NAME + '*' + DEFAULT_IMG_POSIX
        for path_dir in path_dirs:
            path_imgs = glob.glob(os.path.join(path_dir, name_pattern))
            path_atlas.append(sorted(path_imgs)[-1])
    return df_filter, path_atlas


def extend_df(df_encode, df_main):
    if not 'gene_id' in df_encode.columns:
        df_encode = df_encode.merge(df_main, left_index=True, right_on='image',
                                    how='inner')
    return df_encode


def aggregate_encoding(df_encode, column='gene_id', func=np.mean):
    df_result = pd.DataFrame()
    list_ptns = [c for c in df_encode if c.startswith('ptn ')]
    grouped = df_encode.groupby(column)
    for value, df_group in grouped:
        data = df_group[list_ptns].values
        result = np.apply_along_axis(func, axis=0, arr=data)
        dict_res = dict(zip(list_ptns, result.tolist()))
        dict_res.update({column: value, 'count': len(df_group)})
        df_result = df_result.append(dict_res, ignore_index=True)
    df_result = df_result.set_index(column)
    return df_result


def plot_bpdl_graph_results(df_res, n_group, n_curve, iter_var='nb_labels',
                            l_graphs=DEFAULT_APDL_GRAPHS, figsize=(8, 3)):
    for v, df_group in df_res.groupby(n_group):
        clrs = plt.cm.jet(np.linspace(0, 1, len(df_group)))
        fig, axarr = plt.subplots(len(l_graphs), 1,
                                  figsize=(figsize[0], figsize[1] * len(l_graphs)))
        fig.suptitle('{}'.format(v), fontsize=16)
        for i, col in enumerate(l_graphs):
            for j, (idx, row) in enumerate(df_group.iterrows()):
                if len(row[iter_var]) == 0: continue
                axarr[i].plot(row[iter_var], row[col], label=row[n_curve], color=clrs[j])
                axarr[i].set_xlim([min(row[iter_var]), max(row[iter_var])])
            axarr[i].set_xlabel(iter_var), axarr[i].set_ylabel(col)
            axarr[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            axarr[i].grid()


def filter_df_results_4_plotting(df_select, iter_var='nb_labels',
                                 n_group='version', n_class='init_tp',
                                 cols=DEFAULT_APDL_GRAPHS):
    dict_samples = {}
    logging.info('number of selected: %i', len(df_select))
    df_res = pd.DataFrame()
    # print 'version', df_select['version'].unique().tolist()
    for v, df_gr0 in df_select.groupby(n_group):
        dict_samples[v] = {}
        # print 'method', df_select['method'].unique().tolist()
        for v1, df_gr1 in df_gr0.groupby(n_class):
            nb_samples = []
            # print iter_var, df_select[iter_var].unique().tolist()
            d = {n_group: v, n_class: v1}
            d_vals = {col: [] for col in cols + [iter_var]}
            for v2, df_gr2 in df_gr1.groupby(iter_var):
                d_vals[iter_var].append(v2)
                for col in cols:
                    d_vals[col].append(np.mean(df_gr2[col]))
                nb_samples.append(len(df_gr2[iter_var]))
            d.update(d_vals)
            dict_samples[v][v1] = nb_samples
            df_res = df_res.append(d, ignore_index=True)
    # df_res = df_res.set_index('class')
    logging.info('number of rows: %i columns: %s', len(df_res),
                 repr(df_res.columns.tolist()))
    logging.info('over samples: %s', repr(dict_samples))
    return df_res, dict_samples
