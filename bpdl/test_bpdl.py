"""
Testing for some integrations...

Copyright (C) 2015-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import time
import logging

import numpy as np
import matplotlib.pyplot as plt

import bpdl.data_utils as tl_data
import bpdl.pattern_atlas as ptn_atlas
import bpdl.dictionary_learning as dict_learn


def simple_show_case(atlas, imgs, ws):
    """ simple experiment

    >>> atlas = tl_data.create_simple_atlas()
    >>> imgs = tl_data.create_sample_images(atlas)
    >>> ws=([1, 0, 0], [0, 1, 1], [0, 0, 1])
    >>> res, fig = simple_show_case(atlas, imgs, ws)
    >>> res.astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> fig  # doctest: +ELLIPSIS
    <...>
    """
    # implement simple case just with 2 images and 2/3 classes in atlas
    fig, axarr = plt.subplots(2, len(imgs) + 2)

    plt.title('w: {}'.format(repr(ws)))
    axarr[0, 0].set_title('atlas')
    cm = plt.cm.get_cmap('jet', len(np.unique(atlas)))
    im = axarr[0, 0].imshow(atlas, cmap=cm, interpolation='nearest')
    fig.colorbar(im, ax=axarr[0, 0])

    for i, (img, w) in enumerate(zip(imgs, ws)):
        axarr[0, i + 1].set_title('w:{}'.format(w))
        axarr[0, i + 1].imshow(img, cmap='gray', interpolation='nearest')

    t = time.time()
    uc = ptn_atlas.compute_relative_penalty_images_weights(imgs, np.array(ws))
    logging.info('elapsed TIME: %s', repr(time.time() - t))
    res = dict_learn.estimate_atlas_graphcut_general(imgs, np.array(ws), 0.)

    axarr[0, -1].set_title('result')
    im = axarr[0, -1].imshow(res, cmap=cm, interpolation='nearest')
    fig.colorbar(im, ax=axarr[0, -1])
    uc = uc.reshape(atlas.shape + uc.shape[2:])

    # logging.debug(ws)
    for i in range(uc.shape[2]):
        axarr[1, i].set_title('cost lb #{}'.format(i))
        im = axarr[1, i].imshow(uc[:, :, i], vmin=0, vmax=1,
                                interpolation='nearest')
        fig.colorbar(im, ax=axarr[1, i])
    # logging.debug(uc)
    return res, fig
