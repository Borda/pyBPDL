"""
Using the try/except import since the init is called in setup  to get pkg info
before satisfying install requirements

"""

import os
import sys

try:
    import joblib
    import matplotlib
    import numpy as np
    import pandas as pd

    # in case you are running on machine without display, e.g. server
    if not os.environ.get('DISPLAY', '') and matplotlib.rcParams['backend'] != 'agg':
        print('No display found. Using non-interactive Agg backend.')
        # https://matplotlib.org/faq/usage_faq.html
        matplotlib.use('Agg')

    # parse the numpy versions
    np_version = [int(i) for i in np.version.full_version.split('.')]
    # comparing strings does not work for version lower 1.10
    if np_version >= [1, 14]:
        np.set_printoptions(legacy='1.13')

    # default display size was changed in pandas v0.23
    pd.set_option('display.max_columns', 20)

    # ModuleNotFoundError: No module named 'sklearn.externals.joblib'
    sys.modules['sklearn.externals.joblib'] = joblib

except ImportError:
    import traceback
    traceback.print_exc()

__version__ = '0.2.3'
__author__ = 'Jiri Borovec'
__author_email__ = 'jiri.borovec@fel.cvut.cz'
__license__ = 'BSD 3-clause'
__homepage__ = 'https://borda.github.io/pyBPDL'
__copyright__ = 'Copyright (c) 2014-2019, %s.' % __author__
__doc__ = 'BPDL - Binary pattern Dictionary Learning'
__long_doc__ = "# %s" % __doc__ + """

The package contain Binary pattern Dictionary Learning (BPDL) which is image processing tool
 for unsupervised pattern extraction and atlas estimation. Moreover the project/repository
 contains comparisons with State-of-the-Art decomposition methods applied to image domain.
 The package also includes useful tools for dataset handling and around real microscopy images.

## Main features
* implementation of BPDL package
* using fuzzy segmentation as inputs
* deformation via daemon registration
* experimental setting & synthetic dataset
* comparison with NMF, SPCS, ICA, CanICA, MSDL, etc.
* visualisations and notebook samples

## References
Borovec J., Kybic J. (2016) Binary Pattern Dictionary Learning for Gene Expression Representation
 in Drosophila Imaginal Discs. In: Computer Vision - ACCV 2016 Workshops. Lecture Notes in Computer
 Science, vol 10117, Springer. DOI: 10.1007/978-3-319-54427-4_40.
"""
