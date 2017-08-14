"""
The build/compilations setup

>> python setup.py build_ext --inplace

Copyright (C) 2014-2017 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


setup(
    name='BPDL',
    version='0.1',
    author='Jiri Borovec',
    author_email='jiri.borovec@fel.cvut.cz',
    url='https://github.com/Borda/pyBPDL',
    license='BSD 3-clause',
    description='Binary Pattern Dictionary Learning',
    long_description="""
Image processing package for unsupervised pattern extraction.
""",
    cmdclass = {'build_ext': build_ext},
    packages=["bpdl"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
