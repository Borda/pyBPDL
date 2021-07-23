"""
The build/compilations setup

>> pip install -r requirements.txt
>> python setup.py build_ext --inplace
>> python setup.py install

Copyright (C) 2014-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

import os
try:
    from setuptools import setup
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup
    from distutils.command.build_ext import build_ext

import bpdl

TEMP_EGG = '#egg='


def _parse_requirements(file_path):
    with open(file_path) as fp:
        reqs = [r.rstrip() for r in fp.readlines() if not r.startswith('#')]
        # parse egg names if there are paths
        reqs = [r[r.index(TEMP_EGG) + len(TEMP_EGG):] if TEMP_EGG in r else r for r in reqs]
        return reqs


HERE = os.path.abspath(os.path.dirname(__file__))
install_reqs = _parse_requirements(os.path.join(HERE, 'requirements.txt'))

setup(
    name='BPDL',
    version=bpdl.__version__,
    author=bpdl.__author__,
    author_email=bpdl.__author_email__,
    url=bpdl.__homepage__,
    license=bpdl.__license__,
    description='Binary Pattern Dictionary Learning',
    packages=["bpdl"],
    cmdclass={'build_ext': build_ext},
    install_requires=install_reqs,
    long_description=bpdl.__doc__,
    long_description_content_type='text/markdown',
    keywords='image segmentation decomposition atlas encoding benchmark',
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
