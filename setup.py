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


def _parse_requirements(file_path):
    with open(file_path) as fp:
        reqs = [r.rstrip() for r in fp.readlines() if not r.startswith('#')]
        return reqs


HERE = os.path.abspath(os.path.dirname(__file__))
install_reqs = _parse_requirements(os.path.join(HERE, 'requirements.txt'))

setup(
    name='BPDL',
    version='0.2.0',
    author='Jiri Borovec',
    author_email='jiri.borovec@fel.cvut.cz',
    url='https://borda.github.com/pyBPDL',
    license='BSD 3-clause',

    description='Binary Pattern Dictionary Learning',
    packages=["bpdl"],
    cmdclass={'build_ext': build_ext},
    install_requires=install_reqs,

    long_description="""
Image processing package for unsupervised pattern extraction and atlas estimation.
""",
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
