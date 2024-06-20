# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from setuptools import setup, find_packages, Extension
from pathlib import Path

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []

#########
# Setup #
#########

version_file = Path(__file__).parent / 'xwakes/_version.py'
dd = {}
with open(version_file.absolute(), 'r') as fp:
    exec(fp.read(), dd)
__version__ = dd['__version__']

setup(
    name='xwakes',
    version=__version__,
    description='Wakefield and impedance toolbox for Xsuite',
    long_description='Wakefield and impedance toolbox for Xsuite',
    url='https://xsuite.web.cern.ch/',
    author='G. Iadarola et al.',
    license='Apache 2.0',
    download_url="https://pypi.python.org/pypi/xwakes",
    project_urls={
            "Bug Tracker": "https://github.com/xsuite/xsuite/issues",
            "Source Code": "https://github.com/xsuite/xwakes/",
        },
    packages=find_packages(),
    ext_modules = extensions,
    include_package_data=True,
    install_requires=[
        'numpy>=1.0',
        'scipy',
        'pyyaml',
        ],
    extras_require={
        'tests': ['pytest'],
        },
    )
