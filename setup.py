#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
from numpy.distutils.system_info import get_info

# try to find LAPACK and BLAS
blas_info = get_info('blas_opt')
try:
    # OS X
    blas_include = blas_info['extra_compile_args'][1][2:]
    if not os.path.exists(blas_include):
        # for yosemite
        blas_include = '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers'
        if not os.path.exists(blas_include):
            raise RuntimeError("Could not locate blas libraries")
except KeyError:
    # Linux
    blas_include = "/usr/include/atlas/"

includes = [blas_include, np.get_include()]

extensions = [
    Extension(
        "bayesian_quadrature.linalg_c", ["bayesian_quadrature/linalg_c.pyx"],
        include_dirs=includes,
        libraries=["m", "lapack", "blas"]
    ),

    Extension(
        "bayesian_quadrature.gauss_c", ["bayesian_quadrature/gauss_c.pyx"],
        include_dirs=includes,
        libraries=["m"]
    ),

    Extension(
        "bayesian_quadrature.bq_c", ["bayesian_quadrature/bq_c.pyx"],
        include_dirs=includes,
        libraries=["m"]
    ),

    Extension(
        "bayesian_quadrature.util_c", ["bayesian_quadrature/util_c.pyx"],
        include_dirs=includes,
        libraries=["m"]
    )
]

setup(
    name='bayesian_quadrature',
    version=open('VERSION.txt').read().strip(),
    description='Python library for performing Bayesian Quadrature',
    author='Jessica B. Hamrick',
    author_email='jhamrick@berkeley.edu',
    url='https://github.com/jhamrick/bayesian-quadrature',
    packages=['bayesian_quadrature', 'bayesian_quadrature.tests'],
    ext_modules=cythonize(extensions),
    keywords='bayesian quadrature statistics',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    install_requires=[
        'numpy',
        'scipy',
        'Cython',
        'gaussian_processes'
    ]
)
