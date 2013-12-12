#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "bayesian_quadrature.bq_c", ["bayesian_quadrature/bq_c.pyx"],
        include_dirs=[np.get_include()],
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
