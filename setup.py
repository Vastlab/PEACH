#!/usr/bin/env python

from distutils.core import setup

setup(name='PEACH',
      version='1.0',
      description='Provable Extreme-value Agglomerative Clustering-threshold Estimation',
      author='Chunchun Li & tboult ',
      author_email='cli@vast.uccs.edu',
      url='https://github.com/Vastlab/PEACH/',
      py_modules = ['PEACH', 'tau_flann_pytorch','weibull','merge','evaluate']
     )
