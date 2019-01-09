#!/usr/bin/env python

from distutils.core import setup

setup(name='Rate Limiting alias classifier',
      version='1.0',
      description='Rate limiting classifier',
      author='Kevin Vermeulen',
      author_email='kevin.vermeulen@sorbonne-universite.fr',
      url='',
      packages=['Algorithms', 'Classification', 'Cluster', 'Cpp', 'Data', 'Files', 'Validation'],
      install_requires=[
          'tensorflow',
          'seaborn',
          'sklearn',

      ],
     )