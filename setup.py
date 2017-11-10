#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
  name='textsum',
  version='1.0.0',
  long_description='',
  author='Sam Wenke',
  author_email='samwenke@gmail.com',
  description=(''),
  packages=find_packages('.', exclude=[
    'examples*',
    'test*',
    'dist',
    'build',
  ]),
  install_requires=[
    'tensorflow',
    'nltk',
    'numpy',
  ],
  platforms='any',
)
