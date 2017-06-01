#!/usr/bin/env python

import re
from setuptools import setup

main_py = open('flatcat/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__ = '([^']+)'", main_py))

requires = [
    'morfessor>=2.0.2alpha1'
]

setup(name='Morfessor FlatCat',
      version=metadata['version'],
      author=metadata['author'],
      author_email='morfessor@cis.hut.fi',
      url='https://github.com/aalto-speech/flatcat',
      description='Morfessor FlatCat',
      keywords='semi-supervised morphological segmentation',
      packages=['flatcat', 'flatcat.tests'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
      ],
      license="BSD",
      scripts=['scripts/flatcat',
               'scripts/flatcat-train',
               'scripts/flatcat-segment',
               'scripts/flatcat-evaluate',
               'scripts/flatcat-advanced-segment.py',
               #'scripts/flatcat-compare-models.py',
               #'scripts/flatcat-reformat-list.py',
               'scripts/flatcat-restitch.py',
              ],
      install_requires=requires,
      extras_require={
          'docs': [l.strip() for l in open('docs/build_requirements.txt')]
      }
     )
