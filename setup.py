#!/usr/bin/env python

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup

import re
main_py = open('flatcat/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__ = '([^']+)'", main_py))

requires = [
        'morfessor>=2.0.2alpha1'
]

setup(name='Morfessor FlatCat',
      version=metadata['version'],
      author=metadata['author'],
      author_email='morfessor@cis.hut.fi',
      url='http://www.cis.hut.fi/projects/morpho/',
      description='Morfessor FlatCat',
      packages=['flatcat', 'flatcat.tests'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
      ],
      license="BSD",
      scripts=['scripts/flatcat',
               'scripts/flatcat-train',
               'scripts/flatcat-segment',
               'scripts/flatcat-advanced-segment.py',
               'scripts/flatcat-compare-models.py',
               'scripts/flatcat-reformat-list.py',
               'scripts/flatcat-restitch.py',
               ],
      install_requires=requires,
      extras_require={
          'docs': [l.strip() for l in open('docs/build_requirements.txt')]
      }
      )
