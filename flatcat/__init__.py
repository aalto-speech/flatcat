#!/usr/bin/env python
"""
Morfessor 2.0 FlatCat - Python implementation of
the FlatCat variant of the Morfessor method
"""
import logging


__all__ = ['MorfessorException', 'ArgumentException', 'FlatcatIO',
           'FlatcatModel', 'flatcat_main', 'get_flatcat_argparser',
           'MorphUsageProperties', 'HeuristicPostprocessor']

__version__ = '1.0.7'
__author__ = 'Stig-Arne Gronroos'
__author_email__ = "morfessor@cis.hut.fi"

show_progress_bar = True

_logger = logging.getLogger(__name__)


def get_version(numeric=False):
    if numeric:
        return __version__
    return 'FlatCat {}'.format(__version__)


# The public api imports need to be at the end of the file,
# so that the package global names are available to the modules
# when they are imported.

from .flatcat import FlatcatModel, AbstractSegmenter
from .flatcat import FlatcatAnnotatedCorpusEncoding
from .categorizationscheme import MorphUsageProperties, HeuristicPostprocessor
from .categorizationscheme import WORD_BOUNDARY, CategorizedMorph
from .cmd import flatcat_main, get_flatcat_argparser
from .exception import MorfessorException, ArgumentException
from .io import FlatcatIO
from .utils import _progress
