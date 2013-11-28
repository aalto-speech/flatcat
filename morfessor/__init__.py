#!/usr/bin/env python
"""
Morfessor 2.0 - Python implementation of the Morfessor method
"""
import logging


__all__ = ['MorfessorException', 'ArgumentException', 'MorfessorIO',
           'BaselineModel', 'FlatcatModel', 'FlatcatIO',
           'MorphUsageProperties', 'main', 'get_default_argparser']

__version__ = '2.0.0alpha3'
__author__ = 'Sami Virpioja, Peter Smit, Stig-Arne Gronroos'
__author_email__ = "morfessor@cis.hut.fi"

show_progress_bar = True

_logger = logging.getLogger(__name__)


def get_version():
    return __version__

# The public api imports need to be at the end of the file,
# so that the package global names are available to the modules
# when they are imported.

from .baseline import BaselineModel
from .categorizationscheme import MorphUsageProperties
from .cmd import main, get_default_argparser
from .exception import MorfessorException, ArgumentException
from .flatcat import FlatcatModel
from .io import MorfessorIO, FlatcatIO
from .utils import _progress
