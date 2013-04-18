#!/usr/bin/env python
"""
Morfessor 2.0 - Python implementation of the Morfessor method
"""
import logging

from .baseline import BaselineModel
from .cmd import main, get_default_argparser
from .exception import MorfessorException, ArgumentException
from .io import MorfessorIO
from .utils import _progress

__all__ = ['MorfessorException', 'ArgumentException', 'MorfessorIO',
           'BaselineModel', 'main', 'get_default_argparser']

__version__ = '2.0.0alpha3'
__author__ = 'Sami Virpioja, Peter Smit'
__author_email__ = "morfessor@cis.hut.fi"
show_progress_bar = True

_logger = logging.getLogger(__name__)


def get_version():
    return __version__
