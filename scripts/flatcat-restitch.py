#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import locale
import re
import string
import sys

import flatcat
from flatcat.exception import ArgumentException
from flatcat import utils


PY3 = sys.version_info.major == 3

LICENSE = """
Copyright (c) 2015, Stig-Arne Gronroos
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

def get_argparser():
    parser = argparse.ArgumentParser(
        prog='flatcat-advanced-segment',
        description="""
Morfessor FlatCat advanced segmentation and reformatting
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    add_arg = parser.add_argument

    add_arg('infile', metavar='<infile>',
            help='The input file. The type will be sniffed automatically, '
                 'or can be specified manually.')
    add_arg('outfile', metavar='<outfile>',
            help='The output file. The type is defined by preset '
                 'or can be specified manually.')

    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')

    add_arg('--input-format', dest='input_format', type=str,
            default=None, metavar='<id>',
            help='Input format')

    return parser


def restitcher(fmt):
    pass
    #{re.compile(r'\+ \+'):      '+',   # FIXME
    #re.compile(r'(?<!\+) \+'): ' ',   # FIXME
    #re.compile(r'\+ (?!\+)'):  ' '},  # FIXME


def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)

    with io._open_text_file_write(args.outfile) as fobj:
        pipe = io._read_text_file(args.infile)
        pipe = utils._generator_progress(pipe)
        pipe = restitcher(args.input_format)(pipe)

        for token in pipe:
            fobj.write(token)


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
