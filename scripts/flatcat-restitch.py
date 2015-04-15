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


RE_BOTH_SIDES = re.compile(r'\+ \+')
RE_RIGHT_ONLY = re.compile(r'(?<!\+) \+')
RE_LEFT_ONLY = re.compile(r'\+ (?!\+)')
COMPOUND = r' +@+ '
COMPOUND_BOTH = r'@ @'
COMPOUND_LEFT = r'@ '


def restitcher(fmt, line):
    if fmt == 'both_sides':
        #ala+ +kive+ +n+   +kolo+ +on
        line = RE_RIGHT_ONLY.sub(' ', line)
        line = RE_LEFT_ONLY.sub( ' ', line)
        line = RE_BOTH_SIDES.sub('',  line)
        return line
    elif fmt == 'right_only':
        line = RE_RIGHT_ONLY.sub('', line)
        return line
    elif fmt == 'compound_symbol':
        #ala+  kive  +n <c> kolo  +on
        line = line.replace(COMPOUND, '')
        line = RE_RIGHT_ONLY.sub('', line)
        line = RE_LEFT_ONLY.sub( '', line)
        return line
    elif fmt == 'compound_both_sides':
        #ala+ +kive+ +n>   <kolo+ +on
        line = line.replace(COMPOUND_BOTH, '')
        line = RE_RIGHT_ONLY.sub(' ', line)
        line = RE_LEFT_ONLY.sub( ' ', line)
        line = RE_BOTH_SIDES.sub('',  line)
        return line
    elif (fmt == 'compound_affix'
          or fmt == 'compound_modifier_affix'):
        #ala+  kive  +n>    kolo  +on
        line = line.replace(COMPOUND_LEFT, '')
        line = RE_RIGHT_ONLY.sub('', line)
        line = RE_LEFT_ONLY.sub( '', line)
        line = RE_BOTH_SIDES.sub('',  line)
        return line
    elif fmt == 'advanced':
        line = RE_RIGHT_ONLY.sub('', line)
        line = RE_LEFT_ONLY.sub( '', line)
        line = RE_BOTH_SIDES.sub('',  line)
        return line


def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)

    with io._open_text_file_write(args.outfile) as fobj:
        pipe = io._read_text_file(args.infile)
        if args.outfile != '-':
            pipe = utils._generator_progress(pipe)
        pipe = (restitcher(args.input_format, line)
                for line in pipe)

        for line in pipe:
            fobj.write(line)
            fobj.write('\n')


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
