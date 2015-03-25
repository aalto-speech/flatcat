#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import sys

import flatcat
from flatcat.exception import ArgumentException

LICENSE = """
Copyright (c) 2014, Stig-Arne Gronroos
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
        prog='flatcat-compare-models',
        description="""
Morfessor FlatCat model comparison diagnostics
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    add_arg = parser.add_argument
    add_arg('modelfiles',
        metavar='<modelfile>',
        nargs='+')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')

    # FIXME: hardcoded to alpha atm
    #add_arg('-x', '--variable)

    add_arg('--aligned-reference', dest='alignref', default=None,
            metavar='<file>',
            help='FIXME')
    add_arg('--aligned-to-segment', dest='alignseg', default=None,
            metavar='<file>',
            help='FIXME')
    add_arg('--aligned-loss', dest="alignloss", type=str, default='abs',
            metavar='<type>', choices=['abs', 'square', 'zeroone', 'tot'],
            help="loss function for FIXME ('abs', 'square', 'zeroone' or"
                 "'tot'; default '%(default)s')")
    return parser

def load_model(io, modelfile):
    init_is_pickle = (modelfile.endswith('.pickled') or
                      modelfile.endswith('.pickle') or
                      modelfile.endswith('.bin'))

    init_is_tarball = (modelfile.endswith('.tar.gz') or
                       modelfile.endswith('.tgz'))
    if not init_is_pickle and not init_is_tarball:
        raise ArgumentException(
            'This tool can only load tarball and binary models')

    if init_is_pickle:
        model = io.read_binary_model_file(modelfile)
    else:
        model = io.read_tarball_model_file(modelfile)
    model.reestimate_probabilities()
    return model

def get_basic_stats(model):
    corpussize = len(model.segmentations)
    unsplit = 0
    twopart = 0
    multistem = 0
    nostem = 0
    for wa in model.segmentations:
        mlen = len(wa.analysis)
        stemcount = sum(1 for cmorph in wa.analysis
                        if cmorph.category == 'STM')
        if mlen == 1:
            unsplit += 1
        elif mlen == 2:
            twopart += 1

        if stemcount == 0:
            nostem += 1
        elif stemcount > 2:
            multistem += 1
    lexsize = 0
    lexstems = 0        # morphs that are predominantly used as stem
    lexsuffixes = 0     # morphs that are predominantly used as suffix
    hapaxstems = 0
    hapaxsuffixes = 0
    for (morph, counts) in model.get_lexicon():
        lexsize += 1
        if counts.SUF > (counts.PRE + counts.STM + counts.ZZZ):
            lexsuffixes += 1
            if counts.SUF == 1:
                hapaxsuffixes += 1
        if counts.STM > (counts.PRE + counts.SUF + counts.ZZZ):
            lexstems += 1
            if counts.STM == 1:
                hapaxstems += 1
    return {
        "corpussize": corpussize,
        "unsplit": unsplit,
        "twopart": twopart,
        "multistem": multistem,
        "nostem": nostem,
        "unsplit": unsplit,
        "twopart": twopart,
        "multistem": multistem,
        "nostem": nostem,
        "lexsize": lexsize,
        "lexstems": lexstems,
        "lexsuffixes": lexsuffixes,
        "hapaxstems": hapaxstems,
        "hapaxsuffixes": hapaxsuffixes,
    }

def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)
    models = [load_model(io, model)
              for model in args.modelfiles]
    for model in models:
        print(get_basic_stats(model))


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
