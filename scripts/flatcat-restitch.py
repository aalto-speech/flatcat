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
    add_arg('preset', metavar='<subcommand>', type=str,
            choices=['segment', 'restitch'],
            help='Presets defining sensible default values for what to do. '
                 'segment: reads in a corpus (running tokens, not a list), '
                 'outputs the segmented tokens. '
                 'restitch: reads in a segmented corpus, '
                 'outputs the reconstructed surface forms of words.')

    add_arg('infile', metavar='<infile>',
            help='The input file. The type will be sniffed automatically, '
                 'or can be specified manually.')
    add_arg('outfile', metavar='<outfile>',
            help='The output file. The type is defined by preset '
                 'or can be specified manually.')
    add_arg('-m', '--model', dest='model', metavar='<flatcat model>',
            help='A FlatCat model (tarball or binary), '
                 'for the operations that require a model to work.')

    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')

    add_arg('--input-category-separator', dest='catseparator', type=str,
            default=None, metavar='<regexp>',
            help='Manually set input morph category tag separator. ')
    add_arg('--input-format', dest='input_format', type=str,
            default=None, metavar='<id>',
            help='Input format')

    add_arg('--output-format', dest='output_format', type=str,
            default=None, metavar='<id>',
            help='Output format')

    add_arg('--remove-nonmorphemes', dest='rm_nonmorph', default=False,
            action='store_true',
            help='Use heuristic postprocessing to remove nonmorphemes '
                 'from output segmentations.')

    return parser


def corpus_reader(io, infile):
    for line in io._read_text_file(infile):
        for token in line.split(' '):
            yield token
        yield '\n'


def preprocessor(fmt):
    if fmt == 'tags':
        def _pre(pipe):
            for token in pipe:
                yield io._morph_or_cmorph(x)
    else:
        def _pre(pipe):
            for token in pipe:
                yield CategorizedMorph(token, None)
        

#def restitcher(fmt):
#    if fmt == 'both_sides':
#        def _res(pipe):
#            prev = None
#            expect = False
#            for token in pipe:
#                if token.morph[0] == '+':
#                    if expect:
#                        token = prev.morph + token.morph


# FIXME: adding and removing morphs from the analysis (no longer concatenative)
# FIXME: joining some morphs depending on tags (e.g. join compound modifier)
# FIXME: different delimiters for different surrounding tags
# FIXME: restitching

class FlatcatWrapper(object):
    def __init__(self, model, remove_nonmorphemes=True):
        self.model = model
        if remove_nonmorphemes:
            self.hpp = flatcat.categorizationscheme.HeuristicPostprocessor()
        else:
            self.hpp = None

    def segment(self, word):
        (analysis, cost) = self.model.viterbi_analyze(word.word)
        if self.hpp is not None:
            analysis = self.hpp.remove_nonmorphemes(analysis, self.model)
        return analysis



#analysis = flatcat.flatcat._wb_wrap(word.analysis)
#    if cmorph.category == flatcat.WORD_BOUNDARY:
#        continue
#    out.append(self._morph_formatter(cmorph))
#formatted = ''.join(out)

def _make_morph_formatter(self, category_sep, strip_tags):
    if not strip_tags:
        def output_morph(cmorph):
            if cmorph.category is None:
                return cmorph.morph
            return '{}{}{}'.format(cmorph.morph,
                                    category_sep,
                                    cmorph.category)
    else:
        def output_morph(cmorph):
            try:
                return cmorph.morph
            except AttributeError:
                return cmorph
    return output_morph


# ''.join(cmorph.morph for cmorph in word.analysis)


class SegmentationCache(object):
    def __init__(self, seg_func, limit=1000000):
        self.seg_func = seg_func
        self.limit = limit
        self._cache = {}

    def segment(self, word):
        if len(self._cache) > self.limit:
            # brute solution clears whole cache once limit is reached
            self._cache = {}
        if word not in self._cache:
            self._cache[word] = self.seg_func(word)
        return self._cache[word]


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
    model.initialize_hmm()
    return model


# FIXME: these should be in utils?
_preferred_encoding = locale.getpreferredencoding()


# FIXME: these should be in utils?
def _locale_decoder(s):
    """ Decodes commandline input in locale """
    return unicode(s.decode(_preferred_encoding))


def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)
    model = None
    if args.model is not None:
        model = load_model(io, args.model)

    tsep = args.outputtagseparator
    if not PY3:
        if tsep is not None:
            tsep = _locale_decoder(tsep)

    if tsep is None:
        tsep = '/'

    if args.preset == 'segment':
        assert model is not None
        model_wrapper = FlatcatWrapper(
            model,
            remove_nonmorphemes=args.rm_nonmorph,
            clogp=False)
        cache = SegmentationCache(process_item)

    with io._open_text_file_write(args.outfile) as fobj:
        pipe = corpus_reader(io, args.infile)
        pipe = preprocessor(args.input_format)(pipe)
        pipe = utils._generator_progress(pipe)
        if args.preset == 'segment':
            pipe = cache.segment_from(pipe)
        elif args.preset == 'restitch':
            pipe = restitcher(args.input_format)(pipe)
        pipe = postrocessor(args.output_format)(pipe)

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
