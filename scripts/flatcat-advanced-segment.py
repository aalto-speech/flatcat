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
Morfessor FlatCat advanced segmentation
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    add_arg = parser.add_argument

    add_arg('model', metavar='<flatcat model>',
            help='A FlatCat model (tarball or binary)')
    add_arg('infile', metavar='<infile>',
            help='The input file. The type will be sniffed automatically, '
                 'or can be specified manually.')
    add_arg('outfile', metavar='<outfile>',
            help='The output file. The type is defined by preset '
                 'or can be specified manually.')

    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')

    add_arg('--output-format', dest='output_format', type=str,
            default=None, metavar='<id>',
            help='Output format')

    add_arg('--dont-remove-nonmorphemes', dest='no_rm_nonmorph', default=False,
            action='store_true',
            help='Use heuristic postprocessing to remove nonmorphemes '
                 'from output segmentations.')
    add_arg('--passthrough-regex-file', dest='re_file', type=str,
            default=None, metavar='<file>',
            help='File containing regular expressions for tokens '
                 'that should be passed through without segmentation.')

    return parser


def corpus_reader(io, infile):
    for line in io._read_text_file(infile):
        for (i, token) in enumerate(line.split(' ')):
            if i != 0:
                yield ' '
            yield token
        yield '\n'



class FlatcatWrapper(object):
    def __init__(self, model, remove_nonmorphemes=True):
        self.model = model
        if remove_nonmorphemes:
            self.hpp = flatcat.categorizationscheme.HeuristicPostprocessor()
        else:
            self.hpp = None

    def segment(self, word):
        (analysis, cost) = self.model.viterbi_analyze(word)
        if self.hpp is not None:
            analysis = self.hpp.remove_nonmorphemes(analysis, self.model)
        return analysis


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

# FIXME: read segmentation bypass regexes from a file
# FIXME: adding and removing morphs from the analysis (no longer concatenative)
# FIXME: joining some morphs depending on tags (e.g. join compound modifier)
# FIXME: different delimiters for different surrounding tags

#analysis = flatcat.flatcat._wb_wrap(word.analysis)
#    if cmorph.category == flatcat.WORD_BOUNDARY:
#        continue
#    out.append(self._morph_formatter(cmorph))
#formatted = ''.join(out)

# ''.join(cmorph.morph for cmorph in word.analysis)


def split_compound(morphs):
    out = []
    current = []
    prev = None
    for morph in morphs:
        if prev is not None and prev != 'PRE':
            if morph.category in ('PRE', 'STM'):
                out.append(current)
                current = []
        current.append(morph)
        prev = morph.category
    out.append(current)
    return out

def mark_by_tag(morphs):
    for morph in morphs:
        if morph.category == 'PRE':
            yield '{}+'.format(morph.morph)
        elif morph.category == 'STM':
            yield '{}'.format(morph.morph)
        elif morph.category == 'SUF':
            yield '+{}'.format(morph.morph)
        elif morph.category is None:
            yield morph.morph
        else:
            assert False, morph.category

def postprocess(fmt, morphs):
    if fmt == 'both_sides':
        #ala+ +kive+ +n+   +kolo+ +on
        return '+ +'.join(cmorph.morph for cmorph in morphs)
    if fmt == 'right_only':
        #ala  +kive  +n    +kolo  +on
        return ' +'.join(cmorph.morph for cmorph in morphs)
    #elif fmt == 'affix_only':
    #    #ala+  kive  +n?    kolo  +on
    #    compound = split_compound(morphs)
    #    pass
    elif fmt == 'compound_symbol':
        #ala+  kive  +n <c> kolo  +on
        parts = split_compound(morphs)
        parts = [mark_by_tag(part) for part in parts]
        parts = [' '.join(part) for part in parts]
        return ' +@+ '.join(parts)
    elif fmt == 'compound_both_sides':
        #ala+ +kive+ +n>   <kolo+ +on
        parts = split_compound(morphs)
        parts = [[morph.morph for morph in part]
                 for part in parts]
        parts = ['+ +'.join(part) for part in parts]
        return '@ @'.join(parts)
    elif fmt == 'compound_affix':
        #ala+  kive  +n>    kolo  +on
        parts = split_compound(morphs)
        parts = [mark_by_tag(part) for part in parts]
        parts = [' '.join(part) for part in parts]
        return '@ '.join(parts)
    elif fmt == 'compound_modifier_affix':
        #alakiven>          kolo  +on
        parts = split_compound(morphs)
        out = []
        for part in parts[:-1]:
            part = [morph.morph for morph in part]
            out.append(''.join(part))
        part = mark_by_tag(parts[-1])
        out.append(' '.join(part))
        return '@ '.join(out)


class SegmentationCache(object):
    def __init__(self, seg_func, passthrough=None, limit=1000000):
        self.seg_func = seg_func
        if passthrough is not None:
            self.passthrough = passthrough
        else:
            self.passthrough = []
        self.limit = limit
        self._cache = {}
        self.seg_count = 0
        self.unseg_count = 0

    def segment(self, word):
        if any(pattern.match(word)
               for pattern in self.passthrough):
            return [flatcat.CategorizedMorph(word, None)]
        if len(self._cache) > self.limit:
            # brute solution clears whole cache once limit is reached
            self._cache = {}
        if word not in self._cache:
            self._cache[word] = self.seg_func(word)
        seg = self._cache[word]
        if len(seg) > 1:
            self.seg_count += 1
        else:
            self.unseg_count += 1
        return seg

    def segment_from(self, pipe):
        for word in pipe:
            yield self.segment(word)


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


def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)

    passthrough = []
    if args.re_file is not None:
        for line in io._read_text_file(args.re_file):
            passthrough.append(
                re.compile(line))
    model = load_model(io, args.model)
    model_wrapper = FlatcatWrapper(
        model,
        remove_nonmorphemes=(not args.no_rm_nonmorph))
    cache = SegmentationCache(model_wrapper.segment, passthrough)

    with io._open_text_file_write(args.outfile) as fobj:
        pipe = corpus_reader(io, args.infile)
        pipe = utils._generator_progress(pipe, 10000)
        pipe = cache.segment_from(pipe)
        # FIXME: transformations (joining/filtering) here
        pipe = (postprocess(args.output_format, morphs)
                for morphs in pipe)

        for token in pipe:
            fobj.write(token)
    tot_count = cache.seg_count + cache.unseg_count
    seg_prop = float(cache.seg_count) / float(tot_count)
    print('{} segmented ({}), {} unsegmented, {} total'.format(
        cache.seg_count, seg_prop, cache.unseg_count, tot_count))


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
