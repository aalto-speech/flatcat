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
    #FIXME: mixing corpus and list formatting is just a bad idea
    add_arg('preset', metavar='<preset>', type=str,
            choices=['custom', 'segment', 'restitch', 'reformat-list'],
            help='Presets defining sensible default values for what to do. '
                 'custom: you must manually specify everything. '
                 'segment: reads in a corpus (running tokens, not a list), '
                 'outputs the segmented tokens. '
                 'restitch: reads in a segmented corpus, '
                 'outputs the reconstructed surface forms of words.'
                 'reformat-list: manipulate a list(/table) of segmented '
                 'word types.')

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

    add_arg('--input-column-separator', dest='cseparator', type=str,
            default=None, metavar='<regexp>',
            help='Manually set input column separator regexp.')
    add_arg('--input-morph-separator', dest='consseparator', type=str,
            default=None, metavar='<string>',
            help='Manually set input morph (construction) separator string.')
    add_arg('--input-category-separator', dest='catseparator', type=str,
            default=None, metavar='<regexp>',
            help='Manually set input morph category tag separator. ')
    add_arg('--input-not-tagged', dest='input_not_tagged', default=False,
            action='store_true',
            help='Input format hint: input does not contain category tags.')

    add_arg('--output-format', dest='outputformat', type=str,
            default=None,
            metavar='<format>',
            help='Format string for --output file (default: "%(default)s"). '
                 'Valid keywords are: '
                 '{analysis} = morphs of the word, '
                 '{compound} = word, '
                 '{count} = count of the word (currently always 1), and '
                 '{logprob} = log-probability of the analysis. Valid escape '
                 'sequences are "\\n" (newline) and "\\t" (tabular)')
    add_arg('--output-morph-separator', dest='outputconseparator',
            type=str, default=None, metavar='<str>',
            help='Construction separator for analysis in output.')
    add_arg('--output-category-separator', dest='outputtagseparator',
            type=str, default=None, metavar='<str>',
            help='Category tag separator for analysis in --output file ')
    add_arg('--strip-categories', dest='striptags', default=False,
            action='store_true',
            help='Remove tags if present in the input')
    add_arg('--output-newlines', dest='outputnewlines', default=False,
            action='store_true',
            help='For each newline in input, print newline in --output file '
            '(default: "%(default)s")')

    add_arg('--remove-nonmorphemes', dest='rm_nonmorph', default=False,
            action='store_true',
            help='Use heuristic postprocessing to remove nonmorphemes '
                 'from output segmentations.')
    add_arg('--cache', dest='use_cache', default=False,
            action='store_true',
            help='Use a cache for segmentations. Useful for corpora (tokens) '
                 'but wasteful for lists (types).')

    return parser


IntermediaryFormat = collections.namedtuple('IntermediaryFormat',
    ['count', 'word', 'analysis', 'logp', 'clogp'])

# FIXME sniffer:
# if input is a flatcat tgz or bin model: extract analysis part
# columns? type of first whitespace? mix of spaces and tabs?
# is first column a number?
# is there a known morph delimiter?
# is there a known category delimiter? category tags?
# is the first column a concatenation of later morphs?

# FIXME: adding and removing morphs from the analysis (no longer concatenative)
# FIXME: joining some morphs depending on tags (e.g. join compound modifier)
# FIXME: different delimiters for different surrounding tags
# FIXME: restitching


class FlatcatWrapper(object):
    def __init__(self, model, remove_nonmorphemes=True, clogp=False):
        self.model = model
        if remove_nonmorphemes:
            self.hpp = flatcat.categorizationscheme.HeuristicPostprocessor()
        else:
            self.hpp = None
        self.clogp = clogp

    def segment(self, word):
        (analysis, cost) = self.model.viterbi_analyze(word.word)
        if self.hpp is not None:
            analysis = self.hpp.remove_nonmorphemes(analysis, self.model)
        if self.clogp:
            clogp = self.model.forward_logprob(word)
        else:
            clogp = 0
        return IntermediaryFormat(
            word.count,
            word.word,
            analysis,
            cost,
            clogp)


class AnalysisFormatter(object):
    def __init__(self,
                 morph_sep=' + ',   # can also be func(tag, tag)
                 category_sep='/',
                 strip_tags=False):
        if utils._is_string(morph_sep):
            def morph_sep_func(left, right):
                if (left == flatcat.WORD_BOUNDARY or
                        right == flatcat.WORD_BOUNDARY):
                    return ''
                else:
                    return morph_sep
            self._morph_sep = morph_sep_func
        else:
            self._morph_sep = morph_sep
        self.category_sep = category_sep
        self.strip_tags = strip_tags
        self._morph_formatter = self._make_morph_formatter(category_sep,
                                                           strip_tags)

    def segment(self, word):
        analysis = flatcat.flatcat._wb_wrap(word.analysis)
        out = []
        for (i, cmorph) in enumerate(analysis):
            out.append(self._morph_sep(
                analysis[i - 1].category,
                cmorph.category))
            if cmorph.category == flatcat.WORD_BOUNDARY:
                continue
            out.append(self._morph_formatter(cmorph))
        formatted = ''.join(out)
        return IntermediaryFormat(
            word.count,
            word.word,
            formatted,
            word.logp,
            word.clogp)

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


def restitch(word):
    if word.word is None or len(word.word) == 0:
        stitched = ''.join(cmorph.morph
                           for cmorph in word.analysis)
    else:
        stitched = word.word
    return IntermediaryFormat(
        word.count,
        stitched,
        word.analysis,
        word.logp,
        word.clogp)


# FIXME: has nothing specificly with segmentation to do: rename
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


# FIXME: should be part of sniffer?
# FIXME: input format overrides not used atm
def dummy_reader(io, infile):
    for item in io.read_corpus_file(infile):
        (count, compound, atoms) = item
        yield IntermediaryFormat(
            count,
            compound,
            atoms,
            0, 0)


def segmented_corpus_reader(io, infile, mapping=None, not_tagged=False):
    if mapping is None:
        mapping = {}
    for line in io._read_text_file(infile):
        for (pattern, repl) in mapping.items():
            line = pattern.sub(repl, line)
        # after this '+' is morph boundary, ' ' is word boundary
        words = line.split(' ')
        for word in words:
            morphs = word.split('+')
            if not_tagged:
                morphs = tuple(flatcat.CategorizedMorph(x, None)
                               for x in morphs)
            else:
                morphs = tuple(io._morph_or_cmorph(x) for x in morphs)
            yield IntermediaryFormat(
                1,
                None,
                morphs,
                0, 0)
        yield IntermediaryFormat(0, '\n', (), 0, 0)


def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)
    model = None
    if args.model is not None:
        model = load_model(io, args.model)

    outformat = args.outputformat
    csep = args.outputconseparator
    tsep = args.outputtagseparator
    if not PY3:
        if outformat is not None:
            outformat = _locale_decoder(outformat)
        if csep is not None:
            csep = _locale_decoder(csep)
        if tsep is not None:
            tsep = _locale_decoder(tsep)

    cache = args.use_cache
    outputnewlines = args.outputnewlines
    if args.preset == 'custom':
        if csep is None: raise ArgumentException('--output-morph-separator')
        if tsep is None: raise ArgumentException('--output-category-separator')
        if outformat is None: raise ArgumentException('--output-format')
    elif args.preset == 'segment':
        cache = True
        outputnewlines = True
        if outformat is None:
            outformat = r'{analysis} '
    elif args.preset == 'restitch':
        cache = True
        outputnewlines = True
        if outformat is None:
            outformat = r'{compound} '  # FIXME: extra sentence final space
    elif args.preset == 'reformat-list':
        if outformat is None:
            outformat = r'{analysis}\n'

    if csep is None:
        csep = ' + '
    if tsep is None:
        tsep = '/'
    outformat = outformat.replace(r"\n", "\n")
    outformat = outformat.replace(r"\t", "\t")
    keywords = [x[1] for x in string.Formatter().parse(outformat)]

    # chain of functions to apply to each item
    item_steps = []

    if model is not None:   # FIXME: what condition?
        model_wrapper = FlatcatWrapper(
            model,
            remove_nonmorphemes=args.rm_nonmorph,
            clogp=('clogprob' in keywords))
        item_steps.append(model_wrapper.segment)

    if 'compound' in keywords:
        item_steps.append(restitch)

    analysis_formatter = AnalysisFormatter(
        csep,   # FIXME
        tsep,
        args.striptags)
    item_steps.append(analysis_formatter.segment)

    def process_item(item):
        for func in item_steps:
            item = func(item)
        return item

    cache = SegmentationCache(process_item)

    with io._open_text_file_write(args.outfile) as fobj:
        if args.preset == 'restitch':    # FIXME
            pipe = segmented_corpus_reader(
                io, args.infile,
                {re.compile(r'\+ \+'):      '+',   # FIXME
                 re.compile(r'(?<!\+) \+'): ' ',   # FIXME
                 re.compile(r'\+ (?!\+)'):  ' '},  # FIXME
                args.input_not_tagged
            )
        else:
            pipe = dummy_reader(io, args.infile)
        pipe = utils._generator_progress(pipe)
        for item in pipe:
            if len(item.analysis) == 0:
                # is a corpus newline marker
                if outputnewlines:
                    fobj.write("\n")
                continue
            item = cache.segment(item)
            fobj.write(outformat.format(
                       count=item.count,
                       compound=item.word,
                       analysis=item.analysis,
                       logprob=item.logp,
                       clogprob=item.clogp))


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
