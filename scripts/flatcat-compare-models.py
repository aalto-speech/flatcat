#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import sys
import numpy as np
from matplotlib import pyplot as plt

import morfessor
import flatcat
from flatcat.exception import ArgumentException

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
    add_arg('-x', '--variable', dest='xvar', default='corpusweight')

    add_arg('--weight-threshold', dest='threshold', default=0.01,
            metavar='<float>', type=float,
            help='percentual stopping threshold for corpusweight updaters')
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

def get_aligned_token_cost(updater, model):
    distribution = collections.Counter()
    (cost, direction) = updater.evaluation(
        model, distribution=distribution)
    return {
        'align_cost': cost,
        'align_dir': direction,
        'align_distribution': distribution}

def aligned_token_counts(model, updater):
    pass

def lineplot(stats, xvar, yvars,
    normalize=None, direction=None, ylab=None):
    # FIXME direction not implemented
    plt.figure()
    models = sorted((stats[model][xvar], model)
                     for model in stats.keys())
    xs = [x for (x, model) in models]
    models = [model for (x, model) in models]
    for yvar in yvars:
        ys = [stats[model][yvar] for model in models]
        if normalize is not None:
            ys = [float(y) / stats[model][normalize] for y in ys]
        if direction is None:
            marker = 'o'
            markers = None
        else:
            marker = None
            dirs = [stats[model][direction] for model in models]
            markers = ['^' if y > 0 else 'v' for y in dirs]
        print(xs, ys)
        plt.plot(xs, ys, marker=marker, label=yvar)
        if markers is not None:
            for (x, y, m) in zip(xs, ys, markers):
                color = 'green' if m == '^' else 'red'
                plt.plot(x, y, marker=m, markerfacecolor=color)
    plt.legend()
    plt.xlabel(xvar)
    if ylab is not None:
        plt.ylabel(ylab)

def distrplot(stats, xvar, yvar):
    models = sorted((stats[model][xvar], model)
                     for model in stats.keys())
    for (i, pair) in enumerate(models):
        (x, model) = pair
        plt.subplot(len(models), 1, i + 1)

        s = sorted(stats[model][yvar].items())
        labels, values = zip(*s)
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.xlabel(yvar)
        plt.title(model)
    pass

def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)
    models = [load_model(io, model)
              for model in args.modelfiles]

    updater = None
    if args.alignref is not None:
        if args.alignseg is None:
            raise ArgumentException(
                'If --aligned-reference is specified, '
                'you must also specify --aligned-to-segment')
        postfunc = None
        if args.alignloss == 'abs':
            lossfunc = abs
        elif args.alignloss == 'square':
            lossfunc = lambda x: x**2
        elif args.alignloss == 'zeroone':
            lossfunc = lambda x: 0 if x == 0 else 1
        elif args.alignloss == 'tot':
            lossfunc = lambda x: x
            postfunc = abs
        else:
            raise ArgumentException(
                "unknown alignloss type '{}'".format(args.alignloss))
        updater = morfessor.baseline.AlignedTokenCountCorpusWeight(
            io._read_text_file(args.alignseg),
            io._read_text_file(args.alignref),
            args.threshold,
            lossfunc,
            postfunc)

    stats = collections.defaultdict(dict)
    for (name, model) in zip(args.modelfiles, models):
        stats[name].update(model.get_params())
        stats[name].update(get_basic_stats(model))
        if updater is not None:
            stats[name].update(get_aligned_token_cost(updater, model))
        print(stats[name])

    # FIXME: save and load stats (or separate tool?)

    lineplot(stats, args.xvar,
             ['unsplit', 'twopart', 'multistem', 'nostem', ],
             normalize='corpussize', ylab='proportion of corpus')
    lineplot(stats, args.xvar,
             ['hapaxstems', 'lexstems', 'hapaxsuffixes', 'lexsuffixes'],
             ylab='morph types')
    lineplot(stats, args.xvar, ['lexsize'], ylab='morph types')
    lineplot(stats, args.xvar, ['align_cost'], direction='align_dir')
    distrplot(stats, args.xvar, 'align_distribution')
    plt.show()

if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
