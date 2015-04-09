#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import sys
import cPickle as pickle
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
        nargs='*')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')

    add_arg('-s', '--save', dest='savefile', default=None)
    add_arg('-l', '--load', dest='loadfile', default=None)

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

class OnlyDiffDistr(object):
    def __init__(self):
        self.distribution = collections.Counter()

    def count(self, ref, unsegcount, diff):
        self.distribution[diff] += 1

    def get_diff_distr(self):
        return self.distribution

class ScatterDistr(OnlyDiffDistr):
    def __init__(self):
        super(ScatterDistr, self).__init__()
        self.ref_diff = collections.Counter()
        self.ref_unseg = collections.Counter()
        self.diff_unseg = collections.Counter()

    def count(self, ref, unsegcount, diff):
        super(ScatterDistr, self).count(ref, unsegcount, diff)
        self.ref_diff[(ref, diff)] += 1
        self.ref_unseg[(ref, unsegcount)] += 1
        self.diff_unseg[(diff, unsegcount)] += 1

def get_aligned_token_cost(updater, model):
    #distribution = OnlyDiffDistr()
    distribution = ScatterDistr()
    (cost, direction) = updater.evaluation(
        model, distribution=distribution.count)
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
    absmax = 0
    for (_, model) in models:
        distr = stats[model][yvar].get_diff_distr()
        absmax = max([absmax] + [abs(y) for y in distr.keys()])
    plt.figure()
    for (i, pair) in enumerate(models):
        (x, model) = pair
        distr = stats[model][yvar].get_diff_distr()
        plt.subplot(len(models), 1, i + 1)

        labels = range(-absmax, absmax + 1)
        values = [distr[xx]
                  for xx in labels]
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.xlabel(yvar)
        plt.xlim([0, 2*absmax])
        plt.title('{} = {}'.format(xvar, x))
    pass

def scatterplot(stats):
    # just one arbitrary model (recommended to run with just one)
    model = stats.itervalues().next()
    try:
        distr = model['align_distribution'].ref_diff
    except AttributeError:
        return
    plt.figure()
    norm = None
    for (pair, val) in distr.most_common():
        if norm is None:
            normdiv = max(1., float(val) / 25.)
            norm = lambda x: max(0.5, float(val) / normdiv)
        (ref, diff) = pair
        plt.plot(ref, diff, marker='o', ms=norm(val))
    plt.xlabel('ref')
    plt.ylabel('diff')

    distr = model['align_distribution'].diff_unseg
    plt.figure()
    for (pair, val) in distr.most_common():
        (diff, unseg) = pair
        plt.plot(unseg, diff, marker='o', ms=norm(val))
    plt.xlabel('unseg')
    plt.ylabel('diff')

    distr = model['align_distribution'].ref_unseg
    plt.figure()
    maxref = maxus = 0
    for (pair, val) in distr.most_common():
        (ref, unseg) = pair
        maxref = max(ref, maxref)
        maxus = max(unseg, maxus)
        plt.plot(ref, unseg, marker='o', ms=norm(val))
    discardlim = 15
    linemax = min(maxref, maxus)
    plt.plot([discardlim, linemax], [0, linemax-discardlim], color=(.8,.8,.8))
    plt.plot([0, linemax-discardlim], [discardlim, linemax], color=(.8,.8,.8))
    plt.xlabel('ref')
    plt.ylabel('unseg')


def main(args):
    io = flatcat.io.FlatcatIO(encoding=args.encoding)

    if args.loadfile is None and len(args.modelfiles) == 0:
        raise ArgumentException(
            'Must specify either modelfiles or --load')

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

    if args.loadfile is not None:
        with open(args.loadfile, 'rb') as fobj:
            stats = pickle.load(fobj)
    else:
        stats = collections.defaultdict(dict)
    for name in args.modelfiles:
        model = load_model(io, name)
        stats[name].update(model.get_params())
        stats[name].update(get_basic_stats(model))
        if updater is not None:
            stats[name].update(get_aligned_token_cost(updater, model))
        print(stats[name])

    if args.savefile is not None:
        with open(args.savefile, 'wb') as fobj:
            pickle.dump(stats, fobj, pickle.HIGHEST_PROTOCOL)

    lineplot(stats, args.xvar,
             ['unsplit', 'twopart', 'multistem', 'nostem', ],
             normalize='corpussize', ylab='proportion of corpus')
    lineplot(stats, args.xvar,
             ['hapaxstems', 'lexstems', 'hapaxsuffixes', 'lexsuffixes'],
             ylab='morph types')
    lineplot(stats, args.xvar, ['lexsize'], ylab='morph types')
    if updater is not None or 'align_cost' in stats.itervalues().next():
        lineplot(stats, args.xvar, ['align_cost'], direction='align_dir')
        distrplot(stats, args.xvar, 'align_distribution')
        scatterplot(stats)
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
