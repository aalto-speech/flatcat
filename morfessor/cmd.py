from __future__ import unicode_literals

import logging
import math
import random
import sys
import time

from . import get_version
from .baseline import BaselineModel
from .catmap import CatmapModel, CorpusWeightUpdater, train_batch
from .categorizationscheme import MorphUsageProperties, HeuristicPostprocessor
from .diagnostics import IterationStatistics
from .exception import ArgumentException
from .io import MorfessorIO, CatmapIO
from .utils import _generator_progress, LOGPROB_ZERO

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


LICENSE = """
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


def get_default_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        prog='morfessor.py',
        description="""
Morfessor {version}

Copyright (c) 2012, Sami Virpioja and Peter Smit
All rights reserved.
{license}

Command-line arguments:
""" .format(version=get_version(), license=LICENSE),
        epilog="""
Simple usage examples (training and testing):

  %(prog)s -t training_corpus.txt -s model.pickled
  %(prog)s -l model.pickled -T test_corpus.txt -o test_corpus.segmented

Interactive use (read corpus from user):

  %(prog)s -m online -v 2 -t -

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)

    # Options for input data files
    add_arg = parser.add_argument_group('input data files').add_argument
    add_arg('-l', '--load', dest="loadfile", default=None, metavar='<file>',
            help="load existing model from file (pickled model object)")
    add_arg('-L', '--load-segmentation', dest="loadsegfile", default=None,
            metavar='<file>',
            help="load existing model from segmentation "
                 "file (Morfessor 1.0 format)")
    add_arg('-t', '--traindata', dest='trainfiles', action='append',
            default=[], metavar='<file>',
            help="input corpus file(s) for training (text or bz2/gzipped text;"
                 " use '-' for standard input; add several times in order to "
                 "append multiple files)")
    add_arg('-T', '--testdata', dest='testfiles', action='append',
            default=[], metavar='<file>',
            help="input corpus file(s) to analyze (text or bz2/gzipped text;  "
                 "use '-' for standard input; add several times in order to "
                 "append multiple files)")

    # Options for output data files
    add_arg = parser.add_argument_group('output data files').add_argument
    add_arg('-o', '--output', dest="outfile", default='-', metavar='<file>',
            help="output file for test data results (for standard output, "
                 "use '-'; default '%(default)s')")
    add_arg('-s', '--save', dest="savefile", default=None, metavar='<file>',
            help="save final model to file (pickled model object)")
    add_arg('-S', '--save-segmentation', dest="savesegfile", default=None,
            metavar='<file>',
            help="save model segmentations to file (Morfessor 1.0 format)")
    add_arg('-x', '--lexicon', dest="lexfile", default=None, metavar='<file>',
            help="output final lexicon to given file")

    # Options for data formats
    add_arg = parser.add_argument_group(
        'data format options').add_argument
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help="encoding of input and output files (if none is given, "
                 "both the local encoding and UTF-8 are tried)")
    add_arg('--lowercase', dest="lowercase", default=False,
            action='store_true',
            help="lowercase input data")
    add_arg('--traindata-list', dest="list", default=False,
            action='store_true',
            help="input file(s) for batch training are lists "
                 "(one compound per line, optionally count as a prefix)")
    add_arg('--atom-separator', dest="separator", type=str, default=None,
            metavar='<regexp>',
            help="atom separator regexp (default %(default)s)")
    add_arg('--compound-separator', dest="cseparator", type=str, default='\s+',
            metavar='<regexp>',
            help="compound separator regexp (default '%(default)s')")
    add_arg('--analysis-separator', dest='analysisseparator', type=str,
            default=',', metavar='<str>',
            help="separator for different analyses in an annotation file. Use"
                 "  NONE for only allowing one analysis per line")
    add_arg('--output-format', dest='outputformat', type=str,
            default=r'{analysis}\n', metavar='<format>',
            help="format string for --output file (default: '%(default)s'). "
            "Valid keywords are: "
            "{analysis} = constructions of the compound, "
            "{compound} = compound string, "
            "{count} = count of the compound (currently always 1), and "
            "{logprob} = log-probability of the compound. Valid escape "
            "sequences are '\\n' (newline) and '\\t' (tabular)")
    add_arg('--output-format-separator', dest='outputformatseparator',
            type=str, default=' ', metavar='<str>',
            help="construction separator for analysis in --output file "
            "(default: '%(default)s')")

    # Options for model training
    add_arg = parser.add_argument_group(
        'training and segmentation options').add_argument
    add_arg('-m', '--mode', dest="trainmode", default='init+batch',
            metavar='<mode>',
            choices=['none', 'batch', 'init', 'init+batch', 'online',
                     'online+batch'],
            help="training mode ('none', 'init', 'batch', 'init+batch', "
                 "'online', or 'online+batch'; default '%(default)s')")
    add_arg('-a', '--algorithm', dest="algorithm", default='recursive',
            metavar='<algorithm>', choices=['recursive', 'viterbi'],
            help="algorithm type ('recursive', 'viterbi'; default "
                 "'%(default)s')")
    add_arg('-d', '--dampening', dest="dampening", type=str, default='none',
            metavar='<type>', choices=['none', 'log', 'ones'],
            help="frequency dampening for training data ('none', 'log', or "
                 "'ones'; default '%(default)s')")
    add_arg('-f', '--forcesplit', dest="forcesplit", type=list, default=['-'],
            metavar='<list>',
            help="force split on given atoms (default %(default)s)")
    add_arg('-F', '--finish-threshold', dest='finish_threshold', type=float,
            default=0.005, metavar='<float>',
            help="Stopping threshold. Training stops when "
                 "the improvement of the last iteration is"
                 "smaller then finish_threshold * #boundaries; "
                 "(default '%(default)s')")
    add_arg('-r', '--randseed', dest="randseed", default=None,
            metavar='<seed>',
            help="seed for random number generator")
    add_arg('-R', '--randsplit', dest="splitprob", default=None, type=float,
            metavar='<float>',
            help="initialize new words by random splitting using the given "
                 "split probability (default no splitting)")
    add_arg('--skips', dest="skips", default=False, action='store_true',
            help="use random skips for frequently seen compounds to speed up "
                 "training")
    add_arg('--batch-minfreq', dest="freqthreshold", type=int, default=1,
            metavar='<int>',
            help="compound frequency threshold for batch training (default "
                 "%(default)s)")
    add_arg('--max-epochs', dest='maxepochs', type=int, default=None,
            metavar='<int>',
            help='hard maximum of epochs in training')
    add_arg('--nosplit-re', dest="nosplit", type=str, default=None,
            metavar='<regexp>',
            help="if the expression matches the two surrounding characters, "
                 "do not allow splitting (default %(default)s)")
    add_arg('--online-epochint', dest="epochinterval", type=int,
            default=10000, metavar='<int>',
            help="epoch interval for online training (default %(default)s)")
    add_arg('--viterbi-smoothing', dest="viterbismooth", default=0,
            type=float, metavar='<float>',
            help="additive smoothing parameter for Viterbi training "
                 "and segmentation (default %(default)s)")
    add_arg('--viterbi-maxlen', dest="viterbimaxlen", default=30,
            type=int, metavar='<int>',
            help="maximum construction length in Viterbi training "
                 "and segmentation (default %(default)s)")

    # Options for semi-supervised model training
    add_arg = parser.add_argument_group(
        'semi-supervised training options').add_argument
    add_arg('-A', '--annotations', dest="annofile", default=None,
            metavar='<file>',
            help="load annotated data for semi-supervised learning")
    add_arg('-D', '--develset', dest="develfile", default=None,
            metavar='<file>',
            help="load annotated data for tuning the corpus weight parameter")
    add_arg('-w', '--corpusweight', dest="corpusweight", type=float,
            default=1.0, metavar='<float>',
            help="corpus weight parameter (default %(default)s); "
                 "sets the initial value if --develset is used")
    add_arg('-W', '--annotationweight', dest="annotationweight",
            type=float, default=None, metavar='<float>',
            help="corpus weight parameter for annotated data (if unset, the "
                 "weight is set to balance the number of tokens in annotated "
                 "and unannotated data sets)")

    # Options for logging
    add_arg = parser.add_argument_group('logging options').add_argument
    add_arg('-v', '--verbose', dest="verbose", type=int, default=1,
            metavar='<int>',
            help="verbose level; controls what is written to the standard "
                 "error stream or log file (default %(default)s)")
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help="write log messages to file in addition to standard "
                 "error stream")
    add_arg('--progressbar', dest='progress', default=False,
            action='store_true',
            help="Force the progressbar to be displayed (possibly lowers the "
                 "log level for the standard error stream)")

    add_arg = parser.add_argument_group('other options').add_argument
    add_arg('-h', '--help', action='help',
            help="show this help message and exit")
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help="show version number and exit")

    return parser


def main(args):
    if args.verbose >= 2:
        loglevel = logging.DEBUG
    elif args.verbose >= 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logging_format = '%(asctime)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    default_formatter = logging.Formatter(logging_format, date_format)
    plain_formatter = logging.Formatter('%(message)s')
    logging.basicConfig(level=loglevel)
    _logger.propagate = False  # do not forward messages to the root logger

    # Basic settings for logging to the error stream
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(plain_formatter)
    _logger.addHandler(ch)

    # Settings for when log_file is present
    if args.log_file is not None:
        fh = logging.FileHandler(args.log_file, 'w')
        fh.setLevel(loglevel)
        fh.setFormatter(default_formatter)
        _logger.addHandler(fh)
        # If logging to a file, make INFO the highest level for the
        # error stream
        ch.setLevel(max(loglevel, logging.INFO))

    # If debug messages are printed to screen or if stderr is not a tty (but
    # a pipe or a file), don't show the progressbar
    global show_progress_bar
    if (ch.level > logging.INFO or
            (hasattr(sys.stderr, 'isatty') and not sys.stderr.isatty())):
        show_progress_bar = False

    if args.progress:
        show_progress_bar = True
        ch.setLevel(min(ch.level, logging.INFO))

    if (args.loadfile is None and
            args.loadsegfile is None and
            len(args.trainfiles) == 0):
        raise ArgumentException("either model file or training data should "
                                "be defined")

    if args.randseed is not None:
        random.seed(args.randseed)

    io = MorfessorIO(encoding=args.encoding,
                     compound_separator=args.cseparator,
                     atom_separator=args.separator,
                     lowercase=args.lowercase)

    # Load exisiting model or create a new one
    if args.loadfile is not None:
        model = io.read_binary_model_file(args.loadfile)

    else:
        model = BaselineModel(forcesplit_list=args.forcesplit,
                              corpusweight=args.corpusweight,
                              use_skips=args.skips,
                              nosplit_re=args.nosplit)

    if args.loadsegfile is not None:
        model.load_segmentations(io.read_segmentation_file(args.loadsegfile))

    analysis_sep = (args.analysisseparator
                    if args.analysisseparator != 'NONE' else None)

    if args.annofile is not None:
        annotations = io.read_annotations_file(args.annofile,
                                               analysis_sep=analysis_sep)
        model.set_annotations(annotations, args.annotationweight)

    if args.develfile is not None:
        develannots = io.read_annotations_file(args.develfile,
                                               analysis_sep=analysis_sep)
    else:
        develannots = None

    # Set frequency dampening function
    if args.dampening == 'none':
        dampfunc = None
    elif args.dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif args.dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % args.dampening)

    # Set algorithm parameters
    if args.algorithm == 'viterbi':
        algparams = (args.viterbismooth, args.viterbimaxlen)
    else:
        algparams = ()

    # Train model
    if args.trainmode == 'none':
        pass
    elif args.trainmode == 'batch':
        if len(model.get_compounds()) == 0:
            _logger.warning("Model contains no compounds for batch training."
                            " Use 'init+batch' mode to add new data.")
        else:
            if len(args.trainfiles) > 0:
                _logger.warning("Training mode 'batch' ignores new data "
                                "files. Use 'init+batch' or 'online' to "
                                "add new compounds.")
            ts = time.time()
            e, c = model.train_batch(args.algorithm, algparams, develannots,
                                     args.finish_threshold, args.maxepochs)
            te = time.time()
            _logger.info("Epochs: %s" % e)
            _logger.info("Final cost: %s" % c)
            _logger.info("Training time: %.3fs" % (te - ts))
    elif len(args.trainfiles) > 0:
        ts = time.time()
        if args.trainmode == 'init':
            if args.list:
                data = io.read_corpus_list_files(args.trainfiles)
            else:
                data = io.read_corpus_files(args.trainfiles)
            c = model.load_data(data, args.freqthreshold, dampfunc,
                                args.splitprob)
        elif args.trainmode == 'init+batch':
            if args.list:
                data = io.read_corpus_list_files(args.trainfiles)
            else:
                data = io.read_corpus_files(args.trainfiles)
            c = model.load_data(data, args.freqthreshold, dampfunc,
                                args.splitprob)
            e, c = model.train_batch(args.algorithm, algparams, develannots,
                                     args.finish_threshold, args.maxepochs)
            _logger.info("Epochs: %s" % e)
        elif args.trainmode == 'online':
            data = io.read_corpus_files(args.trainfiles)
            e, c = model.train_online(data, dampfunc, args.epochinterval,
                                      args.algorithm, algparams,
                                      args.splitprob, args.maxepochs)
            _logger.info("Epochs: %s" % e)
        elif args.trainmode == 'online+batch':
            data = io.read_corpus_files(args.trainfiles)
            e, c = model.train_online(data, dampfunc, args.epochinterval,
                                      args.algorithm, algparams,
                                      args.splitprob, args.maxepochs)
            e, c = model.train_batch(args.algorithm, algparams, develannots,
                                     args.finish_threshold, args.maxepochs - e)
            _logger.info("Epochs: %s" % e)
        else:
            raise ArgumentException("unknown training mode '%s'"
                                    % args.trainmode)
        te = time.time()
        _logger.info("Final cost: %s" % c)
        _logger.info("Training time: %.3fs" % (te - ts))
    else:
        _logger.warning("No training data files specified.")

    # Save model
    if args.savefile is not None:
        io.write_binary_model_file(args.savefile, model)

    if args.savesegfile is not None:
        io.write_segmentation_file(args.savesegfile, model.get_segmentations())

    # Output lexicon
    if args.lexfile is not None:
        io.write_lexicon_file(args.lexfile, model.get_constructions())

    # Segment test data
    if len(args.testfiles) > 0:
        _logger.info("Segmenting test data...")
        outformat = args.outputformat
        csep = args.outputformatseparator
        if not PY3:
            outformat = unicode(outformat)
            csep = unicode(csep)
        outformat = outformat.replace(r"\n", "\n")
        outformat = outformat.replace(r"\t", "\t")
        with io._open_text_file_write(args.outfile) as fobj:
            testdata = io.read_corpus_files(args.testfiles)
            i = 0
            for count, compound, atoms in testdata:
                constructions, logp = model.viterbi_segment(
                    atoms, args.viterbismooth, args.viterbimaxlen)
                analysis = csep.join(constructions)
                fobj.write(outformat.format(
                           analysis=analysis, compound=compound,
                           count=count, logprob=logp))
                i += 1
                if i % 10000 == 0:
                    sys.stderr.write(".")
            sys.stderr.write("\n")
        _logger.info("Done.")


def get_catmap_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='catmap.py',
        description="""
Morfessor Categories-MAP {version}

Copyright (c) 2013, Stig-Arne Gronroos
All rights reserved.
{license}

Command-line arguments:
""" .format(version=get_version(), license=LICENSE),
        epilog="""
Simple usage examples (training and testing):

  %(prog)s -B baseline_segmentation.txt -p 10 -s model.pickled
  %(prog)s -l model.pickled -T test_corpus.txt -o test_corpus.segmented

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)

    # Options for input data files
    add_arg = parser.add_argument_group('input data files').add_argument
    add_arg('-l', '--load', dest="loadfile", default=None, metavar='<file>',
            help="load existing model from file (pickled model object).")
    add_arg('-B', '--load-baseline', dest="baselinefiles", default=[],
            action='append', metavar='<file>',
            help='load baseline segmentation from file ' +
                 '(Morfessor 1.0 format). ' +
                 'Can be used together with --load, ' +
                 'in which case the pickled model is extended with the ' +
                 'loaded segmentation.')
    add_arg('-L', '--load-segmentation', dest="loadsegfiles", default=[],
            action='append', metavar='<file>',
            help='load existing model from tagged segmentation ' +
                 'file (Morfessor 2.0 Categories-MAP format). ' +
                 'The probabilities are not stored in the file, ' +
                 'and must be re-estimated. ' +
                 'Can be used together with --load, ' +
                 'in which case the pickled model is extended with the ' +
                 'loaded segmentation.')
    add_arg('-T', '--testdata', dest='testfiles', action='append',
            default=[], metavar='<file>',
            help="input corpus file(s) to analyze (text or gzipped text;  "
                 "use '-' for standard input; add several times in order to "
                 "append multiple files).")
    add_arg('--loadparamsfile', dest='loadparamsfile', default=None,
            metavar='<file>',
            help='Load learned and estimated parameters from file.')

    # Options for output data files
    add_arg = parser.add_argument_group('output data files').add_argument
    add_arg('-o', '--output', dest="outfile", default='-', metavar='<file>',
            help="output file for test data results (for standard output, "
                 "use '-'; default '%(default)s').")
    add_arg('-s', '--save', dest="savefile", default=None, metavar='<file>',
            help="save final model to file (pickled model object).")
    add_arg('-S', '--save-segmentation', dest="savesegfile", default=None,
            metavar='<file>',
            help="save model segmentations to file (Morfessor 1.0 format).")
    add_arg('--saveparamsfile', dest='saveparamsfile', default=None,
            metavar='<file>',
            help='Save learned and estimated parameters to file.')

    # Options for data formats
    add_arg = parser.add_argument_group(
        'data format options').add_argument
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help="encoding of input and output files (if none is given, "
            "both the local encoding and UTF-8 are tried).")
#    add_arg('--atom-separator', dest="separator", type=str, default=None,
#            metavar='<regexp>',
#            help="atom separator regexp (default %(default)s).")
    add_arg('--compound-separator', dest="cseparator", type=str, default='\s+',
            metavar='<regexp>',
            help="compound separator regexp (default '%(default)s').")
    add_arg('--analysis-separator', dest='analysisseparator', type=str,
            default=',', metavar='<regexp>',
            help="separator for different analyses in an annotation file. Use"
                 "  NONE for only allowing one analysis per line.")
    add_arg('--category-separator', dest='catseparator', type=str, default='/',
            metavar='<regexp>',
            help='separator for the category tag following a morph. ' +
                 '(default %(default)s).')
    add_arg('--output-format', dest='outputformat', type=str,
            default=r'{analysis}\n', metavar='<format>',
            help="format string for --output file (default: '%(default)s'). "
            "Valid keywords are: "
            "{analysis} = constructions of the compound, "
            "{compound} = compound string, "
            "{count} = count of the compound (currently always 1), and "
            "{logprob} = log-probability of the compound. Valid escape "
            "sequences are '\\n' (newline) and '\\t' (tabular)")
    add_arg('--output-format-separator', dest='outputformatseparator',
            type=str, default=' ', metavar='<str>',
            help="construction separator for analysis in --output file "
            "(default: '%(default)s')")
    add_arg('--output-tags', dest='test_output_tags', default=False,
            action='store_true',
            help='output category tags in test data. ' +
                 'Default is to output only the morphs')

    # Options for training and segmentation
    add_arg = parser.add_argument_group(
        'training and segmentation options').add_argument
    add_arg('-m', '--mode', dest="trainmode", default='batch',
            metavar='<mode>',
            choices=['none', 'batch', 'online', 'online+batch'],
            help="training mode ('none', 'batch', "
                 "'online', or 'online+batch'; default '%(default)s')")
    add_arg('-p', '--perplexity-threshold', dest='ppl_threshold', type=float,
            default=100., metavar='<float>',
            help='threshold value for sigmoid used to calculate ' +
                 'probabilities from left and right perplexities. ' +
                 '(default %(default)s).')
    add_arg('--perplexity-slope', dest='ppl_slope', type=float, default=None,
            metavar='<float>',
            help='slope value for sigmoid used to calculate ' +
                 'probabilities from left and right perplexities. ' +
                 '(default 10 / perplexity-threshold).')
    add_arg('--length-threshold', dest='length_threshold', type=float,
            default=3., metavar='<float>',
            help='threshold value for sigmoid used to calculate ' +
                 'probabilities from length of morph. ' +
                 '(default %(default)s).')
    add_arg('--length-slope', dest='length_slope', type=float, default=2.,
            metavar='<float>',
            help='slope value for sigmoid used to calculate ' +
                 'probabilities from length of morph. ' +
                 '(default %(default)s).')
    add_arg('--type-perplexity', dest='type_ppl', default=False,
            action='store_true',
            help='use word type -based perplexity instead of the default ' +
                 'word token -based perplexity.')
    add_arg('--min-perplexity-length', dest='min_ppl_length', type=int,
            default=4, metavar='<int>',
            help='morphs shorter than this length are ' +
                 'ignored when calculating perplexity. ' +
                 '(default %(default)s).')
    add_arg('-d', '--dampening', dest="dampening", type=str, default='none',
            metavar='<type>', choices=['none', 'log', 'ones'],
            help="frequency dampening for training data ('none', 'log', or "
                 "'ones'; default '%(default)s').")
    add_arg('-f', '--forcesplit', dest="forcesplit", type=list, default=['-'],
            metavar='<list>',
            help="force split on given atoms. " +
            "Each character in the string will be included as a " +
            "forcesplit atom. " +
            "(default '-').")
    # FIXME not yet fully supported: segmentation may still split between these
    add_arg('--nosplit-re', dest="nosplit", type=str, default=None,
            metavar='<regexp>',
            help="if the expression matches the two surrounding characters, "
                 "do not allow splitting (default %(default)s)")
    add_arg('--remove-nonmorphemes', dest='rm_nonmorph', default=False,
            action='store_true',
            help='use heuristic postprocessing to remove nonmorfemes ' +
                 'from output segmentations.')
    add_arg('--nonmorpheme-heuristics', dest='heuristic_ops', type=str,
            default=','.join(HeuristicPostprocessor.DEFAULT_OPERATIONS),
            metavar='<list>',
            help='List of heuristics to use for removal of non-morfemes. ' +
                 'The format of the list is a string of (unquoted) ' +
                 'operation names separated by single commas (no space). ' +
                 'Has no effect unless used together with ' +
                 '--remove-nonmorphemes.' +
                 " (default '%(default)s').")
    add_arg('--max-heuristic-join-stem-length', dest='max_join_stem_len',
            type=int, default=4, metavar='<int>',
            help='Stems longer than this length are ' +
                 'not considered for heuristic joining with nonmorphemes. ' +
                 '(default %(default)s).')
    add_arg('--batch-minfreq', dest="freqthreshold", type=int, default=1,
            metavar='<int>',
            help="compound frequency threshold (default %(default)s).")
    add_arg('--max-shift-distance', dest='max_shift_distance',
            type=int, default=2, metavar='<int>',
            help='Maximum number of letters that the break between morphs ' +
                 'can move in the shift operation. ' +
                 '(default %(default)s).')
    add_arg('--min-shift-remainder', dest='min_shift_remainder',
            type=int, default=2, metavar='<int>',
            help='Minimum number of letters remaining in the shorter morph ' +
                 'after a shift operation. ' +
                 '(default %(default)s).')
#     add_arg('--viterbi-smoothing', dest="viterbismooth", default=0,
#             type=float, metavar='<float>',
#             help="additive smoothing parameter for Viterbi "
#             "segmentation (default %(default)s).")
#     add_arg('--viterbi-maxlen', dest="viterbimaxlen", default=30,
#             type=int, metavar='<int>',
#             help="maximum construction length in Viterbi "
#             "segmentation (default %(default)s).")

    # Options for controlling training iteration sequence
    add_arg = parser.add_argument_group(
        'training iteration sequence options').add_argument
    add_arg('--min-epoch-cost-gain', dest='min_epoch_cost_gain', type=float,
            default=0.0025, metavar='<float>',
            help='Stop iterating if cost reduction between epochs ' +
                 'is below this limit * #boundaries. ' +
                 '(default %(default)s).')
    add_arg('--min-iteration-cost-gain', dest='min_iter_cost_gain', type=float,
            default=0.005, metavar='<float>',
            help='Stop iterating if cost reduction between iterations ' +
                 'is below this limit * #boundaries. ' +
                 '(default %(default)s).')
    add_arg('--min-difference-proportion', dest='min_diff_prop', type=float,
            default=0.005, metavar='<float>',
            help='Stop iterating if proportion of words with changed ' +
                 'segmentation or category tags is below this limit. ' +
                 '(default %(default)s).')
    add_arg('--max-iterations', dest='max_iterations', type=int, default=7,
            metavar='<int>',
            help='Maximum number of iterations. (default %(default)s).')
    add_arg('--max-epochs-first', dest='max_epochs_first', type=int, default=1,
            metavar='<int>',
            help='Maximum number of epochs of each operation in ' +
                 'the first iteration. ' +
                 '(default %(default)s).')
    add_arg('--max-epochs', dest='max_epochs', type=int, default=1,
            metavar='<int>',
            help='Maximum number of epochs of each operation in ' +
                 'the subsequent iterations. ' +
                 '(default %(default)s).')
    add_arg('--max-resegment-epochs', dest='max_resegment_epochs',
            type=int, default=2, metavar='<int>',
            help='Maximum number of epochs of resegmentation in ' +
                 'all iterations. Resegmentation is the heaviest operation. ' +
                 '(default %(default)s).')
    add_arg('--training-operations', dest='training_operations', type=str,
            default=','.join(CatmapModel.DEFAULT_TRAIN_OPS), metavar='<list>',
            help='The sequence of training operations. ' +
                 'Valid training operations are strings for which ' +
                 'CatmapModel has a function named _op_X_generator. ' +
                 'The format of the list is a string of (unquoted) ' +
                 'operation names separated by single commas (no space). ' +
                 "(default '%(default)s').")
    add_arg('--online-epochint', dest="epochinterval", type=int,
            default=10000, metavar='<int>',
            help="epoch interval for online training (default %(default)s)")

    # Options for semi-supervised model training
    add_arg = parser.add_argument_group(
        'semi-supervised training options').add_argument
    add_arg('-A', '--annotations', dest="annofile", default=None,
            metavar='<file>',
            help="Load annotated data for semi-supervised learning.")
    add_arg('-D', '--develset', dest="develfile", default=None,
            metavar='<file>',
            help="Load annotated data for tuning the corpus weight parameter.")
    add_arg('--checkpoint', dest="checkpointfile",
            default='model.checkpoint.pickled', metavar='<file>',
            help="Save initialized model to file before weight learning. "
            "Has no effect unless --develset is given.")
    add_arg('-w', '--corpusweight', dest="corpusweight", type=float,
            default=1.0, metavar='<float>',
            help="Corpus weight parameter (default %(default)s); "
            "sets the initial value if --develset is used.")
    add_arg('--weightlearn-epochs-first', dest='weightlearn_epochs_first',
            type=int, default=5, metavar='<int>',
            help='Number of epochs of weight learning ' +
                 'in weight learning performed before the first training ' +
                 'iteration ' +
                 '(default %(default)s).')
    add_arg('--weightlearn-epochs', dest='weightlearn_epochs',
            type=int, default=3, metavar='<int>',
            help='Number of epochs of weight learning ' +
                 'in between-iteration weight updates ' +
                 '(default %(default)s).')
    add_arg('--weightlearn-depth-first', dest='weightlearn_depth_first',
            type=int, default=2, metavar='<int>',
            help='Number of times each training operation is performed' +
                 'in weight learning performed before the first training ' +
                 'iteration ' +
                 '(default %(default)s).')
    add_arg('--weightlearn-depth', dest='weightlearn_depth',
            type=int, default=1, metavar='<int>',
            help='Number of times each training operation is performed' +
                 'in between-iteration weight updates ' +
                 '(default %(default)s).')
    add_arg('--weightlearn-sample-size', dest='wlearn_sample_size', type=int,
            default=1000, metavar='<int>',
            help='A subset of this size is sampled (with repetition, ' +
            'weighting according to occurrence count) from the corpus. ' +
            'When evaluating a value for the corpus weight during weight '
            'learning, the local search of the model training is restricted ' +
            'to this set, to reduce computation time. ' +
            '(default %(default)s); ')
    add_arg('--weightlearn-sample-sets', dest='wlearn_sample_sets', type=int,
            default=3, metavar='<int>',
            help='Make a majority decision based on this number of ' +
            'weightlearning sample sets. ' +
            '(default %(default)s); ')
    add_arg('-W', '--annotationweight', dest="annotationweight",
            type=float, default=None, metavar='<float>',
            help="Corpus weight parameter for annotated data (if unset, the "
                 "weight is set to balance the number of tokens in annotated "
                 "and unannotated data sets).")

    # Options for logging
    add_arg = parser.add_argument_group('logging options').add_argument
    add_arg('-v', '--verbose', dest="verbose", type=int, default=1,
            metavar='<int>',
            help="verbose level; controls what is written to the standard "
                 "error stream or log file (default %(default)s).")
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help="write log messages to file in addition to standard "
            "error stream.")
    add_arg('--progressbar', dest='progress', default=False,
            action='store_true',
            help='Force the progressbar to be displayed.')
    add_arg('--statsfile', dest='stats_file', metavar='<file>',
            help='Collect iteration statistics and pickle them ' +
                 'into this file.')
    add_arg('--stats-annotations', dest="statsannotfile", default=None,
            metavar='<file>',
            help='Load annotated data for f-measure diagnostics. '
                 'Useful for analyzing convergence properties.')

    add_arg = parser.add_argument_group('other options').add_argument
    add_arg('-h', '--help', action='help',
            help="show this help message and exit.")
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help="show version number and exit.")

    return parser


def catmap_main(args):
    # FIXME contains lots of copy-pasta from morfessor.main (refactor)
    if args.verbose >= 2:
        loglevel = logging.DEBUG
    elif args.verbose >= 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    logging_format = '%(asctime)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    default_formatter = logging.Formatter(logging_format, date_format)
    plain_formatter = logging.Formatter('%(message)s')
    logging.basicConfig(level=loglevel)
    _logger.propagate = False  # do not forward messages to the root logger

    # Basic settings for logging to the error stream
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(plain_formatter)
    _logger.addHandler(ch)

    # Settings for when log_file is present
    if args.log_file is not None:
        fh = logging.FileHandler(args.log_file, 'w')
        fh.setLevel(loglevel)
        fh.setFormatter(default_formatter)
        _logger.addHandler(fh)
        # If logging to a file, make INFO the highest level for the
        # error stream
        ch.setLevel(max(loglevel, logging.INFO))

    # If debug messages are printed to screen or if stderr is not a tty (but
    # a pipe or a file), don't show the progressbar
    global show_progress_bar
    if (ch.level > logging.INFO or
            (hasattr(sys.stderr, 'isatty') and not sys.stderr.isatty())):
        show_progress_bar = False

    if args.progress:
        show_progress_bar = True
        ch.setLevel(min(ch.level, logging.INFO))

    # Set frequency dampening function
    if args.dampening == 'none':
        dampfunc = None
    elif args.dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif args.dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % args.dampening)
    # FIXME everything directly pasted up to this point

    if (args.loadfile is None and
            len(args.baselinefiles) == 0 and
            len(args.loadsegfiles) == 0):
        raise ArgumentException('either model file, '
            'tagged segmentation or baseline segmentation must be defined.')

    io = CatmapIO(encoding=args.encoding,
                  compound_separator=args.cseparator,
                  category_separator=args.catseparator)

    # Load exisiting model or create a new one
    model_initialized = False
    training_ops = args.training_operations.split(',')
    if args.loadfile is not None:
        model = io.read_binary_model_file(args.loadfile)
        model_initialized = True
    else:
        m_usage = MorphUsageProperties(
            ppl_threshold=args.ppl_threshold,
            ppl_slope=args.ppl_slope,
            length_threshold=args.length_threshold,
            length_slope=args.length_slope,
            use_word_tokens=not args.type_ppl,
            min_perplexity_length=args.min_ppl_length)
        model = CatmapModel(m_usage, forcesplit=args.forcesplit,
                            nosplit=args.nosplit,
                            corpusweight=args.corpusweight)

    # Set up statistics logging
    stats = None
    if args.stats_file is not None:
        stats = IterationStatistics()
        model.epoch_callbacks.append(stats.callback)
        stats.set_names(model, training_ops)

        if args.statsannotfile is not None:
            stats.set_gold_standard(
                io.read_annotations_file(args.statsannotfile,
                    analysis_sep=args.analysisseparator))

    # Load data
    for f in args.loadsegfiles:
        _logger.info('Calling model.add_corpus_data')
        model.add_corpus_data(io.read_segmentation_file(f),
                              count_modifier=dampfunc,
                              freqthreshold=args.freqthreshold)
        _logger.info('Done with model.add_corpus_data')
    if (not model_initialized and
            len(args.loadsegfiles) > 0 and
            len(args.baselinefiles) > 0):
        # Starting from both tagged and untagged segmentation files,
        # but no trained model: have to initialize from the tagging.
        model.reestimate_probabilities()
        model.initialize_hmm(min_difference_proportion=args.min_diff_prop)
        model_initialized = True
    for f in args.baselinefiles:
        _logger.info('Calling model.add_corpus_data')
        model.add_corpus_data(io.read_segmentation_file(f),
                              count_modifier=dampfunc,
                              freqthreshold=args.freqthreshold)
        _logger.info('Done with model.add_corpus_data')

    if args.annofile is not None:
        annotations = io.read_annotations_file(args.annofile,
            analysis_sep=args.analysisseparator)
        model.add_annotations(annotations,
                              args.annotationweight)

    if args.develfile is not None:
        develannots = io.read_annotations_file(args.develfile,
            analysis_sep=args.analysisseparator)
    else:
        develannots = None

    if args.loadparamsfile is not None:
        _logger.info('Loading learned params from {}'.format(
            args.loadparamsfile))
        model.set_learned_params(
            io.read_parameter_file(args.loadparamsfile))

    # Initialize the model
    must_train = False
    if not model_initialized:
        # Starting from segmentations instead of pickle,
        model.training_operations = training_ops
        # Need to (re)estimate the probabilities
        if len(args.loadsegfiles) == 0:
            # Starting from a baseline model
            _logger.info('Initializing from baseline segmentation...')
            model.initialize_baseline()
            must_train = True
        else:
            model.reestimate_probabilities()
        model.initialize_hmm(min_difference_proportion=args.min_diff_prop)
    elif len(args.baselinefiles) > 0 or len(args.loadsegfiles) > 0:
        # Extending initialized model with new data
        model.viterbi_tag_corpus()
        model.initialize_hmm(min_difference_proportion=args.min_diff_prop)
        must_train = True

    # Heuristic nonmorpheme removal
    heuristic = None
    if args.rm_nonmorph:
        heuristic_ops = args.heuristic_ops.split(',')
        heuristic = HeuristicPostprocessor(operations=heuristic_ops,
                        max_join_stem_len=args.max_join_stem_len)

    # Perform weight learning using development annotations
    if develannots is not None:
        corpus_weight_updater = CorpusWeightUpdater(
            develannots,
            heuristic,
            io,
            args.checkpointfile,
            args.weightlearn_epochs_first,
            args.weightlearn_epochs,
            args.weightlearn_depth_first,
            args.weightlearn_depth)
        weight_learn_func = corpus_weight_updater.weight_learning
        model.generate_focus_samples(
            args.wlearn_sample_sets,
            args.wlearn_sample_size)

        _logger.info('Performing initial weight learning')
        (model, must_train) = corpus_weight_updater.weight_learning(
            model)

        model.training_focus = None
    else:
        weight_learn_func = None

    # Train model, if there is new data to train on
    if args.trainmode == 'none':
        if must_train and len(args.testfiles) > 0:
            raise ArgumentException('Must train before using a model '
                'for segmenting, if new data is added.')
        _logger.info('Using loaded model without training')
    if args.trainmode in ('online', 'online+batch'):
        # Always reads from stdin
        data = io.read_corpus_files('-')
        model.train_online(data, count_modifier=dampfunc,
                           epoch_interval=args.epochinterval,
                           max_epochs=args.max_epochs)
    if args.trainmode in ('batch', 'online+batch'):
        model.batch_parameters(min_epoch_cost_gain=args.min_epoch_cost_gain,
                               min_iter_cost_gain=args.min_iter_cost_gain,
                               max_iterations=args.max_iterations,
                               max_epochs_first=args.max_epochs_first,
                               max_epochs=args.max_epochs,
                               max_resegment_epochs=args.max_resegment_epochs,
                               max_shift_distance=args.max_shift_distance,
                               min_shift_remainder=args.min_shift_remainder)
        ts = time.time()
        model = train_batch(model, weight_learn_func)
        _logger.info('Final cost: {}'.format(model.get_cost()))
        te = time.time()
        _logger.info('Training time: {:.3f}s'.format(te - ts))

    # Save model
    if args.savefile is not None:
        model.toggle_callbacks(None)
        io.write_binary_model_file(args.savefile, model)

    if args.savesegfile is not None:
        if heuristic is not None:
            segs = model.map_segmentations(
                lambda x: heuristic.remove_nonmorfemes(x, model))
        else:
            segs = model.segmentations
        io.write_segmentation_file(args.savesegfile, segs)

    if args.saveparamsfile is not None:
        io.write_parameter_file(args.saveparamsfile,
                                model.get_learned_params())

    # Segment test data
    if len(args.testfiles) > 0:
        _logger.info("Segmenting test data...")
        outformat = args.outputformat
        csep = args.outputformatseparator
        if not PY3:
            outformat = unicode(outformat)
            csep = unicode(csep)
        outformat = outformat.replace(r"\n", "\n")
        outformat = outformat.replace(r"\t", "\t")
        with io._open_text_file_write(args.outfile) as fobj:
            testdata = io.read_corpus_files(args.testfiles)
            for count, compound, atoms in _generator_progress(testdata):
                constructions, logp = model.viterbi_segment(atoms)
                if heuristic is not None:
                    constructions = heuristic.remove_nonmorfemes(
                                        constructions, model)
                if args.test_output_tags:
                    def _output_morph(cmorph):
                        return '{}{}{}'.format(cmorph.morph,
                                                args.catseparator,
                                                cmorph.category)
                else:
                    def _output_morph(cmorph):
                        return cmorph.morph
                constructions = [_output_morph(cmorph)
                                 for cmorph in constructions]
                analysis = csep.join(constructions)
                fobj.write(outformat.format(
                           analysis=analysis, compound=compound,
                           count=count, logprob=logp))
        _logger.info("Done.")

    # Save statistics
    if args.stats_file is not None:
        io.write_binary_file(args.stats_file, stats)
