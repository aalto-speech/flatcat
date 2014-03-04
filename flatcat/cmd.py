from __future__ import unicode_literals

import logging
import math
import random
import os.path
import sys
import time
import string

from . import get_version, _logger, flatcat
from .categorizationscheme import MorphUsageProperties, HeuristicPostprocessor
from .diagnostics import IterationStatistics
from .exception import ArgumentException
from .io import FlatcatIO
from .utils import _generator_progress, LOGPROB_ZERO, memlog

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


class ArgumentGroups(object):
    def __init__(self, parser):
        self.parser = parser
        self._groups = {}

    def get(self, name):
        if name not in self._groups:
            self._groups[name] = (
                self.parser.add_argument_group(name).add_argument)
        return self._groups[name]


def get_flatcat_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='flatcat.py',
        description="""
Morfessor FlatCat {version}

Copyright (c) 2013, Stig-Arne Gronroos
All rights reserved.
{license}

Command-line arguments:
""" .format(version=get_version(), license=LICENSE),
        epilog="""
Simple usage examples (training and testing):

  %(prog)s -B baseline_segmentation.txt -p 10 -s model.pickled
  %(prog)s -m none -l model.pickled -T test_corpus.txt -o test_corpus.segmented

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    groups = ArgumentGroups(parser)
    add_model_io_arguments(groups)
    add_common_io_arguments(groups)
    add_training_arguments(groups)
    add_semisupervised_arguments(groups)
    add_weightlearning_arguments(groups)
    add_other_arguments(groups)
    return parser


def add_model_io_arguments(argument_groups):
    # Options for input data files
    add_arg = argument_groups.get('input data files')
    add_arg('-i', '--initialize', dest='initfile',
            default=None, metavar='<file>',
            help='Initialize by loading model from file. '
                 'Supported formats: '
                 'Untagged segmentation '
                 '(Morfessor Baseline; plaintext, ".gz" or ."bz2"), '
                 'Tagged analysis '
                 '(Morfessor FlatCat; plaintext, ".gz" or ".bz2"), '
                 'Binary FlatCat model (pickled in a ".pickled" file)')
    add_arg('--extend', dest='extendfiles', default=[],
            action='append', metavar='<file>',
            help='Extend the model using the segmentation from a file. '
                 'The supported formats are the same as for --initialize, '
                 'except thet pickled binary models are not supported. '
                 'Untagged segmentations will be tagged with the current '
                 'model.')
    add_arg('-T', '--testdata', dest='testfiles', action='append',
            default=[], metavar='<file>',
            help='Input corpus file(s) to analyze (text or gzipped text;  '
                 'use "-" for standard input; add several times in order to '
                 'append multiple files).')

    # Options for output data files
    add_arg = argument_groups.get('output data files')
    add_arg('-s', '--save-analysis', dest='saveanalysisfile',
            default=None, metavar='<file>',
            help='Save analysis of the unannotated data. ')
    add_arg('--save-annotations', dest='saveannotsfile',
            default=None, metavar='<file>',
            help='Save the annotated data set.')
    add_arg('--save-binary-model', dest='savepicklefile',
            default=None, metavar='<file>',
            help='Save a binary FlatCat model with pickle. '
                 'Use of a filename ending in ".pickled" is recommended '
                 'This format is suceptible to bit-rot, '
                 'and is not recommended for long-time storage.')
    add_arg('-o', '--output', dest='outfile', default='-', metavar='<file>',
            help='Output file for test data results (for standard output, '
                 'use "-"; default "%(default)s").')


def add_common_io_arguments(argument_groups):
    # Options for data formats
    add_arg = argument_groups.get('data format options')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')
    add_arg('--compound-separator', dest='cseparator', type=str, default='\s+',
            metavar='<regexp>',
            help='Compound separator regexp (default "%(default)s").')
    add_arg('--construction-separator', dest='consseparator', type=str,
            default=' + ', metavar='<regexp>',
            help='Compound separator regexp (default "%(default)s").')
    add_arg('--analysis-separator', dest='analysisseparator', type=str,
            default=',', metavar='<regexp>',
            help='Separator for different analyses in an annotation file. Use '
                 'NONE for only allowing one analysis per line.')
    add_arg('--category-separator', dest='catseparator', type=str, default='/',
            metavar='<regexp>',
            help='separator for the category tag following a morph. '
                 '(default %(default)s).')
    add_arg('--output-format', dest='outputformat', type=str,
            default=r'{analysis}\n', metavar='<format>',
            help='Format string for --output file (default: "%(default)s"). '
                 'Valid keywords are: '
                 '{analysis} = morphs of the word, '
                 '{compound} = word, '
                 '{count} = count of the word (currently always 1), and '
                 '{logprob} = log-probability of the analysis. Valid escape '
                 'sequences are "\\n" (newline) and "\\t" (tabular)')
    add_arg('--output-format-separator', dest='outputformatseparator',
            type=str, default=' ', metavar='<str>',
            help='Construction separator for analysis in --output file '
                 '(default: "%(default)s")')
    add_arg('--output-tags', dest='test_output_tags', default=False,
            action='store_true',
            help='Output category tags in test data. '
                 'Default is to output only the morphs')
    add_arg('--output-newlines', dest='outputnewlines', default=False,
            action='store_true',
            help='For each newline in input, print newline in --output file '
            '(default: "%(default)s")')

    # Output post-processing
    add_arg = argument_groups.get('output post-processing options')
    add_arg('--filter-categories', dest='filter_categories', type=str,
            default='', metavar='<list>',
            help='A list of morph categories to omit from the output '
                 'of the test data. Can be used e.g. for approximating '
                 'stemming by specifying "PRE,SUF" or "SUF". '
                 'The format of the list is a string of (unquoted) '
                 'category tags separated by single commas (no space). '
                 'Default: do not filter any categories.')
    add_arg('--remove-nonmorphemes', dest='rm_nonmorph', default=False,
            action='store_true',
            help='Use heuristic postprocessing to remove nonmorphemes '
                 'from output segmentations.')

    # Options for semi-supervised model training
    add_arg = argument_groups.get('semi-supervised training options')
    add_arg('-A', '--annotations', dest='annofiles', default=[],
            action='append', metavar='<file>',
            help='Load annotated data for semi-supervised learning.')
    add_arg('-D', '--develset', dest='develfile', default=None,
            metavar='<file>',
            help='Load annotated data for tuning the corpus weight parameter.')


def add_training_arguments(argument_groups):
    # Options for input data files
    add_arg = argument_groups.get('input data files')
    add_arg('-L', '--load-parameters', dest='loadparamsfile', default=None,
            metavar='<file>',
            help='Load hyperparameters from file.')
    # Options for output data files
    add_arg = argument_groups.get('output data files')
    add_arg('-S', '--save-parameters', dest='saveparamsfile', default=None,
            metavar='<file>',
            help='Save hyperparameters to file.')
    # Options for training and segmentation
    add_arg = argument_groups.get('training and segmentation options')
    add_arg('-m', '--mode', dest='trainmode', default='batch',
            metavar='<mode>',
            choices=['none', 'batch', 'online', 'online+batch'],
            help='Training mode ("none", "batch", '
                 '"online", or "online+batch"; default "%(default)s")')
    add_arg('-p', '--perplexity-threshold', dest='ppl_threshold', type=float,
            default=100., metavar='<float>',
            help='Threshold value for sigmoid used to calculate '
                 'probabilities from left and right perplexities. '
                 '(default %(default)s).')
    add_arg('--perplexity-slope', dest='ppl_slope', type=float, default=None,
            metavar='<float>',
            help='Slope value for sigmoid used to calculate '
                 'probabilities from left and right perplexities. '
                 '(default 10 / perplexity-threshold).')
    add_arg('--length-threshold', dest='length_threshold', type=float,
            default=3., metavar='<float>',
            help='Threshold value for sigmoid used to calculate '
                 'probabilities from length of morph. '
                 '(default %(default)s).')
    add_arg('--length-slope', dest='length_slope', type=float, default=2.,
            metavar='<float>',
            help='Slope value for sigmoid used to calculate '
                 'probabilities from length of morph. '
                 '(default %(default)s).')
    add_arg('--type-perplexity', dest='type_ppl', default=False,
            action='store_true',
            help='Use word type -based perplexity instead of the default '
                 'word token -based perplexity.')
    add_arg('--min-perplexity-length', dest='min_ppl_length', type=int,
            default=4, metavar='<int>',
            help='Morphs shorter than this length are '
                 'ignored when calculating perplexity. '
                 '(default %(default)s).')
    add_arg('-d', '--dampening', dest='dampening', type=str, default='none',
            metavar='<type>', choices=['none', 'log', 'ones'],
            help='Frequency dampening for training data. '
                 'Do not apply dampening if the baseline was already '
                 'dampened: the effect is cumulative. '
                 '("none", "log", or "ones"; default "%(default)s").')
    add_arg('-f', '--forcesplit', dest='forcesplit', type=list, default=['-'],
            metavar='<list>',
            help='Force split on given atoms. '
                 'Each character in the string will be included as a '
                 'forcesplit atom. '
                 '(default "-").')
    add_arg('--nosplit-re', dest='nosplit', type=str, default=None,
            metavar='<regexp>',
            help='If the expression matches the two surrounding characters, '
                 'do not allow splitting (default %(default)s)')
    add_arg('--skips', dest='skips', default=False, action='store_true',
            help='Use random skips for frequently seen words to speed up '
                 'online training. Has no effect on batch training.')
    add_arg('--batch-minfreq', dest='freqthreshold', type=int, default=1,
            metavar='<int>',
            help='Word frequency threshold (default %(default)s).')
    add_arg('--max-shift-distance', dest='max_shift_distance',
            type=int, default=2, metavar='<int>',
            help='Maximum number of letters that the break between morphs '
                 'can move in the shift operation. '
                 '(default %(default)s).')
    add_arg('--min-shift-remainder', dest='min_shift_remainder',
            type=int, default=2, metavar='<int>',
            help='Minimum number of letters remaining in the shorter morph '
                 'after a shift operation. '
                 '(default %(default)s).')

    # Options for controlling training iteration sequence
    add_arg = argument_groups.get('training iteration sequence options')
    add_arg('--max-epochs', dest='max_epochs', type=int, default=7,
            metavar='<int>',
            help='Maximum number of epochs. (default %(default)s).')
    add_arg('--max-iterations-first', dest='max_iterations_first',
            type=int, default=1, metavar='<int>',
            help='Maximum number of iterations of each operation in '
                 'the first epoch. '
                 '(default %(default)s).')
    add_arg('--max-iterations', dest='max_iterations', type=int, default=1,
            metavar='<int>',
            help='Maximum number of iterations of each operation in '
                 'the subsequent epochs. '
                 '(default %(default)s).')
    add_arg('--max-resegment-iterations', dest='max_resegment_iterations',
            type=int, default=2, metavar='<int>',
            help='Maximum number of iterations of resegmentation in '
                 'all epochs. Resegmentation is the heaviest operation. '
                 '(default %(default)s).')
    add_arg('--min-epoch-cost-gain', dest='min_epoch_cost_gain', type=float,
            default=0.005, metavar='<float>',
            help='Stop training if cost reduction between epochs '
                 'is below this limit * #boundaries. '
                 '(default %(default)s).')
    add_arg('--min-iteration-cost-gain', dest='min_iteration_cost_gain',
            type=float, default=0.0025, metavar='<float>',
            help='Stop training if cost reduction between iterations '
                 'is below this limit * #boundaries. '
                 '(default %(default)s).')
    add_arg('--min-difference-proportion', dest='min_diff_prop', type=float,
            default=0.005, metavar='<float>',
            help='Stop training if proportion of words with changed '
                 'segmentation or category tags is below this limit. '
                 '(default %(default)s).')
    add_arg('--training-operations', dest='training_operations', type=str,
            default=','.join(flatcat.FlatcatModel.DEFAULT_TRAIN_OPS),
            metavar='<list>',
            help='The sequence of training operations. '
                 'Valid training operations are strings for which '
                 'FlatcatModel has a function named _op_X_generator. '
                 'The format of the list is a string of (unquoted) '
                 'operation names separated by single commas (no space). '
                 '(default "%(default)s").')
    add_arg('--online-epochint', dest='epochinterval', type=int,
            default=10000, metavar='<int>',
            help='Epoch interval for online training (default %(default)s)')


def add_semisupervised_arguments(argument_groups):
    # Options for semi-supervised model training
    add_arg = argument_groups.get('semi-supervised training options')
    add_arg('-w', '--corpusweight', dest='corpusweight', type=float,
            default=1.0, metavar='<float>',
            help='Corpus weight parameter (default %(default)s); '
                 'sets the initial value if --develset is used.')
    add_arg('-W', '--annotationweight', dest='annotationweight',
            type=float, default=None, metavar='<float>',
            help='Corpus weight parameter for annotated data (if unset, the '
                 'weight is set to balance the number of tokens in annotated '
                 'and unannotated data sets).')


def add_weightlearning_arguments(argument_groups):
    # Options for automatically setting the weight parameters
    add_arg = argument_groups.get('weight learning options')
    add_arg('--checkpoint', dest='checkpointfile',
            default='model.checkpoint.pickled', metavar='<file>',
            help='Save initialized model to file before weight learning. '
                 'Has no effect unless --develset is given.')
    add_arg('--weightlearn-parameters', dest='weightlearn_params', type=str,
            default='annotationweight,corpusweight',
            metavar='<list>',
            help='The sequence of parameters to optimize '
                 'in weight learning. '
                 'The format of the list is a string of (unquoted) '
                 'parameter names separated by single commas (no space). '
                 '(default "%(default)s").')
    add_arg('--weightlearn-iters-first', dest='weightlearn_iters_first',
            type=int, default=2, metavar='<int>',
            help='Number of iterations of weight learning '
                 'in weight learning performed before the first training '
                 'epoch '
                 '(default %(default)s).')
    add_arg('--weightlearn-iters', dest='weightlearn_iters',
            type=int, default=1, metavar='<int>',
            help='Number of iterations of weight learning '
                 'in between-iteration weight updates '
                 '(default %(default)s).')
    add_arg('--weightlearn-evals-first', dest='weightlearn_evals_first',
            type=int, default=5, metavar='<int>',
            help='Number of objective function evaluations per line search '
                 'in weight learning performed before the first training '
                 'iteration. '
                 'Each function evaluation consists of partially training '
                 'the model weightlearn-sample-sets times '
                 '(default %(default)s).')
    add_arg('--weightlearn-evals', dest='weightlearn_evals',
            type=int, default=3, metavar='<int>',
            help='Number of objective function evaluations per line search '
                 'in between-iteration weight updates '
                 '(default %(default)s).')
    add_arg('--weightlearn-depth-first', dest='weightlearn_depth_first',
            type=int, default=2, metavar='<int>',
            help='Number of times each training operation is performed'
                 'in weight learning performed before the first training '
                 'iteration '
                 '(default %(default)s).')
    add_arg('--weightlearn-depth', dest='weightlearn_depth',
            type=int, default=1, metavar='<int>',
            help='Number of times each training operation is performed'
                 'in between-iteration weight updates '
                 '(default %(default)s).')
    add_arg('--weightlearn-sample-size', dest='weightlearn_sample_size',
            type=int, default=2000, metavar='<int>',
            help='A subset of this size is sampled (with repetition, '
                 'weighting according to occurrence count) from the corpus. '
                 'When evaluating a value for the corpus weight during weight '
                 'learning, the local search of the model training '
                 'is restricted to this set, to reduce computation time. '
                 'Setting this to zero uses the whole corpus.'
                 '(default %(default)s); ')
    add_arg('--weightlearn-sample-sets', dest='weightlearn_sample_sets',
            type=int, default=5, metavar='<int>',
            help='Make a majority decision based on this number of '
                 'weightlearning sample sets. '
                 'If the whole corpus is used for weight learning, '
                 'this parameter has no effect. '
                 '(default %(default)s); ')
    add_arg('--weightlearn-cue-rejection-thresh', dest='weightlearn_cuethresh',
            type=int, default=4, metavar='<int>',
            help='Stop using the direction cue after this many rejected steps '
                 '(default %(default)s); ')


def add_other_arguments(argument_groups):
    # Options for logging
    add_arg = argument_groups.get('logging options')
    add_arg('-v', '--verbose', dest='verbose', type=int, default=1,
            metavar='<int>',
            help='Level of verbosity; controls what is written to '
                 'the standard error stream or log file '
                 '(default %(default)s).')
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help='Write log messages to file in addition to standard '
                 'error stream.')
    add_arg('--progressbar', dest='progress', default=False,
            action='store_true',
            help='Force the progressbar to be displayed.')
    add_arg('--statsfile', dest='stats_file', metavar='<file>',
            help='Collect iteration statistics and pickle them '
                 'into this file.')
    add_arg('--stats-annotations', dest='statsannotfile', default=None,
            metavar='<file>',
            help='Load annotated data for f-measure diagnostics. '
                 'Useful for analyzing convergence properties.')

    add_arg = argument_groups.get('other options')
    add_arg('-h', '--help', action='help',
            help='Show this help message and exit.')
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help='Show version number and exit.')


def flatcat_main(args):
    # FIXME contains lots of copy-pasta from morfessor.cmd.main (refactor)
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

    # Arguments needing processing
    training_ops = args.training_operations.split(',')

    if (args.initfile is None):
        raise ArgumentException(
            'An initial Baseline or FlatCat model must be given.')
    init_is_pickle = (args.initfile.endswith('.pickled') or
                      args.initfile.endswith('.pickle'))

    io = FlatcatIO(encoding=args.encoding,
                   construction_separator=args.consseparator,
                   compound_separator=args.cseparator,
                   category_separator=args.catseparator)

    # Load exisiting model or create a new one
    must_train = False
    if init_is_pickle:
        _logger.info('Initializing from binary model...')
        shared_model = flatcat.SharedModel(
            io.read_binary_model_file(args.initfile))
        shared_model.model.post_load()
    else:
        _logger.info('Initializing from segmentation...')
        m_usage = MorphUsageProperties(
            ppl_threshold=args.ppl_threshold,
            ppl_slope=args.ppl_slope,
            length_threshold=args.length_threshold,
            length_slope=args.length_slope,
            use_word_tokens=not args.type_ppl,
            min_perplexity_length=args.min_ppl_length)
        # Make sure that the current model can be garbage collected
        # if it needs to be reloaded from disk
        shared_model = flatcat.SharedModel(flatcat.FlatcatModel(
                            m_usage,
                            forcesplit=args.forcesplit,
                            nosplit=args.nosplit,
                            corpusweight=args.corpusweight,
                            use_skips=args.skips))
        # Add the initial corpus data
        shared_model.model.add_corpus_data(
            io.read_segmentation_file(args.initfile),
            count_modifier=dampfunc,
            freqthreshold=args.freqthreshold)

        # Load the hyperparamters
        shared_model.model.training_operations = training_ops
        if args.loadparamsfile is not None:
            _logger.info('Loading hyperparameters from {}'.format(
                args.loadparamsfile))
            shared_model.model.set_params(
                io.read_parameter_file(args.loadparamsfile))

    # Add annotated data
    for f in args.annofiles:
        annotations = io.read_annotations_file(f,
            analysis_sep=args.analysisseparator)
        shared_model.model.add_annotations(annotations,
                              args.annotationweight)

    if not init_is_pickle:
        # Initialize the model
        must_train = shared_model.model.initialize_hmm(
            min_difference_proportion=args.min_diff_prop)

    if len(args.annofiles) > 0:
        shared_model.model.viterbi_tag_corpus()
        shared_model.model.reestimate_probabilities()
        shared_model.model._update_annotation_choices()

    if args.develfile is not None:
        develannots = io.read_annotations_file(args.develfile,
            analysis_sep=args.analysisseparator)
    else:
        develannots = None

    # Extend the model with new unannotated data
    for f in args.extendfiles:
        shared_model.model.add_corpus_data(io.read_segmentation_file(f),
                              count_modifier=dampfunc,
                              freqthreshold=args.freqthreshold)
    if len(args.extendfiles) > 0:
        shared_model.model.corpus_extended()
        must_train = True

    # Set up statistics logging
    stats = None
    if args.stats_file is not None:
        stats = IterationStatistics()
        shared_model.model.iteration_callbacks.append(stats.callback)
        stats.set_names(shared_model.model, training_ops)

        if args.statsannotfile is not None:
            stats.set_gold_standard(
                io.read_annotations_file(args.statsannotfile,
                    analysis_sep=args.analysisseparator))

    # Heuristic nonmorpheme removal
    heuristic = None
    if args.rm_nonmorph:
        heuristic = HeuristicPostprocessor()

    # Perform weight learning using development annotations
    if develannots is not None:
        weight_learning = flatcat.WeightLearning(
            args.weightlearn_iters_first,
            args.weightlearn_evals_first,
            args.weightlearn_depth_first,
            args.weightlearn_iters,
            args.weightlearn_evals,
            args.weightlearn_depth,
            args.weightlearn_cuethresh,
            develannots,
            shared_model,
            io,
            args.checkpointfile,
            heuristic)
        for param in args.weightlearn_params.split(','):
            if param == 'corpusweight':
                weight_learning.add_corpus_weight()
            elif param == 'annotationweight':
                if shared_model.model._supervised:
                    weight_learning.add_annotation_weight()
        shared_model.model.generate_focus_samples(
            args.weightlearn_sample_sets,
            args.weightlearn_sample_size)

        _logger.info('Performing initial weight learning')
        must_train = weight_learning.optimize(first=True)

        shared_model.model.training_focus = None
    else:
        weight_learning = None

    # Train model, if there is new data to train on
    if args.trainmode == 'none':
        if must_train and len(args.testfiles) > 0:
            raise ArgumentException('Must train before using a model '
                'for segmenting, if new data is added.')
        _logger.info('Using loaded model without training')
    if args.trainmode in ('online', 'online+batch'):
        # Always reads from stdin
        data = io.read_combined_file('-',
                                     annotation_prefix='<',
                                     construction_sep=' ',
                                     analysis_sep=',')
        shared_model.model.train_online(data, count_modifier=dampfunc,
                           epoch_interval=args.epochinterval,
                           max_epochs=(args.max_iterations * args.max_epochs))
    if args.trainmode in ('batch', 'online+batch'):
        shared_model.model.batch_parameters(
                        min_iteration_cost_gain=args.min_iteration_cost_gain,
                        min_epoch_cost_gain=args.min_epoch_cost_gain,
                        max_epochs=args.max_epochs,
                        max_iterations_first=args.max_iterations_first,
                        max_iterations=args.max_iterations,
                        max_resegment_iterations=args.max_resegment_iterations,
                        max_shift_distance=args.max_shift_distance,
                        min_shift_remainder=args.min_shift_remainder)
        ts = time.time()
        flatcat.train_batch(shared_model, weight_learning)
        _logger.info('Final cost: {}'.format(shared_model.model.get_cost()))
        te = time.time()
        _logger.info('Training time: {:.3f}s'.format(te - ts))

    #
    # Save hyperparameters
    if args.saveparamsfile is not None:
        io.write_parameter_file(args.saveparamsfile,
                                shared_model.model.get_params())

    # Save analysis
    if args.saveanalysisfile is not None:
        io.write_segmentation_file(args.saveanalysisfile,
                                   shared_model.model.segmentations)

    # Save annotations
    #FIXME

    # Save binary model
    if args.savepicklefile is not None:
        shared_model.model.toggle_callbacks(None)
        memlog('Before pickle')
        shared_model.model.pre_save()
        io.write_binary_model_file(args.savepicklefile, shared_model.model)
        memlog('After pickle')
        if len(args.testfiles) > 0:
            shared_model.model.post_load()
            memlog('After postload')

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
        if len(args.filter_categories) > 0:
            filter_tags = [x.upper()
                           for x in args.filter_categories.split(',')]
        else:
            filter_tags = None
        with io._open_text_file_write(args.outfile) as fobj:
            testdata = io.read_corpus_files(args.testfiles)
            for count, compound, atoms in _generator_progress(testdata):
                if len(atoms) == 0:
                    # Newline in corpus
                    if args.outputnewlines:
                        fobj.write("\n")
                    continue
                constructions, logp = shared_model.model.viterbi_segment(atoms)
                if heuristic is not None:
                    constructions = heuristic.remove_nonmorphemes(
                                        constructions, shared_model.model)
                if args.test_output_tags:
                    def _output_morph(cmorph):
                        return '{}{}{}'.format(cmorph.morph,
                                                args.catseparator,
                                                cmorph.category)
                else:
                    def _output_morph(cmorph):
                        return cmorph.morph
                if filter_tags is not None:
                    constructions = [cmorph for cmorph in constructions
                                     if cmorph.category not in filter_tags]
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
