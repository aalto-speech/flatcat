from __future__ import unicode_literals

import collections
import locale
import logging
import math
import os
import string
import sys
import time

from morfessor import evaluation as bleval
from morfessor.io import MorfessorIO

from . import get_version, _logger, flatcat, reduced
from . import categorizationscheme, utils
from .diagnostics import IterationStatistics
from .exception import ArgumentException
from .io import FlatcatIO, TarGzModel, BINARY_ENDINGS, TARBALL_ENDINGS
from .utils import _generator_progress

PY3 = sys.version_info.major == 3

# _str is used to convert command line arguments to the right type
# (str for PY3, unicode for PY2)
if PY3:
    _str = str
else:
    _str = lambda x: unicode(x, encoding=locale.getpreferredencoding())

_logger = logging.getLogger(__name__)

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

_preferred_encoding = locale.getpreferredencoding()

DEFAULT_CORPUSWEIGHT = 1.0


class ArgumentGroups(object):
    """Helper class for modular sharing of arguments."""
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
        prog='flatcat',
        description="""
Morfessor {version}

{license}

Command-line arguments:
""" .format(version=get_version(), license=LICENSE),
        epilog="""
Simple usage examples (training and testing):

  %(prog)s baseline_segmentation.txt -p 10 -s analysis.tar.gz
  %(prog)s analysis.tar.gz -m none --remove-nonmorphemes \\
        -T test_corpus.txt -o test_corpus.segmented

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    parser.add_argument('initfile',
        metavar='<init file>',
        help='Initialize by loading model from file. '
                'Supported formats: '
                'Untagged segmentation '
                '(Morfessor Baseline; plaintext, ".gz" or ."bz2"), '
                'Tagged analysis '
                '(Morfessor FlatCat; plaintext, ".gz" or ".bz2"), '
                'Binary FlatCat model (pickled in a ".pickled" file)')
    groups = ArgumentGroups(parser)
    add_model_io_arguments(groups)
    add_common_io_arguments(groups)
    add_training_arguments(groups)
    add_semisupervised_arguments(groups)
    #add_weightlearning_arguments(groups)
    add_diagnostic_arguments(groups)
    add_other_arguments(groups)
    return parser


def add_model_io_arguments(argument_groups):
    # Options for input data files
    add_arg = argument_groups.get('input data files')
    add_arg('--extend', dest='extendfiles', default=[],
            action='append', metavar='<file>',
            help='Extend the model using the segmentation from a file. '
                 'The supported formats are the same as for the '
                 'initialization positional argument, '
                 'except that pickled binary models are not supported. '
                 'Untagged segmentations will be tagged with the current '
                 'model.')
    add_arg('-T', '--testdata', dest='testfiles', action='append',
            default=[], metavar='<file>',
            help='Input corpus file(s) to analyze (text or gzipped text;  '
                 'use "-" for standard input; add several times in order to '
                 'append multiple files).')

    # Options for output data files
    add_arg = argument_groups.get('output data files')
    add_arg('-s', '--save-tarball', dest='savetarballfile',
            default=None, metavar='<file>',
            help='Save model in .tar.gz format. ')
    add_arg('--save-binary-model', dest='savepicklefile',
            default=None, metavar='<file>',
            help='Save a binary FlatCat model with pickle. '
                 'Use of a filename ending in ".pickled" is recommended. '
                 'This format is suceptible to bit-rot, '
                 'and is not recommended for long-time storage.')
    add_arg('--save-reduced', dest="savereduced", default=None,
            metavar='<file>',
            help="save final model to file in reduced form (pickled model "
                 "object). A model in reduced form can only be used for "
                 "segmentation of new words.")
    add_arg('-x', '--lexicon', dest="lexfile", default=None, metavar='<file>',
            help='Output final lexicon with emission counts to given file')
    add_arg('-o', '--output', dest='outfile', default='-', metavar='<file>',
            help='Output file for test data results (for standard output, '
                 'use "-"; default "%(default)s").')
    add_arg('--save-analysis', dest='saveanalysisfile',
            default=None, metavar='<file>',
            help='Save analysis of the unannotated data. '
                 '(Deprecated: use of --save-tarball recommended)')
    add_arg('--save-annotations', dest='saveannotsfile',
            default=None, metavar='<file>',
            help='Save the annotated data set. '
                 '(Deprecated: use of --save-tarball recommended)')

    # for the ordering:
    add_arg = argument_groups.get('data format options')

    # Output post-processing
    add_arg = argument_groups.get('output post-processing options')
    add_arg('--remove-nonmorphemes', dest='rm_nonmorph', default=False,
            action='store_true',
            help='Use heuristic postprocessing to remove nonmorphemes '
                 'from output segmentations by joining or retagging morphs.')
    add_arg('--compound-splitter', dest='compound_split', default=False,
            action='store_true',
            help='Use FlatCat as a compound splitter. '
                 'Affixes will be joined with stems, '
                 'leaving only boundaries between compound parts')


def add_common_io_arguments(argument_groups):
    # Options for data formats
    add_arg = argument_groups.get('data format options')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')
    add_arg('--compound-separator', dest='cseparator', type=str, default='\s+',
            metavar='<regexp>',
            help='Compound separator regexp (default "%(default)s").')
    # FIXME: different defaults needed for diffent situations?
    add_arg('--construction-separator', dest='consseparator', type=str,
            default=' + ', metavar='<string>',
            help='Construction separator string (default "%(default)s").')
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
    add_arg('--output-construction-separator', dest='outputconseparator',
            type=str, default=' ', metavar='<str>',
            help='Construction separator for analysis in --output file '
                 '(default: "%(default)s")')
    add_arg('--output-category-separator', dest='outputtagseparator',
            type=str, default='/', metavar='<str>',
            help='Category tag separator for analysis in --output file '
                 '(default: "%(default)s")')
    add_arg('--output-categories', dest='test_output_tags', default=False,
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
    add_arg('--filter-max-len', dest='filter_len', type=int,
            default=3, metavar='<int>',
            help='Do not filter morphs longer than this limit when stemming. '
                 'Default: 3 (morphs of length 4 will be kept).')


def add_training_arguments(argument_groups):
    # Options for input data files
    add_arg = argument_groups.get('input data files')
    add_arg('--load-parameters', dest='loadparamsfile', default=None,
            metavar='<file>',
            help='Load hyperparameters from file. '
                 '(Deprecated: not needed with tarball (or pickle))')
    # Options for output data files
    add_arg = argument_groups.get('output data files')
    add_arg('--save-parameters', dest='saveparamsfile', default=None,
            metavar='<file>',
            help='Save hyperparameters to file. '
                 '(Deprecated: use of --save-tarball recommended)')
    # Options for training and segmentation
    add_arg = argument_groups.get('training and segmentation options')
    add_arg('-m', '--mode', dest='trainmode', default='batch',
            metavar='<mode>',
            choices=['none', 'batch', 'online', 'online+batch'],
            help='Training mode ("none", "batch", '
                 '"online", or "online+batch"; default "%(default)s")')
    add_arg('-p', '--perplexity-threshold', dest='ppl_threshold', type=float,
            default=None, metavar='<float>',
            help='Threshold value for sigmoid used to calculate '
                 'probabilities from left and right perplexities. '
                 '(default %(default)s).')
    add_arg('--prefix-perplexity-threshold', dest='pre_ppl_threshold',
            type=float, default=None, metavar='<float>',
            help='Separate perplexity threshold for prefixes. '
                 '(default is to use --perplexity-threshold for '
                 'both prefixes and suffixes).')
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
    add_arg('--ml-emissions-epoch', dest='ml_emissions_epoch',
            type=int, default=0, metavar='<int>',
            help='The number of epochs of resegmentation '
                 'using Maximum Likelihood estimation '
                 'for emission probabilities, '
                 'instead of using the morph property based probability. '
                 'These are performed after the normal training. '
                 '(default: do not switch over to ML estimation.')

    # Options for controlling training iteration sequence
    add_arg = argument_groups.get('training iteration sequence options')
    add_arg('--max-epochs', dest='max_epochs', type=int, default=4,
            metavar='<int>',
            help='The number of training epochs. (default %(default)s).')
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
                 'all epochs. (default %(default)s).')
    add_arg('--min-epoch-cost-gain', dest='min_epoch_cost_gain', type=float,
            default=None, metavar='<float>',
            help='Stop training if cost reduction between epochs '
                 'is below this limit * #boundaries. '
                 'In semi-supervised training the cost is not monotonous '
                 'between epochs, so this limit is meaningless. '
                 '(default %(default)s).')
    add_arg('--min-iteration-cost-gain', dest='min_iteration_cost_gain',
            type=float, default=0.0025, metavar='<float>',
            help='Stop training if cost reduction between iterations '
                 'is below this limit * #boundaries. '
                 '(default %(default)s).')
    add_arg('--min-difference-proportion', dest='min_diff_prop', type=float,
            default=0.005, metavar='<float>',
            help='Stop HMM initialization when the proportion of '
                 'words with changed category tags is below this limit. '
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
    add_arg('-A', '--annotations', dest='annofiles', default=[],
            action='append', metavar='<file>',
            help='Load annotated data for semi-supervised learning.')
    add_arg('-w', '--corpusweight', dest='corpusweight', type=float,
            default=None, metavar='<float>',
            help='Corpus weight parameter (default {}); '
                 'sets the initial value if --develset is used.'.format(
                    DEFAULT_CORPUSWEIGHT))
    add_arg('-W', '--annotationweight', dest='annotationweight',
            type=float, default=None, metavar='<float>',
            help='Corpus weight parameter for annotated data (if unset, the '
                 'weight is set to balance the number of tokens in annotated '
                 'and unannotated data sets).')


def add_weightlearning_arguments(argument_groups):
    # Options for automatically setting the weight parameters
    add_arg = argument_groups.get('weight learning options')
    add_arg('-D', '--develset', dest='develfile', default=None,
            metavar='<file>',
            help='Load annotated data for tuning the corpus weight parameter.')
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


def add_diagnostic_arguments(argument_groups):
    # Options for diagnostics
    add_arg = argument_groups.get('diagnostic options')
    add_arg('--statsfile', dest='stats_file', metavar='<file>',
            help='Collect iteration statistics and pickle them '
                 'into this file.')
    add_arg('--stats-annotations', dest='statsannotfile', default=None,
            metavar='<file>',
            help='Load annotated data for f-measure diagnostics. '
                 'Useful for analyzing convergence properties.')


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

    add_arg = argument_groups.get('other options')
    add_arg('-h', '--help', action='help',
            help='Show this help message and exit.')
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(numeric=True),
            help='Show version number and exit.')


def configure_logging(args):
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
    if (ch.level > logging.INFO or
            (hasattr(sys.stderr, 'isatty') and not sys.stderr.isatty())):
        utils.show_progress_bar = False

    if args.progress:
        utils.show_progress_bar = True
        ch.setLevel(min(ch.level, logging.INFO))


def flatcat_main(args):
    configure_logging(args)
    # Set frequency dampening function
    if args.dampening == 'none':
        dampfunc = None
    elif args.dampening == 'log':
        dampfunc = lambda x: int(round(math.log(x + 1, 2)))
    elif args.dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % args.dampening)

    # Arguments needing processing
    training_ops = args.training_operations.split(',')

    if (args.initfile is None):
        raise ArgumentException(
            'An initial Baseline or FlatCat model must be given.')

    init_is_pickle = any(args.initfile.endswith(ending)
                         for ending in BINARY_ENDINGS)
    init_is_tarball = any(args.initfile.endswith(ending)
                          for ending in TARBALL_ENDINGS)
    init_is_complete = (init_is_tarball or init_is_pickle)

    io = FlatcatIO(encoding=args.encoding,
                   construction_separator=args.consseparator,
                   compound_separator=args.cseparator,
                   analysis_separator=args.analysisseparator,
                   category_separator=args.catseparator)

    if ((not init_is_complete) and
            args.ppl_threshold is None and
            args.loadparamsfile is None):
        raise ArgumentException(
            'Perplexity threshold must be specified, '
            'either on command line or in hyper-parameter file. '
            'If you do not know what value to use, try something '
            'between 10 (small corpus / low morphological complexity) '
            'and 400 (large corpus, high complexity)')

    # Load exisiting model or create a new one
    must_train = False
    if init_is_pickle:
        _logger.info('Initializing from binary model...')
        model = io.read_binary_model_file(args.initfile)
    elif init_is_tarball:
        _logger.info('Initializing from tarball...')
        model = io.read_tarball_model_file(args.initfile)
    else:
        m_usage = categorizationscheme.MorphUsageProperties(
            ppl_threshold=args.ppl_threshold,
            ppl_slope=args.ppl_slope,
            length_threshold=args.length_threshold,
            length_slope=args.length_slope,
            type_perplexity=args.type_ppl,
            min_perplexity_length=args.min_ppl_length,
            pre_ppl_threshold=args.pre_ppl_threshold)
        if args.corpusweight is None:
            corpusweight = DEFAULT_CORPUSWEIGHT
        else:
            corpusweight = args.corpusweight
        model = flatcat.FlatcatModel(
            m_usage,
            forcesplit=args.forcesplit,
            nosplit=args.nosplit,
            corpusweight=corpusweight,
            use_skips=args.skips,
            ml_emissions_epoch=args.ml_emissions_epoch)
        _logger.info('Initializing from segmentation...')
        # Add the initial corpus data
        model.add_corpus_data(
            io.read_segmentation_file(args.initfile),
            count_modifier=dampfunc,
            freqthreshold=args.freqthreshold)
        model.training_operations = training_ops

    # Load the hyperparameters
    if not init_is_complete:
        if args.loadparamsfile is not None:
            _logger.info('Loading hyperparameters from {}'.format(
                args.loadparamsfile))
            model.set_params(
                io.read_parameter_file(args.loadparamsfile))

    # Add annotated data
    for f in args.annofiles:
        annotations = io.read_annotations_file(f)
        model.add_annotations(annotations,
                              args.annotationweight)

    # Override loaded values with values specified on the commandline
    if args.corpusweight is not None:
        model.set_corpus_coding_weight(args.corpusweight)
    if args.annotationweight is not None:
        model.set_annotation_coding_weight(args.annotationweight)
    if args.ppl_threshold is not None:
        model._morph_usage.set_params({
            'perplexity-threshold': args.ppl_threshold,
            'perplexity-slope': args.ppl_slope,
            'pre-perplexity-threshold': args.pre_ppl_threshold})

    # Initialize the model
    must_train = model.initialize_hmm(
        min_difference_proportion=args.min_diff_prop)

    # Extend the model with new unannotated data
    for f in args.extendfiles:
        model.add_corpus_data(io.read_segmentation_file(f),
                              count_modifier=dampfunc,
                              freqthreshold=args.freqthreshold)
        must_train = True

    # Set up statistics logging
    stats = None
    if args.stats_file is not None:
        stats = IterationStatistics()
        model.iteration_callbacks.append(stats.callback)
        stats.set_names(model, training_ops)

        if args.statsannotfile is not None:
            stats.set_gold_standard(
                io.read_annotations_file(args.statsannotfile))

    # Heuristic output postprocessing
    # nonmorpheme removal
    if args.rm_nonmorph:
        processor = categorizationscheme.HeuristicPostprocessor()
        if processor not in model.postprocessing:
            model.postprocessing.append(processor)
    # compound splitter
    if args.compound_split:
        processor = categorizationscheme.CompoundSegmentationPostprocessor()
        if processor not in model.postprocessing:
            model.postprocessing.append(processor)
    # FIXME: stemmer as postprocessor?

    # Perform weight learning using development annotations
#     if args.develfile is not None:
#         develannots = io.read_annotations_file(args.develfile)
#     else:
#         develannots = None
#
#     if develannots is not None:
#         weight_learning = flatcat.WeightLearning(
#             args.weightlearn_iters_first,
#             args.weightlearn_evals_first,
#             args.weightlearn_depth_first,
#             args.weightlearn_iters,
#             args.weightlearn_evals,
#             args.weightlearn_depth,
#             args.weightlearn_cuethresh,
#             develannots,
#             shared_model,
#             io,
#             args.checkpointfile,
#             heuristic)
#         for param in args.weightlearn_params.split(','):
#             if param == 'corpusweight':
#                 weight_learning.add_corpus_weight()
#             elif param == 'annotationweight':
#                 if shared_model.model._supervised:
#                     weight_learning.add_annotation_weight()
#         shared_model.model.generate_focus_samples(
#             args.weightlearn_sample_sets,
#             args.weightlearn_sample_size)
#
#         _logger.info('Performing initial weight learning')
#         must_train = weight_learning.optimize(first=True)
#
#         shared_model.model.training_focus = None
#     else:
#         weight_learning = None

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
        model.train_online(data, count_modifier=dampfunc,
                           epoch_interval=args.epochinterval,
                           max_epochs=(args.max_iterations * args.max_epochs))
    if args.trainmode in ('batch', 'online+batch'):
        ts = time.time()
        model.train_batch(
            min_iteration_cost_gain=args.min_iteration_cost_gain,
            min_epoch_cost_gain=args.min_epoch_cost_gain,
            max_epochs=args.max_epochs,
            max_iterations_first=args.max_iterations_first,
            max_iterations=args.max_iterations,
            max_resegment_iterations=args.max_resegment_iterations,
            max_shift_distance=args.max_shift_distance,
            min_shift_remainder=args.min_shift_remainder)
        _logger.info('Final cost: {}'.format(model.get_cost()))
        te = time.time()
        _logger.info('Training time: {:.3f}s'.format(te - ts))

    #
    # Save tarball
    if args.savetarballfile is not None:
        io.write_tarball_model_file(args.savetarballfile, model)

    # Old single-file saving formats (for hysterical raisins)
    # Save hyperparameters
    if args.saveparamsfile is not None:
        _logger.info("Saving hyperparameters...")
        io.write_parameter_file(args.saveparamsfile,
                                model.get_params())
        _logger.info("Done.")
    # Save analysis
    if args.saveanalysisfile is not None:
        _logger.info("Saving model as analysis...")
        io.write_segmentation_file(args.saveanalysisfile,
                                   model.segmentations)
        _logger.info("Done.")
    # Save annotations
    if model._supervised and args.saveannotsfile is not None:
        io.write_annotations_file(
            args.saveannotsfile,
            model.annotations,
            construction_sep=' ',
            output_tags=True)

    # Save lexicon
    if args.lexfile is not None:
        io.write_lexicon_file(args.lexfile, model.get_lexicon())

    # Save binary model
    if args.savepicklefile is not None:
        _logger.info("Saving binary model...")
        model.toggle_callbacks(None)
        io.write_binary_model_file(args.savepicklefile, model)
        _logger.info("Done.")

    # Segment test data
    if len(args.testfiles) > 0:
        if args.outfile == '-':
            utils.show_progress_bar = False
        _logger.info("Segmenting test data...")
        outformat = args.outputformat
        csep = args.outputconseparator
        tsep = args.outputtagseparator
        if not PY3:
            outformat = _str(outformat)
            csep = _str(csep)
            tsep = _str(tsep)
        outformat = outformat.replace(r"\n", "\n")
        outformat = outformat.replace(r"\t", "\t")
        keywords = [x[1] for x in string.Formatter().parse(outformat)]

        if len(args.filter_categories) > 0:
            filter_tags = [x.upper()
                           for x in args.filter_categories.split(',')]
        else:
            filter_tags = None

        def newline_func(item):
            (_, _, atoms) = item
            return len(atoms) == 0

        def segment_func(item):
            (count, compound, atoms) = item
            (constructions, logp) = model.viterbi_analyze(atoms)
            for processor in model.postprocessing:
                constructions = processor.apply_to(constructions, model)
            if 'clogprob' in keywords:
                clogp = model.forward_logprob(atoms)
            else:
                clogp = 0
            return (count, compound, [constructions], logp, clogp)

        io.write_formatted_file(
            args.outfile,
            outformat,
            io.read_corpus_files(args.testfiles),
            segment_func,
            newline_func=newline_func,
            output_newlines=args.outputnewlines,
            output_tags=args.test_output_tags,
            construction_sep=csep,
            category_sep=tsep,
            filter_tags=filter_tags,
            filter_len=args.filter_len)

        _logger.info("Done.")

    # Save statistics
    if args.stats_file is not None:
        io.write_binary_file(args.stats_file, stats)

    if args.savereduced is not None:
        reduced_model = reduced.FlatcatSegmenter(model)
        io.write_binary_file(args.savereduced, reduced_model)


def add_reformatting_arguments(argument_groups):
    # File format options
    add_arg = argument_groups.get('file format options')
    add_arg('-i', '--input-filetype', dest='infiletype',
            default='analysis', metavar='<format>',
            help='Format of input data. '
                 '("analysis", "annotations", "test") '
                 '(default: %(default)s)')
    add_arg('-o', '--output-filetype', dest='outfiletype',
            default='analysis', metavar='<format>',
            help='Format of output data. '
                 'Custom applies --output-format. '
                 '("analysis", "annotations", "test", "custom") '
                 '(default: %(default)s)')
    # FIXME: giving output-format should override default to custom,
    # or maybe die

    # Output post-processing
    add_arg = argument_groups.get('output post-processing options')
    add_arg('--map-categories', dest='map_categories', type=str,
            default=[], action='append', metavar='<from>,<to>',
            help='Map the <from> morph category to the <to> category. '
                 'Separate the categories with a single ",". '
                 'To perform multiple mappings simultaneously, '
                 'specify the option several times. '
                 '(default: do not map any categories.)')
    add_arg('--strip-categories', dest='strip_tags', default=False,
            action='store_true',
            help='If the input contains category tags, '
                 'omit them from the output. '
                 'Has no effect on untagged input. '
                 '(default: output category tags if they are known)')
    add_arg('--filter-junk', dest='filter_junk', default=False,
            action='store_true',
            help='Filter unwanted words from data. ')
    add_arg('--filter-max-chars', dest='filter_max_chars', type=int,
            default=60, metavar='<len>',
            help='Maximum number of characters in word')
    add_arg('--filter-max-morphs', dest='filter_max_morphs', type=int,
            default=15, metavar='<len>',
            help='Maximum number of morphs in initial segmentation')

    # Annotation processing
    add_arg = argument_groups.get('annotation processing options')
    add_arg('--first', dest='first_only', default=False,
            action='store_true',
            help='For each annotation in input, '
                 'only use the first alternative '
                 '(default: use all alternatives in sequence)')


def get_reformat_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='reformat.py',
        description="""
Morfessor {version} reformatting tool

{license}

Command-line arguments:
""" .format(version=get_version(), license=LICENSE),
        epilog="""
Usage examples:

  Convert CatMAP test output to FlatCat test output format:
    %(prog)s catmap_segmentation.final.gz analysis.txt \\
        --strip-categories

  Removal of short affixes from tagged segmentation:
    %(prog)s segmentation.tagged segmentation.txt \\
        --filter-categories PRE,SUF --strip-categories

  Convert the first alternative in the annotations to analysis format:
    %(prog)s annotations.txt analysis.txt -i annotations --first

  Convert analysis into annotation format:
    %(prog)s analysis.gz annotations.gz -o annotations

  Replace all affix tags with stem tags:
    %(prog)s analysis.gz modified.gz --map-categories PRE,STM \\
        --map-categories SUF,STM
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)
    groups = ArgumentGroups(parser)
    parser.add_argument('input', metavar='<infile>',
                        help='Input file to process.')
    parser.add_argument('output', metavar='<outfile>',
                        help='Output file for reformatted results.')
    add_common_io_arguments(groups)
    add_reformatting_arguments(groups)
    add_other_arguments(groups)
    return parser


IntermediaryFormat = collections.namedtuple('IntermediaryFormat',
    ['count', 'compound', 'alternatives'])

PseudoAnnotation = collections.namedtuple('PseudoAnnotation', ['alternatives'])


def reformat_main(args):
    configure_logging(args)

    inio = FlatcatIO(encoding=args.encoding,
                     construction_separator=args.consseparator,
                     compound_separator=args.cseparator,
                     analysis_separator=args.analysisseparator,
                     category_separator=args.catseparator,
                     strict=False)
    # encoding, compound separator and analysis separator not yet modifiable
    outio = FlatcatIO(encoding=args.encoding,
                      construction_separator=args.outputconseparator,
                      compound_separator=args.cseparator,
                      analysis_separator=args.analysisseparator,
                      category_separator=args.outputtagseparator,
                      strict=False)

    def read_analysis(file_name):
        for (count, analysis) in inio.read_segmentation_file(file_name):
            yield IntermediaryFormat(count,
                                     ''.join([x.morph for x in analysis]),
                                     [analysis])

    # This reader also works for test data
    def read_annotation(file_name):
        annotations = inio.read_annotations_file(file_name)
        for (compound, alternatives) in sorted(annotations.items()):
            if args.first_only:
                alternatives = alternatives[:1]
            yield IntermediaryFormat(1, compound, alternatives)

    def filter_junk(data, max_chars, max_morphs):
        for item in data:
            if len(item.alternatives) > 1:
                # Can't filter annotations
                yield item
            if len(item.compound) > max_chars:
                continue
            if len(item.alternatives[0]) > max_morphs:
                continue
            yield item

    def map_categories(data, from_cat, to_cat):
        for item in data:
            yield IntermediaryFormat(
                item.count,
                item.compound,
                [categorizationscheme.map_category(
                        analysis, from_cat, to_cat)
                 for analysis in item.alternatives])

    def custom_conversion(item):
        return (item.count, item.compound, item.alternatives, 0, 0)

    def separate_analyses(data):
        for item in data:
            for analysis in item.alternatives:
                yield IntermediaryFormat(
                    item.count, item.compound, [analysis])

    def write_analysis(file_name, data):
        if args.infiletype == 'annotations' and not args.first_only:
            data = separate_analyses(data)
        outio.write_segmentation_file(
            file_name,
            ((item.count, item.alternatives[0])
             for item in data),
            output_tags=(not args.strip_tags),
            comment_string=' reformatting tool')

    def write_annotation(file_name, data):
        data = {item.compound: PseudoAnnotation(item.alternatives)
                for item in data}
        outio.write_annotations_file(
            file_name, data,
            construction_sep=args.outputconseparator,
            output_tags=(not args.strip_tags))

    def write_test(file_name, data):
        if args.infiletype == 'annotations' and not args.first_only:
            data = separate_analyses(data)
        outio.write_formatted_file(
            file_name,
            '{compound}\t{analysis}\n',
            data,
            custom_conversion,
            output_tags=(not args.strip_tags),
            filter_tags=args.filter_categories,
            filter_len=args.filter_len)

    outformat = args.outputformat
    if not PY3:
        outformat = _str(outformat)
    outformat = outformat.replace(r"\n", "\n")
    outformat = outformat.replace(r"\t", "\t")

    def write_custom(file_name, data):
        outio.write_formatted_file(
            file_name,
            outformat,
            data,
            custom_conversion,
            output_tags=(not args.strip_tags),
            filter_tags=args.filter_categories,
            filter_len=args.filter_len)

    readers = {'analysis': read_analysis,
               'annotations': read_annotation,
               'test': read_analysis}
    writers = {'analysis': write_analysis,
               'annotations': write_annotation,
               'test': write_test,
               'custom': write_custom}

    if args.infiletype not in readers:
        raise ArgumentException(
            'Unknown input format "{}". Valid formats are {}.'.format(
            args.infiletype, sorted(readers.keys())))
    if args.outfiletype not in writers:
        raise ArgumentException(
            'Unknown output format "{}". Valid formats are {}.'.format(
            args.outfiletype, sorted(writers.keys())))

    data = readers[args.infiletype](args.input)
    data = _generator_progress(data)
    if args.filter_junk:
        data = filter_junk(data,
                           args.filter_max_chars,
                           args.filter_max_morphs)
    for mapping in args.map_categories:
        (from_cat, to_cat) = mapping.split(',')
        data = map_categories(data, from_cat, to_cat)
    writers[args.outfiletype](args.output, data)


# slightly modified copypasta from baseline
def get_evaluation_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="flatcat-evaluate",
        epilog="""Simple usage example:

  %(prog)s gold_standard model1.tar.gz model2.tar.gz
""",
        description="""
Morfessor {version} evaluation tool

{license}

Command-line arguments:
""" .format(version=get_version(), license=LICENSE),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    add_arg = parser.add_argument_group('evaluation options').add_argument
    add_arg('--num-samples', dest='numsamples', type=int, metavar='<int>',
            default=10, help='number of samples to take for testing')
    add_arg('--sample-size', dest='samplesize', type=int, metavar='<int>',
            default=1000, help='size of each testing samples')

    add_arg = parser.add_argument_group('formatting options').add_argument
    add_arg('--format-string', dest='formatstring', metavar='<format>',
            help='Python new style format string used to report evaluation '
                 'results. The following variables are a value and and action '
                 'separated with and underscore. E.g. fscore_avg for the '
                 'average f-score. The available values are "precision", '
                 '"recall", "fscore", "samplesize" and the available actions: '
                 '"avg", "max", "min", "values", "count". A last meta-data '
                 'variable (without action) is "name", the filename of the '
                 'model See also the format-template option for predefined '
                 'strings')
    add_arg('--format-template', dest='template', metavar='<template>',
            default='default',
            help='Uses a template string for the format-string options. '
                 'Available templates are: default, table and latex. '
                 'If format-string is defined this option is ignored')

    add_arg = parser.add_argument_group('file options').add_argument
    add_arg('--construction-separator', dest="consseparator", type=_str,
            default=' + ', metavar='<regexp>',
            help="construction separator for test segmentation files"
                 " (default '%(default)s')")
    add_arg('--compound-separator', dest='cseparator', type=str, default='\s+',
            metavar='<regexp>',
            help='Compound separator regexp (default "%(default)s").')
    add_arg('--analysis-separator', dest='analysisseparator', type=str,
            default=',', metavar='<regexp>',
            help='Separator for different analyses in an annotation file. Use '
                 'NONE for only allowing one analysis per line.')
    add_arg('--category-separator', dest='catseparator', type=str, default='/',
            metavar='<regexp>',
            help='separator for the category tag following a morph. '
                 '(default %(default)s).')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help="encoding of input and output files (if none is given, "
                 "both the local encoding and UTF-8 are tried)")

    add_arg = parser.add_argument_group('logging options').add_argument
    add_arg('-v', '--verbose', dest="verbose", type=int, default=1,
            metavar='<int>',
            help="verbose level; controls what is written to the standard "
                 "error stream or log file (default %(default)s)")
    add_arg('--logfile', dest='log_file', metavar='<file>',
            help="write log messages to file in addition to standard "
                 "error stream")

    # Output post-processing
    add_arg = parser.add_argument_group('output post-processing options').add_argument
    add_arg('--remove-nonmorphemes', dest='rm_nonmorph', default=False,
            action='store_true',
            help='Use heuristic postprocessing to remove nonmorphemes '
                 'from output segmentations by joining or retagging morphs.')
    add_arg('--compound-splitter', dest='compound_split', default=False,
            action='store_true',
            help='Use FlatCat as a compound splitter. '
                 'Affixes will be joined with stems, '
                 'leaving only boundaries between compound parts')

    add_arg = parser.add_argument_group('other options').add_argument
    add_arg('-h', '--help', action='help',
            help="show this help message and exit")
    add_arg('--version', action='version',
            version='%(prog)s ' + get_version(),
            help="show version number and exit")
    add_arg = parser.add_argument
    add_arg('goldstandard', metavar='<goldstandard>', nargs=1,
            help='gold standard file in standard annotation format')
    add_arg('models', metavar='<model>', nargs='+',
            help='model files to segment with (either tarball or binary. '
                 'Morfessor 1.0 style models NOT supported).')
    add_arg('-t', '--testsegmentation', dest='test_segmentations', default=[],
            action='append',
            help='Segmentation of the test set. Note that all words in the '
                 'gold-standard must be segmented')

    return parser


def main_evaluation(args):
    """ Separate main for running evaluation and statistical significance
    testing. Takes as argument the results of an get_evaluation_argparser()
    """
    io = FlatcatIO(encoding=args.encoding,
                     construction_separator=args.consseparator,
                     compound_separator=args.cseparator,
                     analysis_separator=args.analysisseparator,
                     category_separator=args.catseparator,
                   strict=False)
    blio = MorfessorIO(encoding=args.encoding)

    ev = bleval.MorfessorEvaluation(
        io.read_annotations_file(args.goldstandard[0]))

    results = []

    sample_size = args.samplesize
    num_samples = args.numsamples

    f_string = args.formatstring
    if f_string is None:
        f_string = bleval.FORMAT_STRINGS[args.template]

    for f in args.models:
        model = io.read_any_model(f)
        # Heuristic output postprocessing
        # nonmorpheme removal
        if args.rm_nonmorph:
            processor = categorizationscheme.HeuristicPostprocessor()
            if processor not in model.postprocessing:
                model.postprocessing.append(processor)
        # compound splitter
        if args.compound_split:
            processor = categorizationscheme.CompoundSegmentationPostprocessor()
            if processor not in model.postprocessing:
                model.postprocessing.append(processor)
        # FIXME: stemmer as postprocessor?
        result = ev.evaluate_model(model,
                                   configuration=bleval.EvaluationConfig(
                                        num_samples, sample_size),
                                   meta_data={'name': os.path.basename(f)})
        results.append(result)
        print(result.format(f_string))

    io.construction_separator = args.cseparator
    for f in args.test_segmentations:
        segmentation = blio.read_segmentation_file(f, False)
        result = ev.evaluate_segmentation(segmentation,
                                          configuration=
                                          bleval.EvaluationConfig(
                                                num_samples, sample_size),
                                          meta_data={'name':
                                                     os.path.basename(f)})
        results.append(result)
        print(result.format(f_string))

    if len(results) > 1 and num_samples > 1:
        wsr = bleval.WilcoxonSignedRank()
        r = wsr.significance_test(results)
        bleval.WilcoxonSignedRank.print_table(r)
