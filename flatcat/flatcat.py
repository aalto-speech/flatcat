#!/usr/bin/env python
"""
Morfessor 2.0 FlatCat variant.
"""
from __future__ import unicode_literals

__all__ = ['FlatcatModel', 'FlatcatLexiconEncoding', 'FlatcatEncoding',
           'FlatcatAnnotatedCorpusEncoding']

__author__ = 'Stig-Arne Gronroos'
__author_email__ = "morfessor@cis.hut.fi"

import collections
import logging
import math
import random
import re
import sys

from morfessor import baseline
from . import utils
from .categorizationscheme import MorphUsageProperties, WORD_BOUNDARY
from .categorizationscheme import ByCategory, get_categories, CategorizedMorph
from .categorizationscheme import DEFAULT_CATEGORY, HeuristicPostprocessor
from .categorizationscheme import MaximumLikelihoodMorphUsage
from .exception import InvalidOperationError
from .utils import LOGPROB_ZERO, zlog, _is_string

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)

# Grid node for viterbi algorithm
ViterbiNode = collections.namedtuple('ViterbiNode', ['cost', 'backpointer'])

WordAnalysis = collections.namedtuple('WordAnalysis', ['count', 'analysis'])

AnalysisAlternative = collections.namedtuple('AnalysisAlternative',
                                             ['analysis', 'penalty'])

SortedAnalysis = collections.namedtuple('SortedAnalysis',
                                        ['cost', 'analysis',
                                         'index', 'breakdown'])

Annotation = collections.namedtuple('Annotation',
                                    ['alternatives', 'current', 'i_unannot'])

CONS_SEP_WARNING = """
#################### WARNING ####################
The input does not seem to be segmented.
Are you using the correct construction separator?"""


class AbstractSegmenter(object):
    def __init__(self, corpus_coding, nosplit=None, postprocessing=None):
        self._initialized = False
        # None (= no corpus), "untagged", "partial", "full"
        self._corpus_tagging_level = None
        self._corpus_coding = corpus_coding
        # Do not allow splitting between a letter pair matching this regex
        if nosplit is None:
            self.nosplit_re = None
        elif _is_string(nosplit):
            self.nosplit_re = re.compile(nosplit, re.UNICODE)
        else:
            self.nosplit_re = nosplit
        self.annotations = None
        self._annotations_tagged = None
        self.postprocessing = postprocessing \
            if postprocessing is not None else []

    def initialize_hmm(self, min_difference_proportion=None):
        pass

    def viterbi_segment(self, segments, addcount=None, maxlen=None):
        """Compatibility with Morfessor Baseline.
        Postprocessing heuristics are applied to modify the segmentation.

        Note that the addcount and maxlen arguments are silently ignored.
        """
        analysis, logp = self.viterbi_analyze(segments)
        for processor in self.postprocessing:
            analysis = processor.apply_to(analysis, self)
        return (self.detag_word(analysis), logp)

    def viterbi_analyze(self, segments, strict_annot=True):
        """Simultaneously segment and tag a word using the learned model.
        Can be used to segment unseen words.

        Arguments:
            segments :  A word (or a list of morphs which will be
                        concatenated into a word) to resegment and tag.
            strict_annot :  If the word occurs in the annotated corpus,
                            only consider the segmentations in the annotation.
        Returns:
            best_analysis, :  The resegmented, retagged word
            best_cost      :  The cost of the returned solution
        """

        msg = 'Must initialize model and tag corpus before segmenting'
        assert (self._initialized and
            self._corpus_tagging_level == "full"), msg

        if _is_string(segments):
            word = segments
        else:
            # Throw away old category information, if any
            segments = self.detag_word(segments)
            # Merge potential segments
            word = ''.join(segments)

        # Return the best alternative from annotations if the word occurs there
        if word in self.annotations and strict_annot:
            annotation = self.annotations[word]
            alternatives = annotation.alternatives

            if not self._annotations_tagged:
                alternatives = tuple(self.viterbi_tag(alt, forbid_zzz=True)
                                     for alt in alternatives)

            sorted_alts = self.rank_analyses([AnalysisAlternative(alt, 0)
                                              for alt in alternatives])
            best = sorted_alts[0]
            return best.analysis, best.cost

        # To make sure that internally impossible states are penalized
        # even more than impossible states caused by zero parameters.
        extrazero = LOGPROB_ZERO ** 2

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = get_categories(wb=True)
        categories_nowb = [i for (i, c) in enumerate(categories)
                           if c != WORD_BOUNDARY]
        wb = categories.index(WORD_BOUNDARY)

        # Grid consisting of
        # the lowest accumulated cost ending in each possible state.
        # and back pointers that indicate the best path.
        # The grid is 3-dimensional:
        # grid [POSITION_IN_WORD]
        #      [MORPHLEN_OF_MORPH_ENDING_AT_POSITION - 1]
        #      [TAGINDEX_OF_MORPH_ENDING_AT_POSITION]
        # Initialized to pseudo-zero for all states
        zeros = [ViterbiNode(extrazero, None)] * len(categories)
        grid = [[zeros]]
        # Except probability one that first state is a word boundary
        grid[0][0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = None
        best = ViterbiNode(extrazero, None)

        for pos in range(1, len(word) + 1):
            grid.append([])
            for next_len in range(1, pos + 1):
                grid[pos].append(list(zeros))
                prev_pos = pos - next_len
                morph = self._interned_morph(word[prev_pos:pos])

                if (self.nosplit_re and
                        pos < len(word) and
                        self.nosplit_re.match(word[(pos - 1):(pos + 1)])):
                    # Splitting at this point is forbidden
                    grid[pos][next_len - 1] = zeros
                    continue
                if morph not in self:
                    # The morph corresponding to this substring has not
                    # been encountered: zero probability for this solution
                    grid[pos][next_len - 1] = zeros
                    continue

                for next_cat in categories_nowb:
                    best = ViterbiNode(extrazero, None)
                    if prev_pos == 0:
                        # First morph in word
                        cost = self._corpus_coding.transit_emit_cost(
                            WORD_BOUNDARY, categories[next_cat], morph)
                        if cost <= best.cost:
                            best = ViterbiNode(cost, ((0, wb),
                                CategorizedMorph(morph, categories[next_cat])))
                    # implicit else: for-loop will be empty if prev_pos == 0
                    for prev_cat in categories_nowb:
                        t_e_cost = self._corpus_coding.transit_emit_cost(
                                        categories[prev_cat],
                                        categories[next_cat],
                                        morph)
                        for prev_len in range(1, prev_pos + 1):
                            cost = (t_e_cost +
                                grid[prev_pos][prev_len - 1][prev_cat].cost)
                            if cost <= best.cost:
                                best = ViterbiNode(cost, ((prev_len, prev_cat),
                                    CategorizedMorph(morph,
                                                     categories[next_cat])))
                    grid[pos][next_len - 1][next_cat] = best

        # Last transition must be to word boundary
        best = ViterbiNode(extrazero, None)
        for prev_len in range(1, len(word) + 1):
            for prev_cat in categories_nowb:
                cost = (grid[-1][prev_len - 1][prev_cat].cost +
                        self._corpus_coding.log_transitionprob(
                            categories[prev_cat],
                            WORD_BOUNDARY))
                if cost <= best.cost:
                    best = ViterbiNode(cost, ((prev_len, prev_cat),
                        CategorizedMorph(WORD_BOUNDARY, WORD_BOUNDARY)))

        if best.cost >= LOGPROB_ZERO:
            #_logger.warning(
            #    'No possible segmentation for word {}'.format(word))
            return [CategorizedMorph(word, DEFAULT_CATEGORY)], LOGPROB_ZERO

        # Backtrace for the best morph-category sequence
        result = []
        backtrace = best
        pos = len(word)
        bt_len = backtrace.backpointer[0][0]
        bt_cat = backtrace.backpointer[0][1]
        while pos > 0:
            backtrace = grid[pos][bt_len - 1][bt_cat]
            bt_len = backtrace.backpointer[0][0]
            bt_cat = backtrace.backpointer[0][1]
            result.insert(0, backtrace.backpointer[1])
            pos -= len(backtrace.backpointer[1])
        return tuple(result), best.cost

    def viterbi_tag(self, segments, forbid_zzz=False):
        """Tag a pre-segmented word using the learned model.

        Arguments:
            segments :  A list of morphs to tag.
                        Raises KeyError if morph is not present in the
                        training data.
                        For segmenting and tagging new words,
                        use viterbi_analyze(word).
            forbid_zzz :  If True, no morph can be tagged as a
                          non-morpheme.
        """

        # Throw away old category information, if any
        segments = self.detag_word(segments)
        return self._viterbi_tag_helper(segments, forbid_zzz=forbid_zzz)

    def fast_tag_gaps(self, segments):
        """Tag the gaps in a pre-segmented word where most morphs are already
        tagged. Existing tags can not be changed.
        """
        def constraint(i, cat):
            if segments[i].category is None:
                return False
            if cat == segments[i].category:
                return False
            return True

        return self._viterbi_tag_helper(segments, constraint,
                                        AbstractSegmenter.detag_morph)

    def _viterbi_tag_helper(self, segments,
                            constraint=None, mapping=lambda x: x,
                            forbid_zzz=False):
        # To make sure that internally impossible states are penalized
        # even more than impossible states caused by zero parameters.
        extrazero = LOGPROB_ZERO * 100

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = get_categories(wb=True)
        wb = categories.index(WORD_BOUNDARY)
        forbidden = []
        for (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
            forbidden.append((categories.index(prev_cat),
                              categories.index(next_cat)))
        if forbid_zzz:
            for (prev_cat, next_cat) in MorphUsageProperties.forbid_zzz:
                forbidden.append((categories.index(prev_cat),
                                  categories.index(next_cat)))

        # Grid consisting of
        # the lowest accumulated cost ending in each possible state.
        # and back pointers that indicate the best path.
        # Initialized to pseudo-zero for all states
        grid = [[ViterbiNode(extrazero, None)] * len(categories)]
        # Except probability one that first state is a word boundary
        grid[0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = []
        best = []

        for (i, morph) in enumerate(segments):
            for (next_cat, nc_label) in enumerate(categories):
                if next_cat == wb:
                    # Impossible to visit boundary in the middle of the
                    # sequence
                    best.append(ViterbiNode(extrazero, None))
                    continue
                if constraint is not None and constraint(i, nc_label):
                    # lies outside the constrained path
                    best.append(ViterbiNode(extrazero, None))
                    continue
                morph = mapping(morph)
                for prev_cat in range(len(categories)):
                    if (prev_cat, next_cat) in forbidden:
                        cost.append(extrazero)
                        continue
                    # Cost of selecting prev_cat as previous state
                    # if now at next_cat
                    if grid[i][prev_cat].cost >= extrazero:
                        # This path is already failed
                        cost.append(extrazero)
                    else:
                        cost.append(grid[i][prev_cat].cost +
                                    self._corpus_coding.transit_emit_cost(
                                        categories[prev_cat],
                                        categories[next_cat], morph))
                best.append(ViterbiNode(*utils.minargmin(cost)))
                cost = []
            # Update grid to prepare for next iteration
            grid.append(best)
            best = []

        # Last transition must be to word boundary
        for prev_cat in range(len(categories)):
            pair = (categories[prev_cat], WORD_BOUNDARY)
            cost = (grid[-1][prev_cat].cost +
                    self._corpus_coding.log_transitionprob(*pair))
            best.append(cost)
        backtrace = ViterbiNode(*utils.minargmin(best))

        # Backtrace for the best category sequence
        result = [CategorizedMorph(
                    mapping(segments[-1]),
                    categories[backtrace.backpointer])]
        for i in range(len(segments) - 1, 0, -1):
            backtrace = grid[i + 1][backtrace.backpointer]
            morph = mapping(segments[i - 1])
            result.insert(0, CategorizedMorph(
                morph, categories[backtrace.backpointer]))
        return tuple(result)

    def forward_logprob(self, word):
        """Find log-probability of a word using the forward algorithm.

        Returns:
            cost      : (negative) log-probability of the word.
        """

        # To make sure that internally impossible states are penalized
        # even more than impossible states caused by zero parameters.
        extrazero = LOGPROB_ZERO ** 2

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = get_categories(wb=True)
        categories_nowb = [i for (i, c) in enumerate(categories)
                           if c != WORD_BOUNDARY]
        wb = categories.index(WORD_BOUNDARY)

        # Grid consisting of
        # the accumulated cost ending in each possible state.
        # The grid is 3-dimensional:
        # grid [POSITION_IN_WORD]
        #      [MORPHLEN_OF_MORPH_ENDING_AT_POSITION - 1]
        #      [TAGINDEX_OF_MORPH_ENDING_AT_POSITION]
        # Initialized to pseudo-zero for all states
        zeros = [extrazero] * len(categories)
        grid = [[zeros]]
        # Except probability one that first state is a word boundary
        grid[0][0][wb] = 0

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = None
        psum = 0.0

        for pos in range(1, len(word) + 1):
            grid.append([])
            for next_len in range(1, pos + 1):
                grid[pos].append(list(zeros))
                prev_pos = pos - next_len
                morph = self._interned_morph(word[prev_pos:pos])

                if (self.nosplit_re and
                        pos < len(word) and
                        self.nosplit_re.match(word[(pos - 1):(pos + 1)])):
                    # Splitting at this point is forbidden
                    grid[pos][next_len - 1] = zeros
                    continue
                if morph not in self:
                    # The morph corresponding to this substring has not
                    # been encountered: zero probability for this solution
                    grid[pos][next_len - 1] = zeros
                    continue

                for next_cat in categories_nowb:
                    psum = 0.0
                    if prev_pos == 0:
                        # First morph in word
                        cost = self._corpus_coding.transit_emit_cost(
                            WORD_BOUNDARY, categories[next_cat], morph)
                        psum += math.exp(-cost)
                    # implicit else: for-loop will be empty if prev_pos == 0
                    for prev_cat in categories_nowb:
                        t_e_cost = self._corpus_coding.transit_emit_cost(
                                        categories[prev_cat],
                                        categories[next_cat],
                                        morph)
                        for prev_len in range(1, prev_pos + 1):
                            cost = (t_e_cost +
                                grid[prev_pos][prev_len - 1][prev_cat])
                            psum += math.exp(-cost)
                    if psum > 0:
                        cost = -math.log(psum)
                    else:
                        cost = LOGPROB_ZERO
                    grid[pos][next_len - 1][next_cat] = cost

        # Last transition must be to word boundary
        psum = 0.0
        for prev_len in range(1, len(word) + 1):
            for prev_cat in categories_nowb:
                cost = (grid[-1][prev_len - 1][prev_cat] +
                        self._corpus_coding.log_transitionprob(
                            categories[prev_cat],
                            WORD_BOUNDARY))
                psum += math.exp(-cost)
        if psum > 0:
            cost = -math.log(psum)
        else:
            cost = LOGPROB_ZERO

        return cost

    def rank_analyses(self, choices):
        """Choose the best analysis of a set of choices.

        Observe that the call and return signatures are different
        from baseline: this method is more versatile.

        Arguments:
            choices :  a sequence of AnalysisAlternative(analysis, penalty)
                       namedtuples.
                       The analysis must be a sequence of CategorizedMorphs,
                       (segmented and tagged).
                       The penalty is a float that is added to the cost
                       for this choice. Use 0 to disable.
        Returns:
            A sorted (by cost, ascending) list of
            SortedAnalysis(cost, analysis, index, breakdown) namedtuples. ::
                cost :  the contribution of this analysis to the corpus cost.
                analysis :  as in input.
                breakdown :  A CostBreakdown object, for diagnostics
        """
        out = []
        for (i, choice) in enumerate(choices):
            out.append(self.cost_breakdown(choice.analysis, choice.penalty, i))
        return sorted(out)

    def cost_breakdown(self, segmentation, penalty=0.0, index=0):
        """Returns breakdown of costs for the given tagged segmentation."""
        wrapped = _wb_wrap(segmentation)
        breakdown = CostBreakdown()
        for (prefix, suffix) in utils.ngrams(wrapped, n=2):
            cost = self._corpus_coding.log_transitionprob(prefix.category,
                                                          suffix.category)
            breakdown.transition(cost, prefix.category, suffix.category)
            if suffix.morph != WORD_BOUNDARY:
                cost = self._corpus_coding.log_emissionprob(
                        suffix.category, suffix.morph)
                breakdown.emission(cost, suffix.category, suffix.morph)
        if penalty != 0:
            breakdown.penalty(penalty)
        return SortedAnalysis(breakdown.cost, segmentation, index, breakdown)

    def _interned_morph(self, morph, store=False):
        """Override in subclass"""
        return morph

    def __contains__(self, morph):
        raise AttributeError('Must override __contains__')

    @staticmethod
    def get_categories(wb=False):
        """The category tags supported by this model.

        Arguments:
            wb :  If True, the word boundary will be included. Default: False.
        """
        return get_categories(wb)

    @staticmethod
    def detag_morph(morph):
        if isinstance(morph, CategorizedMorph):
            return morph.morph
        return morph

    @staticmethod
    def detag_word(segments):
        return tuple(AbstractSegmenter.detag_morph(x) for x in segments)

    @staticmethod
    def detag_list(segmentations):
        """Removes category tags from a segmented corpus."""
        for rcount, segments in segmentations:
            yield ((rcount, tuple(AbstractSegmenter.detag_morph(x)
                                  for x in segments)))

    @staticmethod
    def filter_untagged(segmentations):
        """Yields only the part of the corpus that is tagged."""
        for rcount, segments in segmentations:
            is_tagged = all(cmorph.category is not None
                            for cmorph in segments)
            if is_tagged:
                yield (rcount, segments)

    @property
    def word_tokens(self):
        return self._corpus_coding.boundaries


class FlatcatModel(AbstractSegmenter):
    """Morfessor FlatCat model class.

    Arguments:
        morph_usage :  A MorphUsageProperties object describing how
                        the usage of a morph affects the category.
        forcesplit :  Force segmentations around the characters
                        in the given list. The same value should be
                        used in Morfessor Baseline or other initialization,
                        to guarantee results.
        nosplit :  Prevent splitting between character pairs matching
                    this regular expression.  The same value should be
                    used in Morfessor Baseline or other initialization,
                    to guarantee results.
        corpusweight :  Multiplicative weight for the
                        (unannotated) corpus cost.
        use_skips :  Randomly skip frequently occurring constructions
                        to speed up online training. Has no effect on
                        batch training.
        ml_emissions_epoch :  The number of epochs of resegmentation
                                using Maximum Likelihood estimation
                                for emission probabilities,
                                instead of using the morph property
                                based probability. These are performed
                                after the normal training.
                                Default -1 means do not switch over
                                to ML estimation.
        """

    word_boundary = WORD_BOUNDARY

    # 'shift' is no longer included as 3rd op by default
    DEFAULT_TRAIN_OPS = ['split', 'join', 'resegment']

    def __init__(self, morph_usage=None, forcesplit=None, nosplit=None,
                 corpusweight=1.0, use_skips=False, ml_emissions_epoch=-1):
        # Morph usage properties
        if morph_usage is None:
            morph_usage = MorphUsageProperties()
        self._morph_usage = morph_usage

        # Cost variables
        self._lexicon_coding = FlatcatLexiconEncoding(morph_usage)
        # Flatcat encoding also stores the HMM parameters
        self._corpus_coding = FlatcatEncoding(morph_usage,
                                              self._lexicon_coding,
                                              weight=corpusweight)

        super(FlatcatModel, self).__init__(self._corpus_coding,
                                           nosplit=nosplit)
        self._initialized = False
        # None (= no corpus), "untagged", "partial", "full"
        self._corpus_tagging_level = None
        self._segment_only = False

        # The analyzed (segmented and tagged) corpus
        self.segmentations = []

        # Morph occurence backlinks
        # A dict of sets. Keys are morphs, set contents are indices to
        # self.segmentations for words in which the morph occurs
        self.morph_backlinks = collections.defaultdict(set)

        # Cache for custom interning system
        self._interned_morphs = {}

        # Counters for the current epoch and operation within
        # that epoch. These describe the stage of training
        # to allow resuming training of a pickled model.
        # The exact point in training is described by 3 numbers:
        #   - the iteration number (each iteration is one pass over the data
        #     while performing one type of operation).
        #     The iteration number is not restored when loading.
        #   - the operation number (iterations performing the same operation
        #     are repeated until convergence, before moving to
        #     the next operation)
        #   - the epoch number (an epoch consists of the sequence
        #     of all training operations)
        self._epoch_number = 0
        self._operation_number = 0

        # The sequence of training operations.
        # Valid training operations are strings for which FlatcatModel
        # has a function named _op_X_generator, where X is the string
        # which returns a transform generator suitable for
        # passing to _operation_loop.
        # This is done using strings indstead of bound methods,
        # to enable pickling of the model object.
        self.training_operations = self.DEFAULT_TRAIN_OPS

        # Training sequence parameters.
        self._max_epochs = 5
        self._min_epoch_cost_gain = 0.0
        self._min_iteration_cost_gain = 0.0
        self._max_iterations_first = 1
        self._max_iterations = 1
        self._max_resegment_iterations = 1
        self._min_shift_remainder = 2
        self._max_shift = 2
        self._ml_emissions_epoch = ml_emissions_epoch

        # Callbacks for cleanup/bookkeeping after each operation.
        # Should take exactly one argument: the model.
        self.operation_callbacks = []
        self.iteration_callbacks = []
        self._changed_segmentations = None
        self._changed_segmentations_op = None

        # Force these atoms to be kept as separate morphs.
        # Calling morfessor baseline with the same forcesplit value ensures
        # that they are initially separate.
        if forcesplit is None:
            self.forcesplit = []
        else:
            self.forcesplit = tuple(forcesplit)

        # Variables for semi-supervised training
        self._supervised = False
        self._annot_coding = None
        self.annotations = {}   # word -> Annotation
        self._annotations_tagged = None

        # Variables for online learning
        self._online = False
        self.training_focus = None
        self.training_focus_sets = None
        self._use_skips = use_skips  # Random skips for frequent constructions
        self._skipcounter = collections.Counter()

        # Variables for weight learning
        self._weight_learning = None

        # Logging variables
        self._cost_field_width = 9
        self._cost_field_precision = 4

    ### Primary public methods
    #
    def add_corpus_data(self, segmentations, freqthreshold=1,
                        count_modifier=None):
        """Adds the given segmentations (with counts) to the corpus data.
        The new data can be either untagged or tagged.

        If the added data is untagged, you must call viterbi_tag_corpus
        to tag the new data.

        You should also call reestimate_probabilities and consider
        calling initialize_hmm.

        Arguments:
            segmentations :  Segmentations of format:
                             (count, (morph1, morph2, ...))
                             where the morphs can be either strings
                             or CategorizedMorphs.
            freqthreshold :  discard words that occur less than
                             given times in the corpus (default 1).
            count_modifier :  function for adjusting the counts of each
                              word.
        """
        assert isinstance(freqthreshold, (int, float))
        i = len(self.segmentations)
        consecutive_unsegmented = 0
        for row in segmentations:
            count, analysis = row
            if count < freqthreshold:
                continue
            if count_modifier is not None:
                count = count_modifier(count)
            if len(analysis) == 0:
                continue
            if consecutive_unsegmented is not None:
                if len(analysis) == 1:
                    consecutive_unsegmented += 1
                    if consecutive_unsegmented == 100:
                        _logger.warning(CONS_SEP_WARNING)
                else:
                    consecutive_unsegmented = None
            if isinstance(analysis[0], CategorizedMorph):
                is_tagged = all(cmorph.category is not None
                    for cmorph in analysis)
                self._intern_word(analysis)
                if self._corpus_tagging_level is None:
                    if is_tagged:
                        self._corpus_tagging_level = "full"
                    else:
                        self._corpus_tagging_level = "untagged"

                if not is_tagged and self._corpus_tagging_level == "full":
                    self._corpus_tagging_level = "partial"
                if is_tagged and self._corpus_tagging_level == "untagged":
                    self._corpus_tagging_level = "partial"
            else:
                analysis = tuple(CategorizedMorph(
                                    self._interned_morph(morph, store=True),
                                    None)
                                 for morph in analysis)
                if self._corpus_tagging_level is None:
                    self._corpus_tagging_level = "untagged"
                if self._corpus_tagging_level == "full":
                    self._corpus_tagging_level = "partial"
            segmentation = WordAnalysis(count, analysis)
            self.segmentations.append(segmentation)
            for morph in self.detag_word(segmentation.analysis):
                self.morph_backlinks[morph].add(i)
            i += 1
            self._corpus_coding.boundaries += count

    def add_annotations(self, annotations, annotatedcorpusweight=None):
        """Adds data to the annotated corpus."""
        self._supervised = True
        if self._annotations_tagged is None:
            self._annotations_tagged = True
        word_backlinks = {
            ''.join(self.detag_word(seg.analysis)): i
            for (i, seg) in enumerate(self.segmentations)}
        for (word, alternatives) in annotations.items():
            if alternatives[0][0].category is None:
                self._annotations_tagged = False
            if word in word_backlinks:
                i_unannot = word_backlinks[word]
            else:
                # The word is also added to the unannotated corpus,
                # to ensure that the needed morphs are available
                i_unannot = len(self.segmentations)
                self.segmentations.append(
                    WordAnalysis(1, alternatives[0]))
            self.annotations[word] = Annotation(alternatives, None, i_unannot)
        del word_backlinks
        self._calculate_morph_backlinks()
        self._annot_coding = FlatcatAnnotatedCorpusEncoding(
                                self._corpus_coding,
                                weight=annotatedcorpusweight)
        self._annot_coding.boundaries = len(self.annotations)
        if (not self._annotations_tagged and
                self._corpus_tagging_level == "full"):
            self._corpus_tagging_level = "partial"

    def initialize_baseline(self, min_difference_proportion=0.005):
        """Initialize emission and transition probabilities without
        changing the segmentation, using Viterbi EM, from a previously
        added (see add_corpus_data) segmentation produced by a
        morfessor baseline model.
        """

        self._calculate_usage_features()
        self._unigram_transition_probs()
        self.viterbi_tag_corpus()
        self._calculate_transition_counts()
        self._calculate_emission_counts()

        def reestimate_with_unchanged_segmentation():
            self._calculate_transition_counts()
            self._calculate_emission_counts()

        self._convergence_of_analysis(
            reestimate_with_unchanged_segmentation,
            self.viterbi_tag_corpus,
            min_difference_proportion=min_difference_proportion,
            min_cost_gain=-10.0)     # Cost gain will be ~zero.

    def initialize_hmm(self, min_difference_proportion=0.005):
        """Initialize emission and transition probabilities without
        changing the segmentation.
        """

        must_train = False

        fs = ForceSplitter(self.forcesplit, self.nosplit_re)
        (self.segmentations, must_reestimate) = fs.enforce(self.segmentations)
        if must_reestimate:
            self.reestimate_probabilities()
            self._calculate_morph_backlinks()
        self.reestimate_probabilities()

        if self._corpus_tagging_level == "untagged":
            must_train = True
            self.initialize_baseline(min_difference_proportion)

        if self._corpus_tagging_level == "partial":
            self.viterbi_tag_corpus()
            self.reestimate_probabilities()

        if self._supervised:
            self._update_annotation_choices()

        for callback in self.iteration_callbacks:
            callback(self)

        self._epoch_number = 1
        return must_train

    def train_batch(self,
                    min_iteration_cost_gain=0.0025,
                    min_epoch_cost_gain=0.005,
                    max_epochs=5,
                    max_iterations_first=1,
                    max_iterations=1,
                    max_resegment_iterations=1,
                    max_shift_distance=2,
                    min_shift_remainder=2):
        """Perform batch training.

        Arguments:
            min_iteration_cost_gain :  Do not repeat iteration if the gain
                                       in cost was less than this proportion.
                                       No effect if max_iterations is 1.
                                       Set to None to disable.
            min_epoch_cost_gain :  Stop before max_epochs, if the gain
                                   in cost of the previous epoch was less
                                   than this proportion.
                                   Set to None to disable.
            max_epochs :  Maximum number of training epochs.
            max_iterations_first :  Number of iterations of each operator,
                                    in the first epoch.
            max_iterations :  Number of iterations of each operator, in
                              later epochs.
            max_resegment_iterations :  Number of resegment iterations
                                        in any epoch.
            max_shift_distance :  Limit on the distance (in characters)
                                  that the shift operation can move a boundary.
            min_shift_remainder :  Limit on the shortest morph allowed to be
                                   produced by the shift operation.
        """
        self._min_iteration_cost_gain = min_iteration_cost_gain
        self._min_epoch_cost_gain = min_epoch_cost_gain
        self._max_epochs = max_epochs
        self._max_iterations_first = max_iterations_first
        self._max_iterations = max_iterations
        self._max_resegment_iterations = max_resegment_iterations
        self._max_shift = max_shift_distance
        self._min_shift_remainder = min_shift_remainder
        self._online = False

        msg = 'Must initialize model and tag corpus before training'
        assert self._corpus_tagging_level == "full", msg
        self._epoch_update(no_increment=True)
        previous_cost = self.get_cost()
        wl_force_another = False
        u_force_another = False
        if self._ml_emissions_epoch > 0:
            ml_epochs = self._ml_emissions_epoch
        else:
            ml_epochs = 0
        for epoch in range(self._max_epochs + ml_epochs):
            self._train_epoch()

            cost = self.get_cost()
            cost_diff = cost - previous_cost
            limit = self._cost_convergence_limit(
                self._min_epoch_cost_gain)

            if (self._weight_learning is not None and
                    epoch < self._max_epochs - 1):
                wl_force_another = self._weight_learning.optimize()
            u_force_another = self._epoch_update()
            post_update_cost = self.get_cost()
            if post_update_cost != cost:
                _logger.info('Cost from {} to {} in epoch update'.format(
                    cost, post_update_cost))
                cost = post_update_cost

            conv_str = ''
            if limit is None:
                converged = False
                conv_str = 'fixed number of epochs'
            else:
                converged = -cost_diff <= limit
                if converged:
                    conv_str = 'converged'
            if wl_force_another or u_force_another:
                converged = False
                conv_str = 'additional epoch forced'

            self._display_cost(cost_diff, limit, 'epoch',
                            epoch, self._max_epochs, conv_str)
            if converged:
                _logger.info('{:24s} Cost: {}'.format(
                    'final epoch.', cost))
                break
            previous_cost = cost
        self.reestimate_probabilities()

    def train_online(self, data, count_modifier=None, epoch_interval=10000,
                     max_epochs=None, result_callback=None):
        """Adapt the model in online fashion."""

        self._online = True
        self._skipcounter = collections.Counter()
        if count_modifier is not None:
            counts = {}
        word_backlinks = self._online_backlinks()

        _logger.info("Starting online training")

        epochs = 0
        token_num = 0
        more_tokens = True
        self.reestimate_probabilities()
        while more_tokens:
            newcost = self.get_cost()
            _logger.info(
                "Tokens processed: %s\tCost: %s" % (token_num, newcost))

            for _ in utils._progress(range(epoch_interval)):
                try:
                    is_anno, _, w, atoms = next(data)
                except StopIteration:
                    more_tokens = False
                    break

                if count_modifier is not None:
                    if not w in counts:
                        c = 0
                        counts[w] = 1
                        add_count = 1
                    else:
                        c = counts[w]
                        counts[w] = c + 1
                        add_count = count_modifier(c + 1) - count_modifier(c)
                else:
                    add_count = 1

                i_word = word_backlinks.get(w, None)
                if add_count > 0:
                    if is_anno:
                        (i_word, recalculate) = self._online_labeled_token(
                                w, atoms, i_word)
                        if recalculate:
                            word_backlinks = self._online_backlinks()
                    else:
                        i_word = self._online_unlabeled_token(w, add_count,
                                                              i_word)
                    assert i_word is not None
                    word_backlinks[w] = i_word
                (segments, _) = self.viterbi_analyze(w)

                _logger.debug("#%s: %s -> %s" %
                              (token_num, w, segments))
                if result_callback is not None:
                    result_callback(token_num,
                                    w,
                                    segments,
                                    self.detag_word(segments))
                token_num += 1

            # also reestimates the probabilities
            _logger.info("Epoch reached, resegmenting corpus")
            self._viterbi_analyze_corpus()
            if self._supervised:
                self._update_annotation_choices()

            self._skipcounter = collections.Counter()
            epochs += 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break

        self.reestimate_probabilities()
        newcost = self.get_cost()
        _logger.info("Tokens processed: %s\tCost: %s" % (token_num, newcost))
        return epochs, newcost

    def _online_backlinks(self):
        self.morph_backlinks.clear()
        word_backlinks = {}
        for (i, seg) in enumerate(self.segmentations):
            for morph in self.detag_word(seg.analysis):
                self.morph_backlinks[morph].add(i)
            joined = ''.join(self.detag_word(seg.analysis))
            word_backlinks[joined] = i
        return word_backlinks

    def _online_labeled_token(self, word, segments, i_word=None):
        if not self._supervised:
            self._annot_coding = FlatcatAnnotatedCorpusEncoding(
                                    self._corpus_coding,
                                    weight=None)
            self._supervised = True

        if segments[0].category is None:
            self._annotations_tagged = False

        add_annotation_entry = False
        changes_annot = ChangeCounts()

        if word in self.annotations:
            annotation = self.annotations[word]
            if i_word is None:
                i_word = annotation.i_unannot
            else:
                assert i_word == annotation.i_unannot

            # Correcting an earlier annotation
            changes_annot.update(annotation.current, -1)
        else:
            add_annotation_entry = True

        if segments[0].category is None:
            # FIXME: problem if new morphs are introduced?
            new_analysis = self.viterbi_tag(segments, forbid_zzz=True)
        else:
            new_analysis = segments

        if add_annotation_entry:
            if i_word is None:
                i_word = len(self.segmentations)
                self.segmentations.append(
                    WordAnalysis(1, tuple(new_analysis)))
                for morph in self.detag_word(segments):
                    self._modify_morph_count(morph, 1)
                self._calculate_morph_backlinks()
            self.annotations[word] = Annotation((segments,),
                                                tuple(new_analysis),
                                                i_word)
            self._annot_coding.boundaries += 1
            self._annot_coding.update_weight()
        else:
            self.annotations[word] = Annotation((segments,),
                                                tuple(new_analysis),
                                                i_word)

        assert i_word is not None
        changes_annot.update(new_analysis, 1)
        self._annot_coding.update_counts(changes_annot)
        self._annot_coding.reset_contributions()

        self.training_focus = set()
        self.training_focus.add(i_word)
        self._single_iteration_epoch()

        for _ in range(3):
            if (self.detag_word(self.segmentations[i_word].analysis)
                    == self.detag_word(segments)):
                break
            self._single_iteration_epoch()

        return (i_word, add_annotation_entry)

    def _online_unlabeled_token(self, word, add_count, i_word=None):
        skip_this = False
        if (self._use_skips and
                i_word is not None and
                self._test_skip(word)):
            # Only increase the word count, don't analyze
            skip_this = True

        if skip_this:
            segments = self.segmentations[i_word].analysis
        else:
            segments, _ = self.viterbi_analyze(word)

        change_counts = ChangeCounts()
        if i_word is not None:
            # The word is already in the corpus
            old_seg = self.segmentations[i_word]
            change_counts.update(old_seg.analysis,
                                 -old_seg.count,
                                 corpus_index=i_word)
            for morph in self.detag_word(old_seg.analysis):
                self._modify_morph_count(morph, -old_seg.count)
            self.segmentations[i_word] = WordAnalysis(
                old_seg.count + add_count,
                segments)
            self._corpus_coding.boundaries += add_count
        else:
            self.add_corpus_data([WordAnalysis(add_count, segments)])
            i_word = len(self.segmentations) - 1
        new_count = self.segmentations[i_word].count
        change_counts.update(self.segmentations[i_word].analysis,
                                new_count, corpus_index=i_word)
        for morph in self.detag_word(segments):
            self._modify_morph_count(morph, new_count)

        self._update_counts(change_counts, 1)

        if not skip_this:
            self.training_focus = set((i_word,))
            self._single_iteration_epoch()
        assert i_word is not None
        return i_word

    def viterbi_tag_corpus(self):
        """(Re)tags the corpus segmentations using viterbi_tag"""
        num_changed_words = 0
        for (i, word) in enumerate(self.segmentations):
            self.segmentations[i] = WordAnalysis(word.count,
                self.viterbi_tag(word.analysis))
            if word != self.segmentations[i]:
                num_changed_words += 1
        self._corpus_tagging_level = "full"
        return num_changed_words

    ### Secondary public methods
    #
    def get_cost(self):
        """Return current model encoding cost."""
        cost = self._corpus_coding.get_cost() + self._lexicon_coding.get_cost()
        assert cost >= 0
        if self._supervised:
            cost += self._annot_coding.get_cost()
        assert cost >= 0
        return cost

    def reestimate_probabilities(self):
        """Re-estimates model parameters from a segmented, tagged corpus.

        theta(t) = arg min { L( theta, Y(t), D ) }
        """
        self._intern_corpus()
        self._calculate_usage_features()
        self._calculate_transition_counts()
        self._calculate_emission_counts()
        if self._supervised:
            self._annot_coding.reset_contributions()
        self._initialized = True

    def get_params(self):
        """Returns a dict of hyperparameters."""
        params = {'corpusweight': self.get_corpus_coding_weight()}
        params.update(self._morph_usage.get_params())
        if self._supervised:
            params['annotationweight'] = self._annot_coding.weight
        params['forcesplit'] = ''.join(self.forcesplit)
        if self.nosplit_re:
            params['nosplit'] = self.nosplit_re.pattern
        return params

    def set_params(self, params):
        """Sets hyperparameters to loaded values."""
        if 'corpusweight' in params:
            self.set_corpus_coding_weight(float(params['corpusweight']))
        if self._supervised and 'annotationweight' in params:
            self.set_annotation_coding_weight(
                float(params['annotationweight']))
        if 'forcesplit' in params:
            self.forcesplit = [x for x in params['forcesplit']]
        if 'nosplit' in params:
            self.nosplit_re = re.compile(params['nosplit'], re.UNICODE)
        self._morph_usage.set_params(params)

    def get_corpus_coding_weight(self):
        return self._corpus_coding.weight

    def set_corpus_coding_weight(self, weight):
        self._corpus_coding.weight = weight
        _logger.info('Setting corpus coding weight to {}'.format(weight))

    def set_annotation_coding_weight(self, weight):
        self._annot_coding.weight = weight
        self._annot_coding.do_update_weight = False
        _logger.info('Setting annotation weight to {}'.format(weight))

    def get_lexicon(self):
        """Returns morphs in lexicon, with emission counts"""
        assert self._initialized
        for morph in sorted(self._morph_usage.seen_morphs()):
            yield (morph, self._corpus_coding.get_emission_counts(morph))

    def _training_focus_filter(self):
        """Yields segmentations.
        If no training filter is selected, the whole corpus is generated.
        Otherwise only segmentations in the focus sample are generated.
        """
        if self.training_focus is None:
            for seg in self.segmentations:
                yield seg
        else:
            ordered = sorted(self.training_focus)
            for i in ordered:
                yield self.segmentations[i]

    def generate_focus_samples(self, num_sets, num_samples):
        """Generates subsets of the corpus by weighted sampling."""
        if num_samples == 0 or num_samples > len(self.segmentations):
            # Setting num_samples to zero means full set is preferred
            # Also: no point sampling a larger set than the corpus
            self.training_focus = None
            self.training_focus_sets = None
        else:
            self.training_focus_sets = []
            for _ in range(num_sets):
                self.training_focus_sets.append(
                    set(utils.weighted_sample(self.segmentations,
                                              num_samples)))

    def set_focus_sample(self, set_index):
        """Select one pregenerated focus sample set as active."""
        if self.training_focus_sets is None:
            self.training_focus = None
            return
        else:
            self.training_focus = self.training_focus_sets[set_index]

    def toggle_callbacks(self, callbacks=None):
        """Callbacks are not saved in the pickled model, because pickle is
        unable to restore instance methods. If you need callbacks in a loaded
        model, you have to readd them after loading.
        """
        out = (self.operation_callbacks, self.iteration_callbacks)
        if callbacks is None:
            self.operation_callbacks = []
            self.iteration_callbacks = []
        else:
            (self.operation_callbacks, self.iteration_callbacks) = callbacks
        return out

    def __getstate__(self):
        # clear caches of owned objects
        self._corpus_coding.clear_transition_cache()
        self._corpus_coding.clear_emission_cache()
        self._morph_usage.clear()   # this needs to be restored

        # These will be restored
        out = self.__dict__.copy()
        del out['morph_backlinks']
        del out['_interned_morphs']
        del out['_skipcounter']

        # restores cleared _morph_usage
        self.reestimate_probabilities()

        return out

    def __setstate__(self, d):
        self.__dict__ = d
        # recreate deleted fields
        self.morph_backlinks = collections.defaultdict(set)
        self._interned_morphs = {}
        self._skipcounter = collections.Counter()

        # restore cleared caches
        self._calculate_morph_backlinks()
        self.reestimate_probabilities()

    ### Public diagnostic methods
    #
    def rank_analyses(self, choices):
        """Choose the best analysis of a set of choices.

        Observe that the call and return signatures are different
        from baseline: this method is more versatile.

        Arguments:
            choices :  a sequence of AnalysisAlternative(analysis, penalty)
                       namedtuples.
                       The analysis must be a sequence of CategorizedMorphs,
                       (segmented and tagged).
                       The penalty is a float that is added to the cost
                       for this choice. Use 0 to disable.
        Returns:
            A sorted (by cost, ascending) list of
            SortedAnalysis(cost, analysis, index, breakdown) namedtuples. ::
                cost :  the contribution of this analysis to the corpus cost.
                analysis :  as in input.
                breakdown :  A CostBreakdown object, for diagnostics
        """
        out = []
        for (i, choice) in enumerate(choices):
            out.append(self.cost_breakdown(choice.analysis, choice.penalty, i))
        return sorted(out)

    def cost_breakdown(self, segmentation, penalty=0.0, index=0):
        """Returns breakdown of costs for the given tagged segmentation."""
        wrapped = _wb_wrap(segmentation)
        breakdown = CostBreakdown()
        for (prefix, suffix) in utils.ngrams(wrapped, n=2):
            cost = self._corpus_coding.log_transitionprob(prefix.category,
                                                          suffix.category)
            breakdown.transition(cost, prefix.category, suffix.category)
            if suffix.morph != WORD_BOUNDARY:
                cost = self._corpus_coding.log_emissionprob(
                        suffix.category, suffix.morph)
                breakdown.emission(cost, suffix.category, suffix.morph)
        if penalty != 0:
            breakdown.penalty(penalty)
        return SortedAnalysis(breakdown.cost, segmentation, index, breakdown)

    def cost_comparison(self, segmentations, retag=True):
        """Diagnostic function.
        (Re)tag the given segmentations, calculate their cost
        and return the sorted breakdowns of the costs.
        Can be used to analyse reasons for a segmentation choice.
        """

        if len(segmentations) == 0:
            return
        if all(_is_string(s) for s in segmentations):
            segmentations = [segmentations]
        if retag:
            assert isinstance(retag, bool)
            tagged = []
            for seg in segmentations:
                tagged.append(AnalysisAlternative(self.viterbi_tag(seg), 0))
        else:
            tagged = [AnalysisAlternative(x, 0) for x in segmentations]
        return self.rank_analyses(tagged)

    def words_with_morph(self, morph):
        """Diagnostic function.
        Returns all segmentations using the given morph.
        Format: (index_to_segmentations, count, analysis)
        """
        out = []
        for i in self.morph_backlinks[morph]:
            seg = self.segmentations[i]
            out.append((i, seg.count, seg.analysis))
        return sorted(out)

    def morph_count(self, morph):
        return self._morph_usage.count(morph)

    def violated_annotations(self):
        """Yields all segmentations which have an associated annotation,
        but woud currently not be naturally segmented in a way that is included
        in the annotation alternatives,"""
        for (word, anno) in self.annotations.items():
            alts_de = [self.detag_word(alt) for alt in anno.alternatives]
            seg_de = self.detag_word(
                self.viterbi_analyze(
                    word,
                    strict_annot=False)[0])

            if seg_de not in alts_de:
                yield (seg_de, alts_de)

    def viterbi_analyze_list(self, corpus):
        """Convenience wrapper around viterbi_analyze for a
        list of word strings or segmentations with attached counts.
        Segmented input can be with or without tags.
        This function can be used to analyze previously unseen data.
        """
        for line in corpus:
            if (isinstance(line, (WordAnalysis, tuple)) and
                    len(line) == 2 and
                    isinstance(line[0], int)):
                count, word = line
            else:
                word = line
                count = 1
            if _is_string(word):
                word = (word,)
            yield WordAnalysis(count, self.viterbi_analyze(word)[0])

    ### Training operations
    #
    def _generic_bimorph_generator(self, result_func):
        """The common parts of operation generators that operate on
        context-sensitive bimorphs. Don't call this directly.

        Arguments:
            result_func :  A function that takes the prefix an suffix
                           as arguments, and returns all the proposed results
                           as tuples of CategorizedMorphs.
        """

        bigram_freqs = collections.Counter()
        for (count, segments) in self._training_focus_filter():
            segments = _wb_wrap(segments)
            for quad in utils.ngrams(segments, n=4):
                prev_morph, prefix, suffix, next_morph = quad
                if (prefix.morph in self.forcesplit or
                    suffix.morph in self.forcesplit):
                    # don't propose to join morphs on forcesplit list
                    continue
                context_type = MorphUsageProperties.context_type(
                    prev_morph.morph, next_morph.morph,
                    prev_morph.category, next_morph.category)
                bigram_freqs[(prefix, suffix, context_type)] += count

        for (bigram, count) in bigram_freqs.most_common():
            prefix, suffix, context_type = bigram
            # Require both morphs, tags and context to match
            rule = TransformationRule((prefix, suffix),
                                      context_type=context_type)
            temporaries = set()
            transforms = []
            changed_morphs = set((prefix.morph, suffix.morph))
            results = result_func(prefix, suffix)
            for result in results:
                detagged = self.detag_word(result)
                changed_morphs.update(detagged)
                temporaries.update(self._morph_usage.estimate_contexts(
                    (prefix.morph, suffix.morph), detagged))
                transforms.append(Transformation(rule, result))
            # targets will be a subset of the intersection of the
            # occurences of both submorphs
            targets = set(self.morph_backlinks[prefix.morph])
            targets.intersection_update(self.morph_backlinks[suffix.morph])
            if len(targets) > 0:
                yield(transforms, targets, changed_morphs, temporaries)

    def _op_split_generator(self):
        """Generates splits of seen morphs into two submorphs.
        Use with _operation_loop
        """
        if self.training_focus is None:
            unsorted = self._morph_usage.seen_morphs()
        else:
            unsorted = set()
            for (_, segmentation) in self._training_focus_filter():
                for morph in self.detag_word(segmentation):
                    unsorted.add(morph)
        iteration_morphs = sorted(unsorted, key=len)
        for morph in iteration_morphs:
            if len(morph) == 1:
                continue
            if self._morph_usage.count(morph) == 0:
                continue

            # Match the parent morph with any category
            rule = TransformationRule((CategorizedMorph(morph, None),))
            transforms = []
            changed_morphs = set((morph,))
            # Apply to all words in which the morph occurs
            targets = self.morph_backlinks[morph]
            # Temporary estimated contexts
            temporaries = set()
            for splitloc in range(1, len(morph)):
                if (self.nosplit_re and
                        self.nosplit_re.match(
                            morph[(splitloc - 1):(splitloc + 1)])):
                    continue
                prefix = self._interned_morph(morph[:splitloc])
                suffix = self._interned_morph(morph[splitloc:])
                changed_morphs.update((prefix, suffix))
                # Make sure that there are context features available
                # (real or estimated) for the submorphs
                tmp = (self._morph_usage.estimate_contexts(morph,
                                                           (prefix, suffix)))
                temporaries.update(tmp)
                transforms.append(
                    Transformation(rule,
                                   (CategorizedMorph(prefix, None),
                                    CategorizedMorph(suffix, None))))
            yield (transforms, targets, changed_morphs, temporaries)

    def _op_join_generator(self):
        """Generates joins of consecutive morphs into a supermorph.
        Can make different join decisions in different contexts.
        Use with _operation_loop
        """

        def join_helper(prefix, suffix):
            joined = prefix.morph + suffix.morph
            joined = self._interned_morph(joined)
            return ((CategorizedMorph(joined, None),),)

        return self._generic_bimorph_generator(join_helper)

    def _op_shift_generator(self):
        """Generates operations that shift the split point in a bigram.
        Use with _operation_loop
        """

        def shift_helper(prefix, suffix):
            results = []
            for i in range(1, self._max_shift + 1):
                # Move backward
                if len(prefix) - i >= self._min_shift_remainder:
                    new_pre = prefix.morph[:-i]
                    shifted = prefix.morph[-i:]
                    new_suf = shifted + suffix.morph
                    if (not self.nosplit_re or
                            not self.nosplit_re.match(
                                new_pre[-1] + new_suf[0])):
                        new_pre = self._interned_morph(new_pre)
                        new_suf = self._interned_morph(new_suf)
                        results.append((CategorizedMorph(new_pre, None),
                                        CategorizedMorph(new_suf, None)))
                # Move forward
                if len(suffix) - i >= self._min_shift_remainder:
                    new_suf = suffix.morph[i:]
                    shifted = suffix.morph[:i]
                    new_pre = prefix.morph + shifted
                    if (not self.nosplit_re or
                            not self.nosplit_re.match(
                                new_pre[-1] + new_suf[0])):
                        new_pre = self._interned_morph(new_pre)
                        new_suf = self._interned_morph(new_suf)
                        results.append((CategorizedMorph(new_pre, None),
                                        CategorizedMorph(new_suf, None)))
            return results

        return self._generic_bimorph_generator(shift_helper)

    def _op_resegment_generator(self):
        """Generates special transformations that resegment and tag
        all words in the corpus using viterbi_analyze.
        Use with _operation_loop
        """
        if self.training_focus is None:
            source = range(len(self.segmentations))
        else:
            source = self.training_focus
        # Sort by count, ascending
        source = sorted([(self.segmentations[i].count, i) for i in source])
        for (_, i) in source:
            word = self.segmentations[i]
            changed_morphs = set(self.detag_word(word.analysis))
            vrt = ViterbiResegmentTransformation(word, self)
            changed_morphs.update(self.detag_word(vrt.result))

            if word.analysis != vrt.result:
                yield ([vrt], set([i]), changed_morphs, set())

    ### Private: reestimation
    #
    def _calculate_usage_features(self):
        """Recalculates the morph usage features (perplexities).
        """

        self._lexicon_coding.clear()

        # FIXME: unnecessary to restrict to tagged?
        if self._corpus_tagging_level == "untagged":
            segs = self.segmentations
        else:
            # Must make it a list, _morph_usage
            # reads it several times expecting unchanged contents
            segs = list(self.filter_untagged(self.segmentations))
        self._morph_usage.calculate_usage_features(
            lambda: self.detag_list(segs))

        for morph in self._morph_usage.seen_morphs():
            self._lexicon_coding.add(morph)

    def _unigram_transition_probs(self):
        """Initial transition probabilities based on unigram distribution.

        Each tag is presumed to be succeeded by the expectation over all data
        of the number of prefixes, suffixes, stems, non-morphemes and word
        boundaries.
        """

        category_totals = self._morph_usage.category_token_count

        transitions = collections.Counter()
        nclass = {WORD_BOUNDARY: self.word_tokens}
        for (i, category) in enumerate(get_categories()):
            nclass[category] = float(category_totals[i])

        num_tokens_tagged = 0.0
        valid_transitions = MorphUsageProperties.valid_transitions()

        for (cat1, cat2) in valid_transitions:
            # count all possible valid transitions
            num_tokens_tagged += nclass[cat2]
            transitions[(cat1, cat2)] = nclass[cat2]

        if num_tokens_tagged == 0:
            _logger.warning('Tried to train without data')
            return

        for pair in MorphUsageProperties.zero_transitions:
            transitions[pair] = 0.0

        normalization = (sum(nclass.values()) / num_tokens_tagged)
        for (prev_cat, next_cat) in transitions:
            self._corpus_coding.update_transition_count(
                prev_cat, next_cat,
                transitions[(prev_cat, next_cat)] * normalization)
        self._corpus_coding.clear_transition_cache()

    def _calculate_transition_counts(self):
        """Count the number of transitions of each type.
        Can be used to estimate transition probabilities from
        a category-tagged segmented corpus.
        """

        self._corpus_coding.clear_transition_counts()
        for rcount, segments in self.filter_untagged(self.segmentations):
            # Only the categories matter, not the morphs themselves
            categories = [x.category for x in segments]
            # Include word boundaries
            categories.insert(0, WORD_BOUNDARY)
            categories.append(WORD_BOUNDARY)
            for (prev_cat, next_cat) in utils.ngrams(categories, 2):
                pair = (prev_cat, next_cat)
                if pair in MorphUsageProperties.zero_transitions:
                    _logger.warning('Impossible transition ' +
                                    '{!r} -> {!r}'.format(*pair))
                self._corpus_coding.update_transition_count(prev_cat,
                                                            next_cat,
                                                            rcount)
        self._corpus_coding.clear_transition_cache()

    def _calculate_emission_counts(self):
        """Recalculates the emission counts from a retagged segmentation."""
        self._corpus_coding.clear_emission_counts()
        for (count, analysis) in self.filter_untagged(self.segmentations):
            for morph in analysis:
                self._corpus_coding.update_emission_count(morph.category,
                                                          morph.morph,
                                                          count)

    def _calculate_morph_backlinks(self):
        """Recalculates the mapping from morphs to the indices of corpus
        words in which the morphs occur."""
        self.morph_backlinks.clear()
        for (i, segmentation) in enumerate(self.segmentations):
            for morph in self.detag_word(segmentation.analysis):
                self.morph_backlinks[morph].add(i)

    def _epoch_update(self, no_increment=False):
        """Updates performed between training epochs.
        Set the no_increment flag to suppress incrementing
        the epoch number, which is needed when the update is
        performed several times during one epoch e.g. in weight learning.
        """

        force_another = False

        if self._supervised:
            old_cost = self.get_cost()
            _logger.info('Updating annotation choices...')
            self._update_annotation_choices()
            new_cost = self.get_cost()
            if old_cost != new_cost:
                _logger.info('Updated annotation choices, changing cost from '
                            '{} to {}'.format(old_cost, new_cost))
                force_another = True
            self._annot_coding.update_weight()
            _logger.info('Cost balance (W Corp / W Anno): {}'.format(
                self._corpus_coding.get_cost() /
                self._annot_coding.get_cost()))

        self._operation_number = 0
        if not no_increment:
            self._epoch_number += 1

        if (self._ml_emissions_epoch > 0 and
                self._epoch_number == self._max_epochs + 1):
            _logger.info('Switching over to ML-estimation (resegment only)')
            self._morph_usage = MaximumLikelihoodMorphUsage(
                self._corpus_coding, self._morph_usage.get_params())
            self._calculate_usage_features()
            self.training_operations = ['resegment']
            return True
        return force_another

    def _update_annotation_choices(self):
        """Update the selection of alternative analyses in annotations."""
        if not self._supervised:
            return
        # Will need to check for proper expansion when introducing
        # hierarchical morphs

        count_diff = collections.Counter()
        # changes to unannotated corpus counts
        changes_unannot = ChangeCounts()
        # changes to annotated corpus counts
        changes_annot = ChangeCounts()
        overwrite = {}
        for (word, annotation) in self.annotations.items():
            alternatives = annotation.alternatives
            current_unannot = self.segmentations[annotation.i_unannot].analysis

            if not self._annotations_tagged:
                alternatives = tuple(self.viterbi_tag(alt, forbid_zzz=True)
                                     for alt in alternatives)

            sorted_alts = self.rank_analyses([AnalysisAlternative(alt, 0)
                                              for alt in alternatives])
            new_active = sorted_alts[0].analysis

            changes_unannot.update(current_unannot, -1,
                                    corpus_index=annotation.i_unannot)
            changes_unannot.update(new_active, 1,
                                   corpus_index=annotation.i_unannot)
            changes_annot.update(new_active, 1)

            # Active segmentation is changed before removal/adding of morphs
            self.segmentations[annotation.i_unannot] = WordAnalysis(
                1, new_active)
            overwrite[word] = Annotation(
                alternatives,
                tuple(new_active),
                annotation.i_unannot)
            for morph in self.detag_word(current_unannot):
                count_diff[morph] -= 1
            new_detagged = self.detag_word(new_active)
            for morph in new_detagged:
                count_diff[morph] += 1
        for word in overwrite:
            self.annotations[word] = overwrite[word]

        for (morph, count) in count_diff.items():
            if count == 0:
                continue
            self._modify_morph_count(morph, count)
        self._update_counts(changes_unannot, 1)
        # which one depends on if we count changes or start from zero
        self._annot_coding.set_counts(changes_annot)
        #self._annot_coding.update_counts(changes_annot)
        self.reestimate_probabilities()
        # done already by reestimate
        #self._annot_coding.reset_contributions()

    ### Private: model state updaters
    #
    def _modify_morph_count(self, morph, diff_count):
        """Modifies the count of a morph in the lexicon.
        Does not affect transitions or emissions."""
        old_count = self._morph_usage.count(morph)
        new_count = old_count + diff_count
        self._morph_usage.set_count(morph, new_count)
        self._corpus_coding.clear_emission_cache()
        if old_count == 0 and new_count > 0:
            self._lexicon_coding.add(morph)
        elif old_count > 0 and new_count == 0:
            self._lexicon_coding.remove(morph)

    def _update_counts(self, change_counts, multiplier):
        """Updates the model counts according to the pre-calculated
        ChangeCounts object (e.g. calculated in Transformation).

        Arguments:
            change_counts :  A ChangeCounts object
            multiplier :  +1 to apply the change, -1 to revert it.
        """
        for cmorph in change_counts.emissions:
            self._corpus_coding.update_emission_count(
                cmorph.category,
                cmorph.morph,
                change_counts.emissions[cmorph] * multiplier)

        for (prev_cat, next_cat) in change_counts.transitions:
            self._corpus_coding.update_transition_count(
                prev_cat, next_cat,
                change_counts.transitions[(prev_cat, next_cat)] * multiplier)
        self._corpus_coding.clear_transition_cache()

        if multiplier > 0:
            bl_rm = change_counts.backlinks_remove
            bl_add = change_counts.backlinks_add
        else:
            bl_rm = change_counts.backlinks_add
            bl_add = change_counts.backlinks_remove

        for morph in bl_rm:
            self.morph_backlinks[morph].difference_update(
                change_counts.backlinks_remove[morph])
        for morph in bl_add:
            self.morph_backlinks[morph].update(
                change_counts.backlinks_add[morph])

    ### Private: iteration structure
    #
    def _convergence_of_cost(self, train_func, update_func,
                            min_cost_gain=0.005,
                            max_iterations=5):
        """Iterates the specified training function until the model cost
        no longer improves enough or until maximum number of iterations
        is reached.

        On each iteration train_func is called without arguments.
        The data used to train the model must therefore be completely
        contained in the model itself. This can e.g. mean iterating over
        the morphs already stored in the lexicon.

        Arguments:
            train_func :  A method of FlatcatModel which causes some part of
                          the model to be trained.
            update_func :  Updates to the model between iterations,
                           that should not be considered in the convergence
                           analysis. However, if the return value is
                           True, at least one more iteration is forced unless
                           the maximum limit has been reached.
            min_cost_gain :  Stop iterating if cost reduction between
                             iterations is below this limit * #boundaries.
                             Default 0.005.
            max_iterations :  Maximum number of iterations. Default 5.
        """

        previous_cost = self.get_cost()
        for iteration in range(max_iterations):
            cost = self.get_cost()
            msg = ('{:9s} {:2d}/{:<2d}          Cost: {:' +
                   self._cost_field_fmt(cost) + 'f}.')
            _logger.info(msg.format('iteration',
                                    iteration + 1, max_iterations,
                                    cost))

            # perform the optimization
            train_func()

            # perform update between optimization iterations
            if update_func is not None:
                _logger.info('{:24s} Cost: {}'.format(
                    'Before iteration update.', cost))

            # perform update between optimization iterations
            if update_func is not None:
                force_another = update_func()
            else:
                force_another = False

            for callback in self.iteration_callbacks:
                callback(self, iteration)

            cost = self.get_cost()
            cost_diff = cost - previous_cost
            limit = self._cost_convergence_limit(min_cost_gain)
            conv_str = ''
            if limit is None:
                converged = False
                conv_str = 'fixed number of iterations'
            else:
                converged = -cost_diff <= limit
                if converged:
                    conv_str = 'converged'
            if force_another:
                converged = False
                conv_str = 'additional iteration forced'

            self._display_cost(cost_diff, limit, 'iteration',
                           iteration, max_iterations, conv_str)
            if converged:
                _logger.info('{:24s} Cost: {}'.format(
                    'final iteration ({}).'.format(conv_str),
                    cost))
                return
            previous_cost = cost
        if not converged:
            _logger.info('{:24s} Cost: {}'.format(
                'final iteration (max iterations reached).', cost))

    def _convergence_of_analysis(self, train_func, resegment_func,
                                min_difference_proportion=0,
                                min_cost_gain=0, max_iterations=15):
        """Iterates the specified training function until the segmentations
        produced by the model no longer changes more than
        the specified threshold, until the model cost no longer improves
        enough or until maximum number of iterations is reached.

        On each iteration the current optimal analysis for the corpus is
        produced by calling resegment_func.
        This corresponds to:

        Y(t) = arg min { L( theta(t-1), Y, D ) }

        Then the train_func function is called, which corresponds to:

        theta(t) = arg min { L( theta, Y(t), D ) }

        Neither train_func nor resegment_func may require any arguments.

        Arguments:
            train_func :  A method of FlatcatModel which causes some aspect
                          of the model to be trained.
            resegment_func :  A method of FlatcatModel that resegments or
                              retags the segmentations, to produce the
                              results to compare. Should return the number
                              of changed words.
            min_difference_proportion :  Maximum proportion of words with
                                         changed segmentation or category
                                         tags in the final iteration.
                                         Default 0.
            min_cost_gain :  Stop iterating if cost reduction between
                                   iterations is below this limit.
            max_iterations :  Maximum number of iterations. Default 15.
        """

        previous_cost = self.get_cost()
        for iteration in range(max_iterations):
            _logger.info(
                'Iteration {:2d} ({}). {:2d}/{:<2d}'.format(
                    self._epoch_number, train_func.__name__,
                    iteration + 1, max_iterations))

            # perform the optimization
            train_func()

            cost = self.get_cost()
            cost_diff = cost - previous_cost
            cost_limit = min_cost_gain * self._corpus_coding.boundaries
            if -cost_diff <= cost_limit:
                msg = ('Cost difference {:' +
                            self._cost_field_fmt(cost_diff) + 'f} ' +
                       '(limit {}) ' +
                       'in iteration {:2d}    (Converged).')
                _logger.info(msg.format(cost_diff, cost_limit, iteration + 1))
                break

            # perform the reanalysis
            differences = resegment_func()

            limit = min_difference_proportion * len(self.segmentations)
            # the length of the slot needed to display
            # the number of segmentations
            field_width = str(len(str(len(self.segmentations))))
            if differences <= limit:
                msg = ('Segmentation differences: {:' + field_width + 'd} ' +
                       '(limit {:' + field_width + 'd}). ' +
                       'in iteration {:2d}    (Converged).')
                _logger.info(msg.format(differences, int(math.floor(limit)),
                                        iteration + 1))
                break
            msg = ('Segmentation differences: {:' + field_width + 'd} ' +
                   '(limit {:' + field_width + 'd}). Cost difference: {}')
            _logger.info(msg.format(differences, int(math.floor(limit)),
                                    cost_diff))
            previous_cost = cost

    def _train_epoch(self):
        """One epoch of training, which may contain several iterations
        of each operation in sequence.

        The model must have been initialized, either by loading a baseline
        segmentation or a pretrained flatcat model from pickle or tagged
        segmentation file, and calling initialize_baseline and/or
        initialize_hmm
        """
        cost = self.get_cost()
        msg = ('{:9s} {:2d}/{:<2d}          Cost: {:' +
                self._cost_field_fmt(cost) + 'f}.')
        _logger.info(msg.format('epoch',
                                self._epoch_number,
                                self._max_epochs,
                                cost))
        if self._changed_segmentations is not None:
            # FIXME: use for also for convergence, not just stats?
            self._changed_segmentations.clear()
        while self._operation_number < len(self.training_operations):
            operation = self._resolve_operation(self._operation_number)
            min_iteration_cost_gain = self._training_params(
                'min_iteration_cost_gain')
            max_iterations = self._training_params('max_iterations')
            if self._training_params('must_reestimate'):
                update_func = self.reestimate_probabilities
            else:
                update_func = None

            msg = 'Epoch {:2d}, operation {:2d} ({}), max {:2d} iteration(s).'
            _logger.info(msg.format(
                    self._epoch_number, self._operation_number,
                    self.training_operations[self._operation_number],
                    max_iterations))
            self._convergence_of_cost(
                lambda: self._operation_loop(operation()),
                update_func=update_func,
                min_cost_gain=min_iteration_cost_gain,
                max_iterations=max_iterations)
            self.reestimate_probabilities()
            self._operation_number += 1
            for callback in self.operation_callbacks:
                callback(self)

    def _single_iteration_epoch(self):
        """One epoch of training, with exactly one iteration of each
        operation and no convergence checks or update passes."""
        for i in range(len(self.training_operations)):
            operation = self._resolve_operation(i)
            self._operation_loop(operation())

    def _operation_loop(self, transformation_generator):
        """Performs each experiment yielded by the transform generator,
        in sequence, always choosing the alternative that minimizes the
        model cost, or making no change if all choices would increase the
        cost.

        An experiment is a set of transformations sharing a common matching
        rule. The individual transformations in the set are considered to
        be mutually exclusive, only one of them can be chosen.

        Can be used to split, join or otherwise re-segment.
        Can even be abused to alter the corpus: it is up to the caller to
        ensure that the rules and results detokenize to the same string.

        Arguments:
            transformation_generator :  a generator yielding
                (transform_group, targets, changed_morphs, temporaries)
                tuples, where
                    transform_group :  a list of Transform objects.
                    targets :  a set with an initial guess of indices of
                               matching words in the corpus. Can contain false
                               positives, but should not omit positive
                               indices (they will be missed).
                    changed_morphs :  the morphs participating
                                      in the operation.
                    temporaries :  a set of new morphs with estimated
                                   contexts.
        """

        TransformationNode = collections.namedtuple('TransformationNode',
                                                    ['cost',
                                                     'transform',
                                                     'targets'])
        if self._changed_segmentations_op is not None:
            self._changed_segmentations_op.clear()
        if not self._online:
            transformation_generator = utils._generator_progress(
                transformation_generator)
        for experiment in transformation_generator:
            (transform_group, targets,
             changed_morphs, temporaries) = experiment
            if len(transform_group) == 0:
                continue
            # Cost of doing nothing
            old_cost = self.get_cost()
            best = TransformationNode(old_cost, None, set())

            # All transforms in group must match the same words,
            # we can use just the first transform
            matched_targets, num_matches = self._find_in_corpus(
                transform_group[0].rule, targets)
            if num_matches == 0:
                continue

            detagged = self.detag_word(transform_group[0].rule)
            if self._supervised:
                logemissionsum_initial = self._annot_coding.logemissionsum
                # Old contribution to annotation cost needs to be
                # removed before the probability changes
                # (when using ML-estimate, this needs to be done for
                # corpus cost also)
                for morph in changed_morphs:
                    self._annot_coding.modify_contribution(morph, -1)
            for morph in detagged:
                # Remove the old representation, but only from
                # morph counts (emissions and transitions updated later)
                self._modify_morph_count(morph, -num_matches)

            for transform in transform_group:
                #redundant_targets, redundant_num = self._find_in_corpus(transform.rule, targets)
                #assert redundant_targets == matched_targets, (redundant_targets, matched_targets)
                #assert redundant_num == num_matches, (redundant_num, num_matches)
                detagged = self.detag_word(transform.result)
                for morph in detagged:
                    # Add the new representation to morph counts
                    self._modify_morph_count(morph, num_matches)
                for target in matched_targets:
                    old_analysis = self.segmentations[target]
                    transform.apply(old_analysis, self)

                # Apply change to encoding
                self._update_counts(transform.change_counts, 1)
                # Observe that annotation counts are not updated,
                # even if the transform targets an annotation,
                # because that would defeat the purpose of annotations
                if self._supervised:
                    # contribution to annotation cost needs to be readded
                    # after the emission probability has been updated
                    # (ordering with _update_counts relevant for ML-estimate)
                    logemissionsum_tmp = self._annot_coding.logemissionsum
                    for morph in changed_morphs:
                        self._annot_coding.modify_contribution(morph, 1)
                cost = self.get_cost()
                if cost < best.cost:
                    best = TransformationNode(cost, transform, matched_targets)
                # Revert change to encoding
                if self._supervised:
                    # Numerically more stable than adding with reverse sign
                    self._annot_coding.logemissionsum = logemissionsum_tmp
                    #for morph in changed_morphs:
                    #    self._annot_coding.modify_contribution(morph, -1)
                self._update_counts(transform.change_counts, -1)
                for morph in self.detag_word(transform.result):
                    self._modify_morph_count(morph, -num_matches)

            if best.transform is None:
                # Best option was to do nothing. Revert morph count.
                for morph in self.detag_word(transform_group[0].rule):
                    self._modify_morph_count(morph, num_matches)
                if self._supervised:
                    self._annot_coding.logemissionsum = logemissionsum_initial
            else:
                # A real change was the best option
                best.transform.reset_counts()
                for morph in self.detag_word(best.transform.result):
                    # Add the new representation to morph counts
                    self._modify_morph_count(morph, num_matches)
                for target in best.targets:
                    new_analysis = best.transform.apply(
                        self.segmentations[target],
                        self, corpus_index=target)
                    self._intern_word(new_analysis.analysis)
                    self.segmentations[target] = new_analysis
                    # any morph used in the best segmentation
                    # is no longer temporary
                    temporaries.difference_update(
                        self.detag_word(new_analysis.analysis))
                self._update_counts(best.transform.change_counts, 1)
                if self._changed_segmentations is not None:
                    self._changed_segmentations.update(best.targets)
                    self._changed_segmentations_op.update(best.targets)
                if self._supervised:
                    for morph in changed_morphs:
                        self._annot_coding.modify_contribution(morph, 1)

            self._morph_usage.remove_temporaries(temporaries)
            msg = 'Operation incresed the model cost'
            assert self.get_cost() < old_cost + 0.1, msg

    ### Private: secondary
    #
    def _interned_morph(self, morph, store=False):
        """A homebrew approximation of interning,
        to reduce memory footprint of unicode strings with same content.
        Pythons builtin intern functionality is not used,
        because Python 2 does not allow interning of unicode strings.
        In a Python 3 only version this could be simplified.
        """
        if morph in self._interned_morphs:
            return self._interned_morphs[morph]
        if store:
            self._interned_morphs[morph] = morph
        return morph

    def _intern_word(self, cmorphs):
        # This intentionally violates the immutability of CategorizedMorph.
        for cmorph in cmorphs:
            cmorph.morph = self._interned_morph(
                cmorph.morph, store=True)

    def _intern_corpus(self):
        self._interned_morphs.clear()
        for word in self.segmentations:
            self._intern_word(word.analysis)

    def _test_skip(self, word):
        """Return true if word instance should be skipped."""
        if not self._online:
            return False
        if word in self._skipcounter:
            t = self._skipcounter[word]
            if random.random() > 1.0 / max(1, t):
                return True
        self._skipcounter[word] += 1
        return False

    def _resolve_operation(self, op_number):
        """Returns an object method corresponding to the given
        operation number."""
        operation_name = '_op_{}_generator'.format(
            self.training_operations[op_number])
        try:
            operation = self.__getattribute__(operation_name)
        except AttributeError:
            raise InvalidOperationError(
                self.training_operations[op_number],
                operation_name)
        return operation

    def _display_cost(self, cost_diff, limit, iteration_name,
                      iteration, max_iterations, conv_str):
        msg = ('Cost difference {:' +
                    self._cost_field_fmt(cost_diff) + 'f} ' +
                '(limit {}) ' +
                'in {:9s} {:2d}/{:<2d} {}')
        if len(conv_str) > 0:
            conv_str = '({})'.format(conv_str)
        _logger.info(msg.format(cost_diff, limit,
                                iteration_name, iteration + 1,
                                max_iterations, conv_str))

    def _cost_convergence_limit(self, min_cost_gain=0.005):
        if min_cost_gain is None:
            return None
        return min_cost_gain * self._corpus_coding.boundaries

    def _training_params(self, param_name):
        """Parameters for the training operators
        (calls to _convergence_of_cost).
        Can depend on the _epoch_number and _operation_number.
        Customize this if you need more finegrained control of the training.
        """

        if param_name == 'min_iteration_cost_gain':
            return self._min_iteration_cost_gain
        if param_name == 'min_epoch_cost_gain':
            return self._min_epoch_cost_gain
        if param_name == 'max_iterations':
            if (self.training_operations[self._operation_number] ==
                'resegment'):
                return self._max_resegment_iterations
            if self._epoch_number == 1:
                # Perform more iterations in the first epoch.
                # 0 is pre-epoch, 1 is the first actual epoch.
                return self._max_iterations_first
            # After that do just one of each operation.
            return self._max_iterations
        # FIXME This is a bit of ugliness I hope to get rid of
        if param_name == 'must_reestimate':
            return (self.training_operations[self._operation_number] ==
                    'resegment')

    def _find_in_corpus(self, rule, targets=None):
        """Returns the indices of words in the corpus segmentation
        matching the given rule, and the total number of matches, which
        can be larger than the sum of counts of matched_targets, if there
        are several matches in some word(s).

        Arguments:
            rule :  A TransformationRule describing the criteria for a match.
            targets :  A set of indices to limit the search to, or None to
                       search all segmentations. Default: full search.
        """

        if targets is None:
            targets = range(len(self.segmentations))

        matched_targets = set()
        num_matches = 0
        for target in targets:
            old_analysis = self.segmentations[target]
            tmp_matches = (old_analysis.count *
                            rule.num_matches(
                                old_analysis.analysis))
            if tmp_matches > 0:
                matched_targets.add(target)
                num_matches += tmp_matches
        return matched_targets, num_matches

    def _viterbi_analyze_corpus(self):
        """(Re)segments the corpus using viterbi_analyze"""
        num_changed_words = 0
        for (i, word) in enumerate(self.segmentations):
            self.segmentations[i] = WordAnalysis(word.count,
                self.viterbi_analyze(word.analysis)[0])
            if word != self.segmentations[i]:
                num_changed_words += 1
        self.reestimate_probabilities()
        self._calculate_morph_backlinks()
        return num_changed_words

    def _cost_field_fmt(self, cost):
        current = len(str(int(cost))) + self._cost_field_precision + 1
        if current > self._cost_field_width:
            self._cost_field_width = current
        return '{}.{}'.format(self._cost_field_width,
                              self._cost_field_precision)

    def __contains__(self, morph):
        return morph in self._morph_usage

    @property
    def num_compounds(self):
        """Compound (word) types"""
        return len(self.segmentations)

    @property
    def num_constructions(self):
        """Construction (morph) types"""
        return len(self._morph_usage.seen_morphs())


class FlatcatLexiconEncoding(baseline.LexiconEncoding):
    """Extends LexiconEncoding to include the coding costs of the
    encoding cost of morph usage (context) features.

    Arguments:
        morph_usage :  A MorphUsageProperties object,
                       or something that quacks like it.
    """

    def __init__(self, morph_usage):
        super(FlatcatLexiconEncoding, self).__init__()
        self._morph_usage = morph_usage
        self.logfeaturesum = 0.0

    def clear(self):
        """Resets the cost variables.
        Use before fully reprocessing a segmented corpus."""
        self.logtokensum = 0.0
        self.logfeaturesum = 0.0
        self.tokens = 0
        self.boundaries = 0
        self.atoms.clear()

    def add(self, morph):
        super(FlatcatLexiconEncoding, self).add(morph)
        self.logfeaturesum += self._morph_usage.feature_cost(morph)

    def remove(self, morph):
        super(FlatcatLexiconEncoding, self).remove(morph)
        self.logfeaturesum -= self._morph_usage.feature_cost(morph)

    def get_cost(self):
        assert self.boundaries >= 0
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum
                 + self.permutations_cost()
                 + self.logfeaturesum
                )  # * self.weight       # always 1
                + self.frequency_distribution_cost())

    def get_codelength(self, morph):
        cost = super(FlatcatLexiconEncoding, self).get_codelength(morph)
        cost += self._morph_usage.feature_cost(morph)
        return cost


class FlatcatEncoding(baseline.CorpusEncoding):
    """Class for calculating the encoding costs of the grammar and the
    corpus. Also stores the HMM parameters.

    tokens: the number of emissions observed.
    boundaries: the number of word tokens observed.
    """

    def __init__(self, morph_usage, lexicon_encoding, weight=1.0):
        self._morph_usage = morph_usage
        super(FlatcatEncoding, self).__init__(lexicon_encoding, weight)

        # Counts of emissions observed in the tagged corpus.
        # A dict of ByCategory objects indexed by morph. Counts occurences.
        self._emission_counts = utils.Sparse(
            default=utils._nt_zeros(ByCategory))

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

        # Counts of observed category tags.
        # Single Counter object (ByCategory is unsuitable, need break also).
        self._cat_tagcount = collections.Counter()

        # Caches for transition and emission logprobs,
        # to avoid wasting effort recalculating.
        self._log_transitionprob_cache = dict()
        self._log_emissionprob_cache = dict()
        self._persistent_log_emissionprob_cache = dict()
        # How frequent must a morph be to count as frequent
        self._persistence_limit = 3
        self._cache_size = 75000
        # Needed very often, reducing function calls
        self._categories = get_categories()

        self.logcondprobsum = 0.0

    # Transition count methods

    def get_transition_count(self, prev_cat, next_cat):
        return self._transition_counts[(prev_cat, next_cat)]

    def log_transitionprob(self, prev_cat, next_cat):
        """-Log of transition probability P(next_cat|prev_cat)"""
        pair = (prev_cat, next_cat)
        if pair not in self._log_transitionprob_cache:
            if self._cat_tagcount[prev_cat] == 0:
                self._log_transitionprob_cache[pair] = LOGPROB_ZERO
            else:
                self._log_transitionprob_cache[pair] = (
                    zlog(self._transition_counts[(prev_cat, next_cat)]) -
                    zlog(self._cat_tagcount[prev_cat]))
        # Assertion disabled due to performance hit
        #msg = 'transition {} -> {} has probability > 1'.format(
        #    prev_cat, next_cat)
        #assert self._log_transitionprob_cache[pair] >= 0, msg
        return self._log_transitionprob_cache[pair]

    def update_transition_count(self, prev_cat, next_cat, diff_count):
        """Updates the number of observed transitions between
        categories.
        OBSERVE! Clearing the cache is left to the caller.

        Arguments:
            prev_cat :  The name (not index) of the category
                        transitioned from.
            next_cat :  The name (not index) of the category
                        transitioned to.
            diff_count :  The change in the number of transitions.
        """

        # Assertion disabled due to performance hit
        #msg = 'update_transition_count needs category names, not indices'
        #assert not isinstance(prev_cat, int), msg
        #assert not isinstance(next_cat, int), msg
        pair = (prev_cat, next_cat)

        self._transition_counts[pair] += diff_count
        self._cat_tagcount[prev_cat] += diff_count

        # Assertion disabled due to performance hit
        #if self._transition_counts[pair] > 0:
        #    assert pair not in MorphUsageProperties.zero_transitions

        # Assertion disabled due to performance hit
        #msg = 'subzero transition count for {}'.format(pair)
        #assert self._transition_counts[pair] >= 0, msg
        #assert self._cat_tagcount[prev_cat] >= 0

    def clear_transition_counts(self):
        """Resets transition counts, costs and cache.
        Use before fully reprocessing a tagged segmented corpus."""
        self._transition_counts.clear()
        self._cat_tagcount.clear()
        self._log_transitionprob_cache.clear()

    # Emission count methods

    def get_emission_counts(self, morph):
        return self._emission_counts[morph]

    def log_emissionprob(self, category, morph, extrazero=False):
        """-Log of posterior emission probability P(morph|category)"""
        cat_index = self._categories.index(category)
        value = self._emission_helper(morph)[cat_index]
        # Assertion disabled due to performance hit
        #msg = 'emission {} -> {} has probability > 1'.format(category, morph)
        #assert value >= 0, msg
        if extrazero and value >= LOGPROB_ZERO:
            return value ** 2
        return value

    def _emission_helper(self, morph):
        if morph in self._persistent_log_emissionprob_cache:
            return self._persistent_log_emissionprob_cache[morph]
        if morph in self._log_emissionprob_cache:
            return self._log_emissionprob_cache[morph]
        count = self._morph_usage.count(morph)
        zlcount = zlog(count)
        zlctc = self._morph_usage.zlog_category_token_count()
        condprobs = self._morph_usage.condprobs(morph)
        tmp = []
        for (cat_index, cat) in enumerate(self._categories):
            # Not equal to what you get by:
            # zlog(self._emission_counts[morph][cat_index]) +
            if self._cat_tagcount[cat] == 0 or count == 0:
                value = LOGPROB_ZERO
            else:
                value = (zlcount +
                         zlog(condprobs[cat_index]) -
                         zlctc[cat_index])
            tmp.append(value)
        tmp = ByCategory(*tmp)
        if count >= self._persistence_limit:
            if len(self._persistent_log_emissionprob_cache) > self._cache_size:
                # Dont let the cache grow too big
                self._persistent_log_emissionprob_cache.clear()
                self._persistence_limit += 1
            self._persistent_log_emissionprob_cache[morph] = tmp
            return tmp
        if len(self._log_emissionprob_cache) > 10:
            # Small cache regularly emptied
            self._log_emissionprob_cache.clear()
        self._log_emissionprob_cache[morph] = tmp
        return tmp

    def update_emission_count(self, category, morph, diff_count):
        """Updates the number of observed emissions of a single morph from a
        single category, and the logtokensum (which is category independent).
        Updates logcondprobsum.

        Arguments:
            category :  name of category from which emission occurs.
            morph :  string representation of the morph.
            diff_count :  the change in the number of occurences.
        """
        if diff_count == 0:
            return
        assert category is not None
        cat_index = self._categories.index(category)
        old_count = self._emission_counts[morph][cat_index]
        new_count = old_count + diff_count
        logcondprob = -zlog(self._morph_usage.condprobs(morph)[cat_index])
        if old_count > 0:
            self.logcondprobsum -= old_count * logcondprob
        if new_count > 0:
            self.logcondprobsum += new_count * logcondprob
        new_counts = self._emission_counts[morph]._replace(
            **{category: new_count})
        self._set_emission_counts(morph, new_counts)

        # cached probabilities no longer valid
        self.clear_emission_cache()

    def _set_emission_counts(self, morph, new_counts):
        """Set the number of emissions of a morph from all categories
        simultaneously.
        Does not update logcondprobsum.

        Arguments:
            morph :  string representation of the morph.
            new_counts :  ByCategory object with new counts.
        """

        old_total = sum(self._emission_counts[morph])
        self._emission_counts[morph] = new_counts
        new_total = sum(new_counts)

        if old_total > 0:
            if old_total > 1:
                self.logtokensum -= old_total * math.log(old_total)
            self.tokens -= old_total
        if new_total > 0:
            if new_total > 1:
                self.logtokensum += new_total * math.log(new_total)
            self.tokens += new_total

        # cached probabilities no longer valid
        self.clear_emission_cache()

    def clear_emission_counts(self):
        """Resets emission counts and costs.
        Use before fully reprocessing a tagged segmented corpus."""
        self.tokens = 0
        self.logtokensum = 0.0
        self.logcondprobsum = 0.0
        self._emission_counts.clear()
        self._persistent_log_emissionprob_cache.clear()
        self._log_emissionprob_cache.clear()

    def clear_emission_cache(self):
        """Clears the cache for emission probability values.
        Use if an incremental change invalidates cached values."""
        self._persistent_log_emissionprob_cache.clear()
        self._log_emissionprob_cache.clear()

    def clear_transition_cache(self):
        """Clears the cache for emission probability values.
        Use if an incremental change invalidates cached values."""
        self._log_transitionprob_cache.clear()

    # General methods

    def transit_emit_cost(self, prev_cat, next_cat, morph):
        """Cost of transitioning from prev_cat to next_cat and emitting
        the morph."""
        if (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
            return LOGPROB_ZERO
        return (self.log_transitionprob(prev_cat, next_cat) +
                self.log_emissionprob(next_cat, morph))

    def update_count(self, construction, old_count, new_count):
        raise Exception('Inherited method not appropriate for FlatcatEncoding')

    def logtransitionsum(self):
        """Returns the term of the cost function associated with the
        transition probabilities. This term is recalculated on each call
        to get_cost, as the transition matrix is small and
        each segmentation change is likely to modify
        a large part of the transition matrix,
        making cumulative updates unnecessary.
        """
        categories = get_categories(wb=True)
        t_cost = 0.0
        # FIXME: this can be optimized using the same running tally
        # as logtokensum, when getting rid of the assertions
        # except if implementing hierarchy: then the incoming == outgoing
        # assumption doesn't necessarily hold anymore
        sum_transitions_from = collections.Counter()
        sum_transitions_to = collections.Counter()
        forbidden = MorphUsageProperties.zero_transitions
        for prev_cat in categories:
            for next_cat in categories:
                if (prev_cat, next_cat) in forbidden:
                    continue
                count = self._transition_counts[(prev_cat, next_cat)]
                if count == 0:
                    continue
                sum_transitions_from[prev_cat] += count
                sum_transitions_to[next_cat] += count
                t_cost += count * math.log(count)
        for cat in categories:
            # These hold, because for each incoming transition there is
            # exactly one outgoing transition (except for word boundary,
            # of which there are one of each in every word)
            assert sum_transitions_from[cat] == sum_transitions_to[cat]
            assert sum_transitions_to[cat] == self._cat_tagcount[cat]

        assert t_cost >= 0
        return t_cost

    def get_cost(self):
        """Override for the Encoding get_cost function.

        This is P( D_W | theta, Y )
        """
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((self.tokens * math.log(self.tokens)
                 - self.logtokensum
                 - self.logcondprobsum
                 - self.logtransitionsum()
                 + n * math.log(n)
                ) * self.weight
                + self.frequency_distribution_cost()
               )


class FlatcatAnnotatedCorpusEncoding(object):
    """Class for calculating the cost of encoding the annotated corpus"""
    def __init__(self, corpus_coding, weight=None):
        self.corpus_coding = corpus_coding
        if weight is None:
            self.weight = 1.0
            self.do_update_weight = True
        else:
            self.weight = weight
            self.do_update_weight = False
        self.logemissionsum = 0.0
        self.boundaries = 0

        # Counts of emissions observed in the tagged corpus.
        # A dict of ByCategory objects indexed by morph. Counts occurences.
        self._emission_counts = utils.Sparse(
            default=utils._nt_zeros(ByCategory))

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

    def set_counts(self, counts):
        """Sets the counts of emissions and transitions occurring
        in the annotated corpus to precalculated values."""
        self._emission_counts = utils.Sparse(
            default=utils._nt_zeros(ByCategory))
        self._transition_counts = collections.Counter()

        for (cmorph, new_count) in counts.emissions.items():
            assert new_count >= 0
            new_counts = self._emission_counts[cmorph.morph]._replace(
                **{cmorph.category: new_count})
            self._emission_counts[cmorph.morph] = new_counts

        for (pair, count) in counts.transitions.items():
            self._transition_counts[pair] = count
            assert self._transition_counts[pair] >= 0

    def update_counts(self, counts):
        """Updates the counts of emissions and transitions occurring
        in the annotated corpus, building on earlier counts."""
        for (cmorph, delta) in counts.emissions.items():
            cat_index = get_categories().index(cmorph.category)
            new_count = self._emission_counts[cmorph.morph][cat_index] + delta
            assert new_count >= 0
            new_counts = self._emission_counts[cmorph.morph]._replace(
                **{cmorph.category: new_count})
            self._emission_counts[cmorph.morph] = new_counts

        for (pair, delta) in counts.transitions.items():
            self._transition_counts[pair] += delta
            assert self._transition_counts[pair] >= 0

    def reset_contributions(self):
        """Recalculates the contributions of all morphs."""
        self.logemissionsum = 0.0
        categories = get_categories()
        for (morph, counts) in self._emission_counts.items():
            for (i, category) in enumerate(categories):
                msg = 'Annotation emission {} -> {} was subzero {}'.format(
                    category, morph, counts[i])
                assert counts[i] >= 0, msg
                self._contribution_helper(morph, category, counts[i])

    def modify_contribution(self, morph, direction):
        """Removes or readds the complete contribution of a morph to the
        cost function. The contribution must be removed using the same
        probability value as was used when adding it, making ordering of
        operations important.
        """
        categories = get_categories()
        counts = self._emission_counts[morph]
        for (i, category) in enumerate(categories):
            self._contribution_helper(morph, category, counts[i] * direction)

    def transition_cost(self):
        """Returns the term of the cost function associated with the
        transition probabilities. This term is recalculated on each call
        to get_cost, as the transition matrix is small and
        each segmentation change is likely to modify
        a large part of the transition matrix,
        making cumulative updates unnecessary.
        """
        cost = 0.0
        valid_transitions = MorphUsageProperties.valid_transitions()
        for pair in valid_transitions:
            count = self._transition_counts[pair]
            cost += count * self.corpus_coding.log_transitionprob(*pair)
        return cost

    def get_cost(self):
        """Returns the cost of encoding the annotated corpus"""
        if self.boundaries == 0:
            return 0.0
        tc = self.transition_cost()
        assert self.logemissionsum >= 0
        assert tc >= 0
        return (self.logemissionsum + tc) * self.weight

    def update_weight(self):
        """Update the weight of the Encoding by taking the ratio of the
        corpus boundaries and annotated boundaries.
        Does not scale by corpus weight,, unlike Morfessor Baseline.
        """
        if not self.do_update_weight:
            return
        old = self.weight
        self.weight = float(self.corpus_coding.boundaries) / self.boundaries
        if self.weight != old:
            _logger.info('Corpus weight of annotated data set to {}'.format(
                         self.weight))

    def _contribution_helper(self, morph, category, count):
        if count == 0:
            return
        self.logemissionsum += count * self.corpus_coding.log_emissionprob(
            category, morph, extrazero=True)


class ChangeCounts(object):
    """A data structure for the aggregated set of changes to
    emission and transition counts and morph backlinks.
    Used to reduce the number of model updates and to make
    reverting changes easier.
    """

    __slots__ = ['emissions', 'transitions',
                 'backlinks_remove', 'backlinks_add']

    def __init__(self, emissions=None, transitions=None):
        if emissions is None:
            self.emissions = collections.Counter()
        else:
            self.emissions = emissions
        if transitions is None:
            self.transitions = collections.Counter()
        else:
            self.transitions = transitions
        self.backlinks_remove = collections.defaultdict(set)
        self.backlinks_add = collections.defaultdict(set)

    def update(self, analysis, count, corpus_index=None):
        """Updates the counts to add or remove the effects of an analysis.

        Arguments:
            analysis :  A tuple of CategorizedMorphs.
            count :  The occurence count of the analyzed word.
                     A negative count removes the effects of the analysis.
            corpus_index :  If not None, the mapping between the morphs
                            and the indices of words in the corpus that they
                            occur in will be updated. corpus_index is then
                            the index of the current occurence being updated.
        """

        for cmorph in analysis:
            self.emissions[cmorph] += count
            if corpus_index is not None:
                if count < 0:
                    self.backlinks_remove[cmorph.morph].add(corpus_index)
                elif count > 0:
                    self.backlinks_add[cmorph.morph].add(corpus_index)
        wb_extended = _wb_wrap(analysis)
        for (prefix, suffix) in utils.ngrams(wb_extended, n=2):
            self.transitions[(prefix.category, suffix.category)] += count
        # Make sure that backlinks_remove and backlinks_add are disjoint
        # Removal followed by readding is the same as just adding
        for morph in self.backlinks_add:
            self.backlinks_remove[morph].difference_update(
                self.backlinks_add[morph])


class TransformationRule(object):
    """A simple transformation rule that requires a pattern of category
    and/or morph matches. Don't care -values are marked by None.
    This simple rule does not account for context outside the part to
    be replaced.
    """

    def __init__(self, categorized_morphs, context_type=None):
        if isinstance(categorized_morphs, CategorizedMorph):
            categorized_morphs = (categorized_morphs,)
        self._rule = categorized_morphs
        self._context_type = context_type

    def __len__(self):
        return len(self._rule)

    def __iter__(self):
        return iter(self._rule)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._rule)

    def match_at(self, analysis, i):
        """Returns true if this rule matches the analysis
        at the given index."""
        # Compare morphs and categories specified in rule
        for (j, cmorph) in enumerate(analysis[i:(i + len(self))]):
            if self._rule[j].category is not None:
                # Rule requires category at this point to match
                if self._rule[j].category != cmorph.category:
                    return False
            if self._rule[j].morph is not None:
                # Rule requires morph at this point to match
                if self._rule[j].morph != cmorph.morph:
                    return False
        # Compare context type
        if self._context_type is not None:
            if i <= 0:
                prev_morph = WORD_BOUNDARY
                prev_category = WORD_BOUNDARY
            else:
                prev_morph = analysis[i - 1].morph
                prev_category = analysis[i - 1].category
            if (i + len(self)) >= len(analysis):
                next_morph = WORD_BOUNDARY
                next_category = WORD_BOUNDARY
            else:
                next_morph = analysis[i + len(self)].morph
                next_category = analysis[i + len(self)].category
            context_type = MorphUsageProperties.context_type(
                                prev_morph, next_morph,
                                prev_category, next_category)
            if self._context_type != context_type:
                return False

        # No comparison failed
        return True

    def num_matches(self, analysis):
        """Total number of matches of this rule in the analysis.
        Greedy application of the rule is used."""
        i = 0
        matches = 0
        while i + len(self) <= len(analysis):
            if self.match_at(analysis, i):
                i += len(self)
                matches += 1
            else:
                i += 1
        return matches


class Transformation(object):
    """A transformation of a certain pattern of morphs and/or categories,
    to be (partially) replaced by another representation.
    """
    __slots__ = ['rule', 'result', 'change_counts']

    def __init__(self, rule, result):
        self.rule = rule
        if isinstance(result, CategorizedMorph):
            self.result = (result,)
        else:
            self.result = result
        self.change_counts = ChangeCounts()

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.rule, self.result)

    def apply(self, word, model, corpus_index=None):
        """Tries to apply this transformation to an analysis.
        If the transformation doesn't match, the input is returned unchanged.
        If the transformation matches, changes are made greedily from the
        beginning of the analysis, until the whole sequence has been
        processed. After this the segmentation is retagged using viterbi
        tagging with the given model.

        Arguments:
            word :  A WordAnalysis object.
            model :  The current model to use for tagging.
            corpus_index :  Index of the word in the corpus, or None if
                            the change is temporary and morph to word
                            backlinks don't need to be updated.
        """
        i = 0
        out = []
        matches = 0
        while i + len(self.rule) <= len(word.analysis):
            if self.rule.match_at(word.analysis, i):
                out.extend(self.result)
                i += len(self.rule)
                matches += 1
            else:
                out.append(word.analysis[i])
                i += 1
        while i < len(word.analysis):
            out.append(word.analysis[i])
            i += 1

        if matches > 0:
            # Only retag if the rule matched something
            out = model.fast_tag_gaps(out)
            #out = model.viterbi_tag(out)

            self.change_counts.update(word.analysis, -word.count,
                                      corpus_index)
            self.change_counts.update(out, word.count, corpus_index)

        return WordAnalysis(word.count, tuple(out))

    def reset_counts(self):
        self.change_counts = ChangeCounts()


class ViterbiResegmentTransformation(object):
    """Special transformation that resegments and tags
    words in the corpus using viterbi_analyze.
    """

    def __init__(self, word, model):
        self.rule = TransformationRule(tuple(word.analysis))
        self.result, _ = model.viterbi_analyze(word.analysis)
        self.change_counts = ChangeCounts()

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.rule, self.result)

    def apply(self, word, model, corpus_index=None):
        """Apply the new segmentation ot the counts.
        Note that the segmentation was performed already at __init__,
        which means that the morph count changes between the beginning
        of the _operation_loop loop
        and the call to apply do not affect the segmentation.
        """
        if self.rule.num_matches(word.analysis) == 0:
            return word
        self.change_counts.update(word.analysis, -word.count,
                                    corpus_index)
        self.change_counts.update(self.result, word.count, corpus_index)
        return WordAnalysis(word.count, self.result)

    def reset_counts(self):
        self.change_counts = ChangeCounts()


class CostBreakdown(object):
    """Helper for utility functions cost_breakdown and rank_analyses"""
    def __init__(self):
        self.components = []
        self.cost = 0

    def __repr__(self):
        out = '\n'.join(self.components) + '\n'
        return 'CostBreakdown(\n{})'.format(out)

    def transition(self, cost, prev_cat, next_cat):
        self.cost += cost
        self.components.append('transition {:3s} -> {:7s}: {}'.format(
            prev_cat, next_cat, cost))

    def emission(self, cost, cat, morph):
        self.cost += cost
        self.components.append('emission   {:3s} :: {:7s}: {}'.format(
            cat, morph, cost))

    def penalty(self, cost):
        self.cost += cost
        self.components.append('penalty: {}'.format(cost))


class ForceSplitter(object):
    def __init__(self, forcesplit, nosplit_re):
        self.forcesplit = forcesplit
        self.nosplit_re = nosplit_re

    def enforce(self, segmentations):
        segmentation_changed = False
        if self.forcesplit:
            mapper = SegmentationMapper(self._enforce_forcesplit)
            segmentations = mapper.map(segmentations)
            segmentation_changed = (segmentation_changed
                                    or mapper.segmentation_changed)
        if self.nosplit_re:
            mapper = SegmentationMapper(self._enforce_nosplit)
            segmentations = mapper.map(segmentations)
            segmentation_changed = (segmentation_changed
                                    or mapper.segmentation_changed)
        return list(segmentations), segmentation_changed

    def enforce_one(self, analysis):
        if self.forcesplit:
            analysis = self._enforce_forcesplit(analysis)
        if self.nosplit_re:
            analysis = self._enforce_nosplit(analysis)
        return analysis

    def _enforce_forcesplit(self, analysis):
        out = []
        for cmorph in analysis:
            if len(cmorph) == 1:
                out.append(cmorph)
                continue
            j = 0
            for i in range(1, len(cmorph)):
                if cmorph[i] in self.forcesplit:
                    if len(cmorph[j:i]) > 0:
                        out.append(self._part(cmorph, j, i))
                    out.append(self._part(cmorph, i, i + 1))
                    j = i + 1
            if j < len(cmorph):
                out.append(self._part(cmorph, j, len(cmorph)))
        return out

    def _enforce_nosplit(self, analysis):
        out = []
        prev = None
        for cmorph in analysis:
            if prev is None:
                prev = cmorph
                continue
            if self.nosplit_re.match(prev[-1:] + cmorph[0]):
                prev = CategorizedMorph(prev.morph + cmorph.morph,
                                        DEFAULT_CATEGORY)
            else:
                out.append(prev)
                prev = cmorph
        out.append(prev)
        return out

    def _part(self, cmorph, j, i):
        return CategorizedMorph(cmorph[j:i], cmorph.category)


class SegmentationMapper(object):
    def __init__(self, func):
        self.func = func
        self.segmentation_changed = False

    def map(self, segmentations):
        """Apply a mapping to the analysis part of segmentations.
        Convenience function."""
        for word in segmentations:
            newseg = self.func(word.analysis)
            if not self.segmentation_changed and newseg != word.analysis:
                self.segmentation_changed = True
            yield WordAnalysis(word.count, newseg)


def _log_catprobs(probs):
    """Convenience function to convert a ByCategory object containing actual
    probabilities into one with log probabilities"""

    return ByCategory(*[zlog(x) for x in probs])


def _wb_wrap(segments, end_only=False):
    """Add a word boundary CategorizedMorph at one or both ends of
    the segmentation.
    """
    wb = CategorizedMorph(WORD_BOUNDARY, WORD_BOUNDARY)
    if end_only:
        return tuple(list(segments) + [wb])
    else:
        return tuple([wb] + list(segments) + [wb])
