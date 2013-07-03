#!/usr/bin/env python
"""
Morfessor 2.0 Categories-MAP variant.
"""
from __future__ import unicode_literals

# Temporarily disabled to enable from catmap import * in interactive shell,
# Which is a workaround for the pickle namespace problem
#__all__ = ['CatmapIO', 'CatmapModel']

__author__ = 'Stig-Arne Gronroos'
__author_email__ = "morfessor@cis.hut.fi"

import collections
import logging
import math
import sys
import time

from . import baseline
from . import utils
from .categorizationscheme import MorphUsageProperties, WORD_BOUNDARY
from .categorizationscheme import ByCategory, get_categories, CategorizedMorph
from .categorizationscheme import DEFAULT_CATEGORY
from .exception import InvalidOperationError
from .utils import LOGPROB_ZERO, zlog

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)
_logger.level = logging.DEBUG   # FIXME development convenience

# Grid node for viterbi algorithm
ViterbiNode = collections.namedtuple('ViterbiNode', ['cost', 'backpointer'])

WordAnalysis = collections.namedtuple('WordAnalysis', ['count', 'analysis'])

AnalysisAlternative = collections.namedtuple('AnalysisAlternative',
                                             ['analysis', 'penalty'])

SortedAnalysis = collections.namedtuple('SortedAnalysis',
                                        ['cost', 'analysis',
                                         'index', 'brakdown'])


def train_batch(model, weight_learn_func=None):
    model._iteration_update(no_increment=True)
    previous_cost = model.get_cost()
    wl_force_another = False
    u_force_another = False
    for iteration in range(model._max_iterations):
        model._train_iteration()

        cost = model.get_cost()
        cost_diff = cost - previous_cost
        limit = model._cost_convergence_limit(model._min_iter_cost_gain)

        if weight_learn_func is not None:
            (model, wl_force_another) = weight_learn_func(model)
        u_force_another = model._iteration_update()

        converged = ((not wl_force_another) and
                     (not u_force_another) and
                     (-cost_diff <= limit))
        model._display_cost(cost_diff, limit, 'iteration',
                        iteration, model._max_iterations, converged)
        if converged:
            _logger.info('{:24s} Cost: {}'.format(
                'final iteration.', cost))
            break
        previous_cost = cost
    return model


class CatmapModel(object):
    """Morfessor Categories-MAP model class."""

    word_boundary = WORD_BOUNDARY

    DEFAULT_TRAIN_OPS = ['split', 'join', 'shift', 'resegment']

    def __init__(self, morph_usage, forcesplit=None,
                 corpusweight=1.0):
        """Initialize a new model instance.

        Arguments:
            morph_usage -- A MorphUsageProperties object describing how
                           the usage of a morph affects the category.
        """

        self._morph_usage = morph_usage

        # The analyzed (segmented and tagged) corpus
        self.segmentations = []

        # Morph occurence backlinks
        # A dict of sets. Keys are morphs, set contents are indices to
        # self.segmentations for words in which the morph occurs
        self.morph_backlinks = collections.defaultdict(set)

        # Cost variables
        self._lexicon_coding = CatmapLexiconEncoding(morph_usage)
        # Catmap encoding also stores the HMM parameters
        self._corpus_coding = CatmapEncoding(morph_usage, self._lexicon_coding,
                                             weight=corpusweight)

        # Counters for the current iteration and operation within
        # that iteration. These describe the stage of training
        # to allow resuming training of a pickled model.
        # The exact point in training is described by 3 numbers:
        #   - the epoch number (each epoch is one pass over the data
        #     while performing one type of operation).
        #     The epoch number is not restored when loading.
        #   - the operation number (epochs performing the same operation
        #     are repeated until convergence, before moving to
        #     the next operation)
        #   - the iteration number (an iteration consists of the sequence
        #     of all training operations)
        self._iteration_number = 0
        self._operation_number = 0

        # The sequence of training operations.
        # Valid training operations are strings for which CatmapModel
        # has a function named _op_X_generator, where X is the string
        # which returns a transform generator suitable for
        # passing to _transformation_epoch.
        # This is done using strings indstead of bound methods,
        # to enable pickling of the model object.
        self.training_operations = self.DEFAULT_TRAIN_OPS

        # Training sequence parameters.
        self._max_iterations = 10
        self._min_iter_cost_gain = 0.0
        self._min_epoch_cost_gain = 0.0
        self._max_epochs_first = 1
        self._max_epochs = 1
        self._max_resegment_epochs = 1
        self._min_shift_remainder = 2
        self._max_shift = 2

        # Callbacks for cleanup/bookkeeping after each operation.
        # Should take exactly one argument: the model.
        self.operation_callbacks = []
        self.epoch_callbacks = []
        self._changed_segmentations = set()
        self._changed_segmentations_op = set()

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
        self.annotations = []           # (word, (analysis1, analysis2...))
        self._annotations_tagged = False

        # Variables for online learning
        self._online = False
        self.training_focus = None

        self._cost_field_width = 9
        self._cost_field_precision = 4

    def add_corpus_data(self, segmentations, freqthreshold=1,
                        count_modifier=None):
        """Adds the given segmentations (with counts) to the corpus data.
        The new data can be either untagged or tagged.

        If the added data is untagged, you must call viterbi_tag_corpus
        to tag the new data.

        You should also call reestimate_probabilities and consider
        calling initialize_hmm.

        Arguments:
            segmentations -- Segmentations of format:
                             (count, (morph1, morph2, ...))
                             where the morphs can be either strings
                             or CategorizedMorphs.
            freqthreshold -- discard compounds that occur less than
                             given times in the corpus (default 1).
            count_modifier -- function for adjusting the counts of each
                              compound.
        """
        assert isinstance(freqthreshold, (int, float))
        i = len(self.segmentations)
        for row in segmentations:
            count, analysis = row
            if count < freqthreshold:
                continue
            if count_modifier != None:
                count = count_modifier(count)
            segmentation = WordAnalysis(count, analysis)
            self.segmentations.append(segmentation)
            for morph in self.detag_word(segmentation.analysis):
                self.morph_backlinks[morph].add(i)
            i += 1
        utils.memlog('After adding corpus data')

    def initialize_baseline(self):
        """Initialize the model using a previously added
        (see add_corpus_data) segmentation produced by a morfessor
        baseline model.
        """

        self._calculate_usage_features()
        self._unigram_transition_probs()
        self.viterbi_tag_corpus()
        self._calculate_transition_counts()
        self._calculate_emission_counts()

    def initialize_hmm(self, min_difference_proportion=0.005):
        """Initialize emission and transition probabilities without
        changing the segmentation, using Viterbi EM.
        """

        def reestimate_with_unchanged_segmentation():
            self._calculate_transition_counts()
            self._calculate_emission_counts()

        self.convergence_of_analysis(
            reestimate_with_unchanged_segmentation,
            self.viterbi_tag_corpus,
            min_difference_proportion=min_difference_proportion,
            min_cost_gain=-10.0)     # Cost gain will be ~zero.

        for callback in self.epoch_callbacks:
            callback(self)

        self._iteration_number = 1

    def batch_parameters(self, min_epoch_cost_gain=0.0025,
                         min_iter_cost_gain=0.005,
                         max_iterations=10, max_epochs_first=1, max_epochs=1,
                         max_resegment_epochs=1,
                         max_shift_distance=2,
                         min_shift_remainder=2):
        """Set parameters for batch Cat-MAP training of the model."""
        self._min_epoch_cost_gain = min_epoch_cost_gain
        self._min_iter_cost_gain = min_iter_cost_gain
        self._max_iterations = max_iterations
        self._max_epochs_first = max_epochs_first
        self._max_epochs = max_epochs
        self._max_resegment_epochs = max_resegment_epochs
        self._max_shift = max_shift_distance
        self._min_shift_remainder = min_shift_remainder
        self._online = False

    def train_online(self, data, count_modifier=None, epoch_interval=10000,
                     max_epochs=None):
        """Adapt the model in online fashion."""

        self._online = True
        if count_modifier is not None:
            counts = {}
        word_backlinks = {}
        for (i, seg) in enumerate(self.segmentations):
            word_backlinks[''.join(self.detag_word(seg.analysis))] = i

        _logger.info("Starting online training")

        epochs = 0
        i = 0
        more_tokens = True
        self.reestimate_probabilities()
        while more_tokens:
            newcost = self.get_cost()
            _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))

            for _ in utils._progress(range(epoch_interval)):
                try:
                    _, _, w = next(data)
                except StopIteration:
                    more_tokens = False
                    break

                if count_modifier is not None:
                    if not w in counts:
                        c = 0
                        counts[w] = 1
                        addc = 1
                    else:
                        c = counts[w]
                        counts[w] = c + 1
                        addc = count_modifier(c + 1) - count_modifier(c)
                else:
                    addc = 1
                segments, _ = self.viterbi_segment(w)
                if addc > 0:
                    change_counts = ChangeCounts()
                    if w in word_backlinks:
                        i_new = word_backlinks[w]
                        old_seg = self.segmentations[i_new]
                        change_counts.update(old_seg.analysis,
                                             -old_seg.count,
                                             corpus_index=i_new)
                        for morph in self.detag_word(old_seg.analysis):
                            self._modify_morph_count(morph, -old_seg.count)
                        self.segmentations[i_new] = WordAnalysis(
                            old_seg.count + addc,
                            segments)
                    else:
                        self.add_corpus_data([WordAnalysis(addc, segments)])
                        i_new = len(self.segmentations) - 1
                        word_backlinks[w] = i_new
                    new_count = self.segmentations[i_new].count
                    change_counts.update(self.segmentations[i_new].analysis,
                                         new_count, corpus_index=i_new)
                    for morph in self.detag_word(segments):
                        self._modify_morph_count(morph, new_count)

                    self._update_counts(change_counts, 1)

                    self.training_focus = set((i_new,))
                    self._single_epoch_iteration()
                    segments = self.segmentations[i_new].analysis

                _logger.debug("#%s: %s -> %s" %
                              (i, w, segments))
                i += 1

            # also reestimates the probabilities
            _logger.info("Epoch reached, resegmenting corpus")
            self.viterbi_segment_corpus()

            epochs += 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break

        self.reestimate_probabilities()
        newcost = self.get_cost()
        _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))
        return epochs, newcost

    def _resolve_operation(self, op_number):
        operation_name = '_op_{}_generator'.format(
            self.training_operations[op_number])
        try:
            operation = self.__getattribute__(operation_name)
        except AttributeError:
            raise InvalidOperationError(
                self.training_operations[op_number],
                operation_name)
        return operation

    def _train_iteration(self):
        """One iteration of training, which may contain several epochs
        of each operation in sequence.

        The model must have been initialized, either by loading a baseline
        segmentation or a pretrained catmap model from pickle or tagged
        segmentation file, and calling initialize_baseline and/or
        initialize_hmm
        """
        cost = self.get_cost()
        msg = ('{:9s} {:2d}/{:<2d}          Cost: {:' +
                self._cost_field_fmt(cost) + 'f}.')
        _logger.info(msg.format('iteration',
                                self._iteration_number,
                                self._max_iterations,
                                cost))
        self._changed_segmentations = set()  # FIXME: use for convergence
        while self._operation_number < len(self.training_operations):
            operation = self._resolve_operation(self._operation_number)
            min_epoch_cost_gain = self._training_params('min_epoch_cost_gain')
            max_epochs = self._training_params('max_epochs')
            if self._training_params('must_reestimate'):
                update_func = self.reestimate_probabilities
            else:
                update_func = None

            msg = 'Iteration {:2d}, operation {:2d} ({}), max {:2d} epoch(s).'
            _logger.info(msg.format(
                    self._iteration_number, self._operation_number,
                    self.training_operations[self._operation_number],
                    max_epochs))
            utils.memlog('see above')
            self.convergence_of_cost(
                lambda: self._transformation_epoch(operation()),
                update_func=update_func,
                min_cost_gain=min_epoch_cost_gain,
                max_iterations=max_epochs)
            self.reestimate_probabilities()
            self._operation_number += 1
            for callback in self.operation_callbacks:
                callback(self)

    def _iteration_update(self, no_increment=False):
        force_another = False

        if self._supervised:
            old_cost = self.get_cost()
            self._update_annotation_choices()
            new_cost = self.get_cost()
            if old_cost != new_cost:
                _logger.info('Updated annotation choices, changing cost from '
                            '{} to {}'.format(old_cost, new_cost))
                force_another = True
            self._annot_coding.update_weight()
            utils.memlog('after annotation choice update')

        self._operation_number = 0
        if not no_increment:
            self._iteration_number += 1
        return force_another

    def convergence_of_cost(self, train_func, update_func,
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
            train_func -- A method of CatmapModel which causes some part of
                          the model to be trained.
            update_func -- Updates to the model between iterations,
                           that should not be considered in the convergence
                           analysis. However, if the return value is
                           True, at least one more iteration is forced unless
                           the maximum limit has been reached.
            min_cost_gain -- Stop iterating if cost reduction between
                             iterations is below this limit * #boundaries.
                             Default 0.005.
            max_iterations -- Maximum number of iterations (epochs). Default 5.
        """

        previous_cost = self.get_cost()
        for iteration in range(max_iterations):
            cost = self.get_cost()
            msg = ('{:9s} {:2d}/{:<2d}          Cost: {:' +
                   self._cost_field_fmt(cost) + 'f}.')
            _logger.info(msg.format('epoch',
                                    iteration + 1, max_iterations,
                                    cost))

            # perform the optimization
            train_func()

            # perform update between optimization iterations
            if update_func is not None:
                _logger.info('{:24s} Cost: {}'.format(
                    'Before epoch update.', cost))

            # perform update between optimization iterations
            if update_func is not None:
                force_another = update_func()
            else:
                force_another = False

            for callback in self.epoch_callbacks:
                callback(self, iteration)
            
            cost = self.get_cost()
            cost_diff = cost - previous_cost
            limit = self._cost_convergence_limit(min_cost_gain)
            converged = (not force_another) and -cost_diff <= limit
            self._display_cost(cost_diff, limit, 'epoch',
                           iteration, max_iterations, converged)
            if converged:
                _logger.info('{:24s} Cost: {}'.format(
                    'final epoch.', cost))
                return
            previous_cost = cost

    def _display_cost(self, cost_diff, limit, iteration_name,
                      iteration, max_iterations, converged):
        msg = ('Cost difference {:' +
                    self._cost_field_fmt(cost_diff) + 'f} ' +
                '(limit {}) ' +
                'in {:9s} {:2d}/{:<2d} {}')
        if converged:
            conv = '(Converged)'
        else:
            conv = ''
        _logger.info(msg.format(cost_diff, limit,
                                iteration_name, iteration + 1,
                                max_iterations, conv))
        
    def _cost_convergence_limit(self, min_cost_gain=0.005):
        return min_cost_gain * self._corpus_coding.boundaries

    def convergence_of_analysis(self, train_func, resegment_func,
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
            train_func -- A method of CatmapModel which causes some aspect
                          of the model to be trained.
            resegment_func -- A method of CatmapModel that resegments or
                              retags the segmentations, to produce the
                              results to compare. Should return the number
                              of changed words.
            min_difference_proportion -- Maximum proportion of words with
                                         changed segmentation or category
                                         tags in the final iteration.
                                         Default 0.
            min_cost_gain -- Stop iterating if cost reduction between
                                   iterations is below this limit.
            max_iterations -- Maximum number of iterations. Default 15.
        """

        previous_cost = self.get_cost()
        for iteration in range(max_iterations):
            _logger.info(
                'Iteration {:2d} ({}). {:2d}/{:<2d}'.format(
                    self._iteration_number, train_func.__name__,
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

    def _training_params(self, param_name):
        """Parameters for the training operators
        (calls to convergence_of_cost).
        Can depend on the _iteration_number and _operation_number.
        Customize this if you need more finegrained control of the training.
        """

        if param_name == 'min_epoch_cost_gain':
            return self._min_epoch_cost_gain
        if param_name == 'min_iter_cost_gain':
            return self._min_iter_cost_gain
        if param_name == 'max_epochs':
            if (self.training_operations[self._operation_number] ==
                'resegment'):
                return self._max_resegment_epochs
            if self._iteration_number == 1:
                # Perform more epochs in the first iteration.
                # 0 is pre-iteration, 1 is the first actual iteration.
                return self._max_epochs_first
            # After that do just one of each operation.
            return self._max_epochs
        # FIXME This is a bit of ugliness I hope to get rid of
        if param_name == 'must_reestimate':
            return (self.training_operations[self._operation_number] ==
                    'resegment')

    def _single_epoch_iteration(self):
        """One iteration of training, with exactly one epoch of each
        operation and no convergence checks or update passes."""
        for i in range(len(self.training_operations)):
            operation = self._resolve_operation(i)
            self._transformation_epoch(operation())

    def weightlearn_probe(self):
        self._single_epoch_iteration()
        self.reestimate_probabilities()
        self._iteration_update(no_increment=True)

    def reestimate_probabilities(self):
        """Re-estimates model parameters from a segmented, tagged corpus.

        theta(t) = arg min { L( theta, Y(t), D ) }
        """
        self._calculate_usage_features()
        self._calculate_transition_counts()
        self._calculate_emission_counts()

    def _calculate_usage_features(self):
        """Recalculates the morph usage features (perplexities).
        """

        self._lexicon_coding.clear()

        self._corpus_coding.boundaries = (
            self._morph_usage.calculate_usage_features(
                lambda: self.detag_list(self.segmentations)))

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

    def _calculate_transition_counts(self):
        """Count the number of transitions of each type.
        Can be used to estimate transition probabilities from
        a category-tagged segmented corpus.
        """

        self._corpus_coding.clear_transition_counts()
        for rcount, segments in self.segmentations:
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

    def _calculate_emission_counts(self):
        """Recalculates the emission counts from a retagged segmentation."""
        self._corpus_coding.clear_emission_counts()
        for (count, analysis) in self.segmentations:
            for morph in analysis:
                self._corpus_coding.update_emission_count(morph.category,
                                                          morph.morph,
                                                          count)

    def _calculate_morph_backlinks(self):
        self.morph_backlinks.clear()
        for (i, segmentation) in enumerate(self.segmentations):
            for morph in self.detag_word(segmentation.analysis):
                self.morph_backlinks[morph].add(i)

    def _find_in_corpus(self, rule, targets=None):
        """Returns the indices of words in the corpus segmentation
        matching the given rule, and the total number of matches, which
        can be larger than the sum of counts of matched_targets, if there
        are several matches in some word(s).

        Arguments:
            rule -- A TransformationRule describing the criteria for a match.
            targets -- A set of indices to limit the search to, or None to
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

    def _transformation_epoch(self, transformation_generator):
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
            transformation_generator -- a generator yielding
                (transform_group, targets, temporaries)
                tuples, where
                    transform_group -- a list of Transform objects.
                    targets -- a set with an initial guess of indices of
                               matching words in the corpus. Can contain false
                               positives, but should not omit positive
                               indices (they will be missed).
                    temporaries -- a set of new morphs with estimated
                                   contexts.
        """

        EpochNode = collections.namedtuple('EpochNode', ['cost',
                                                         'transform',
                                                         'targets'])
        self._changed_segmentations_op = set()  # FIXME
        if not self._online:
            transformation_generator = utils._generator_progress(
                transformation_generator)
        for experiment in transformation_generator:
            (transform_group, targets, temporaries) = experiment
            if len(transform_group) == 0:
                continue
            # Cost of doing nothing
            best = EpochNode(self.get_cost(), None, set())

            # All transforms in group must match the same words,
            # we can use just the first transform
            matched_targets, num_matches = self._find_in_corpus(
                transform_group[0].rule, targets)
            if num_matches == 0:
                continue

            for morph in self.detag_word(transform_group[0].rule):
                # Remove the old representation, but only from
                # morph counts (emissions and transitions updated later)
                self._modify_morph_count(morph, -num_matches)

            for transform in transform_group:
                for morph in self.detag_word(transform.result):
                    # Add the new representation to morph counts
                    self._modify_morph_count(morph, num_matches)
                for target in matched_targets:
                    is_annot = (self._supervised and
                                target <= len(self.annotations))
                    old_analysis = self.segmentations[target]
                    new_analysis = transform.apply(old_analysis,
                                                   self,
                                                   is_annot=is_annot)

                # Apply change to encoding
                self._update_counts(transform.change_counts, 1)
                cost = self.get_cost()
                if cost < best.cost:
                    best = EpochNode(cost, transform, matched_targets)
                # Revert change to encoding
                self._update_counts(transform.change_counts, -1)
                for morph in self.detag_word(transform.result):
                    self._modify_morph_count(morph, -num_matches)

            if best.transform is None:
                # Best option was to do nothing. Revert morph count.
                for morph in self.detag_word(transform_group[0].rule):
                    self._modify_morph_count(morph, num_matches)
            else:
                # A real change was the best option
                best.transform.reset_counts()
                for target in best.targets:
                    for morph in self.detag_word(best.transform.result):
                        # Add the new representation to morph counts
                        self._modify_morph_count(morph, num_matches)
                    is_annot = (self._supervised and
                                target <= len(self.annotations))
                    new_analysis = best.transform.apply(
                                        self.segmentations[target],
                                        self, corpus_index=target,
                                        is_annot=is_annot)
                    self.segmentations[target] = new_analysis
                    # any morph used in the best segmentation
                    # is no longer temporary
                    temporaries.difference_update(
                        self.detag_word(new_analysis.analysis))
                self._update_counts(best.transform.change_counts, 1)
                self._changed_segmentations.update(best.targets)
                self._changed_segmentations_op.update(best.targets)
            self._morph_usage.remove_temporaries(temporaries)

    def _op_split_generator(self):
        """Generates splits of seen morphs into two submorphs.
        Use with _transformation_epoch
        """
        # FIXME random shuffle or sort by length/frequency?
        if self.training_focus is None:
            unsorted = self._morph_usage.seen_morphs()
        else:
            unsorted = set()
            for (count, segmentation) in self.training_focus_filter():
                for morph in self.detag_word(segmentation):
                    unsorted.add(morph)
        epoch_morphs = sorted(unsorted, key=len)
        for morph in epoch_morphs:
            if len(morph) == 1:
                continue
            if self._morph_usage.count(morph) == 0:
                continue

            # Match the parent morph with any category
            rule = TransformationRule((CategorizedMorph(morph, None),))
            transforms = []
            # Apply to all words in which the morph occurs
            targets = self.morph_backlinks[morph]
            # Temporary estimated contexts
            temporaries = set()
            for splitloc in range(1, len(morph)):
                prefix = morph[:splitloc]
                suffix = morph[splitloc:]
                # Make sure that there are context features available
                # (real or estimated) for the submorphs
                tmp = (self._morph_usage.estimate_contexts(morph,
                                                           (prefix, suffix)))
                temporaries.update(tmp)
                transforms.append(
                    Transformation(rule,
                                   (CategorizedMorph(prefix, None),
                                    CategorizedMorph(suffix, None))))
            yield (transforms, targets, temporaries)

    def _generic_bimorph_generator(self, result_func):
        """The common parts of operation generators that operate on
        context-sensitive bimorphs. Don't call this directly.

        Arguments:
            result_func -- A function that takes the prefix an suffix
                           as arguments, and returns all the proposed results
                           as tuples of CategorizedMorphs.
        """

        # FIXME random shuffle or sort by bigram frequency?
        bigram_freqs = collections.Counter()
        for (count, segments) in self.training_focus_filter():
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
            results = result_func(prefix, suffix)
            for result in results:
                temporaries.update(self._morph_usage.estimate_contexts(
                    (prefix.morph, suffix.morph),
                    self.detag_word(result)))
                transforms.append(Transformation(rule, result))
            # targets will be a subset of the intersection of the
            # occurences of both submorphs
            targets = set(self.morph_backlinks[prefix.morph])
            targets.intersection_update(self.morph_backlinks[suffix.morph])
            if len(targets) > 0:
                yield(transforms, targets, temporaries)

    def _op_join_generator(self):
        """Generates joins of consecutive morphs into a supermorph.
        Can make different join decisions in different contexts.
        Use with _transformation_epoch
        """

        def join_helper(prefix, suffix):
            joined = prefix.morph + suffix.morph
            return ((CategorizedMorph(joined, None),),)

        return self._generic_bimorph_generator(join_helper)

    def _op_shift_generator(self):
        """Generates operations that shift the split point in a bigram.
        Use with _transformation_epoch
        """

        def shift_helper(prefix, suffix):
            results = []
            for i in range(1, self._max_shift + 1):
                # Move backward
                if len(prefix) - i >= self._min_shift_remainder:
                    new_pre = prefix.morph[:-i]
                    shifted = prefix.morph[-i:]
                    new_suf = shifted + suffix.morph
                    results.append((CategorizedMorph(new_pre, None),
                                    CategorizedMorph(new_suf, None)))
                # Move forward
                if len(suffix) - i >= self._min_shift_remainder:
                    new_suf = suffix.morph[i:]
                    shifted = suffix.morph[:i]
                    new_pre = prefix.morph + shifted
                    results.append((CategorizedMorph(new_pre, None),
                                    CategorizedMorph(new_suf, None)))
            return results

        return self._generic_bimorph_generator(shift_helper)

    def _op_resegment_generator(self):
        """Generates special transformations that resegment and tag
        all words in the corpus using viterbi_segment.
        Use with _transformation_epoch
        """
        if self.training_focus is None:
            source = range(len(self.segmentations))
        else:
            source = self.training_focus
        for i in source:
            word = self.segmentations[i]
            yield ([ViterbiResegmentTransformation(word, self)],
                   set([i]), set())

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
        if self._supervised:
            self._annot_coding.update_morph_count(morph,
                                                  old_count,
                                                  new_count)

    def _update_counts(self, change_counts, multiplier):
        """Updates the model counts according to the pre-calculated
        ChangeCounts object (e.g. calculated in Transformation).

        Arguments:
            change_counts -- A ChangeCounts object
            multiplier -- +1 to apply the change, -1 to revert it.
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

    def viterbi_tag(self, segments):
        """Tag a pre-segmented word using the learned model.

        Arguments:
            segments -- A list of morphs to tag.
                        Raises KeyError if morph is not present in the
                        training data.
                        For segmenting and tagging new words,
                        use viterbi_segment(compound).
        """

        # To make sure that internally impossible states are penalized
        # even more than impossible states caused by zero parameters.
        extrazero = LOGPROB_ZERO ** 2

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = get_categories(wb=True)
        wb = categories.index(WORD_BOUNDARY)
        forbidden = []
        for (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
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

        # Throw away old category information, if any
        segments = self.detag_word(segments)
        for (i, morph) in enumerate(segments):
            for next_cat in range(len(categories)):
                if next_cat == wb:
                    # Impossible to visit boundary in the middle of the
                    # sequence
                    best.append(ViterbiNode(extrazero, None))
                    continue
                for prev_cat in range(len(categories)):
                    if (prev_cat, next_cat) in forbidden:
                        cost.append(extrazero)
                        continue
                    # Cost of selecting prev_cat as previous state
                    # if now at next_cat
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
        result = [CategorizedMorph(segments[-1],
                  categories[backtrace.backpointer])]
        for i in range(len(segments) - 1, 0, -1):
            backtrace = grid[i + 1][backtrace.backpointer]
            result.insert(0, CategorizedMorph(segments[i - 1],
                categories[backtrace.backpointer]))
        return result

    def viterbi_tag_corpus(self):
        """(Re)tags the corpus segmentations using viterbi_tag"""
        num_changed_words = 0
        for (i, word) in enumerate(self.segmentations):
            self.segmentations[i] = WordAnalysis(word.count,
                self.viterbi_tag(word.analysis))
            if word != self.segmentations[i]:
                num_changed_words += 1
        return num_changed_words

    def viterbi_segment(self, segments):
        """Simultaneously segment and tag a word using the learned model.
        Can be used to segment unseen words.

        Arguments:
            segments -- A word (or a list of morphs which will be
                        concatenated into a word) to resegment and tag.
        Returns:
            best_analysis, -- The resegmented, retagged word
            best_cost      -- The cost of the returned solution
        """

        if isinstance(segments, basestring):
            word = segments
        else:
            # Throw away old category information, if any
            segments = self.detag_word(segments)
            # Merge potential segments
            word = ''.join(segments)

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
                morph = word[prev_pos:pos]

                if morph not in self._morph_usage:
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
                    for prev_len in range(1, prev_pos + 1):
                        for prev_cat in categories_nowb:
                            cost = (
                                grid[prev_pos][prev_len - 1][prev_cat].cost +
                                self._corpus_coding.transit_emit_cost(
                                    categories[prev_cat],
                                    categories[next_cat],
                                    morph))
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
        return result, best.cost

    def viterbi_segment_corpus(self):
        """(Re)segments the corpus using viterbi_segment"""
        num_changed_words = 0
        for (i, word) in enumerate(self.segmentations):
            self.segmentations[i] = WordAnalysis(word.count,
                self.viterbi_segment(word.analysis)[0])
            if word != self.segmentations[i]:
                num_changed_words += 1
        self.reestimate_probabilities()
        self._calculate_morph_backlinks()
        return num_changed_words

    def viterbi_analyze_list(self, corpus):
        """Convenience wrapper around viterbi_segment for a
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
            if isinstance(word, basestring):
                word = (word,)
            yield WordAnalysis(count, self.viterbi_segment(word)[0])

    def map_segmentations(self, func):
        """Apply a mapping to the analysis part of segmentations.
        Convenience function."""
        for word in self.segmentations:
            yield WordAnalysis(word.count, func(word.analysis))

    def get_cost(self):
        """Return current model encoding cost."""
        cost = self._corpus_coding.get_cost() + self._lexicon_coding.get_cost()
        if self._supervised:
            cost += self._annot_coding.get_cost()
        return cost

    def _update_annotation_choices(self):
        """Update the selection of alternative analyses in annotations."""
        if not self._supervised:
            return
        # Will need to check for proper expansion when introducing
        # hierarchical morphs

        constructions_add = collections.Counter()
        blacklist = set()
        # changes to the normal corpus counts
        constructions_rm = collections.Counter()
        changes = ChangeCounts()
        for (i, annotation) in enumerate(self.annotations):
            (_, alternatives) = annotation

            if not self._annotations_tagged:
                alternatives = [self.viterbi_tag(alt) for alt in alternatives]

            sorted_alts = self.best_analysis([AnalysisAlternative(alt, 0)
                                              for alt in alternatives])
            old_active = self.segmentations[i].analysis
            new_active = sorted_alts[0].analysis

            if old_active is not None:
                changes.update(old_active, -1, corpus_index=i)
            changes.update(new_active, 1, corpus_index=i)

            # Active segmentation is changed before removal/adding of morphs
            self.segmentations[i] = WordAnalysis(1, new_active)
            # Only morphs in both new_active and old_active will get penalty,
            # which will be cancelled out when adding new_active.
            if old_active is not None:
                for morph in self.detag_word(old_active):
                    constructions_rm[morph] += 1
            new_detagged = self.detag_word(new_active)
            for morph in new_detagged:
                constructions_add[morph] += 1
            for (prefix, suffix) in utils.ngrams(new_detagged, n=2):
                blacklist.add(prefix + suffix)

        for (morph, count) in constructions_rm.items():
            self._modify_morph_count(morph, -count)
        for (morph, count) in constructions_add.items():
            self._modify_morph_count(morph, count)
        self._update_counts(changes, 1)
        self._annot_coding.set_constructions(constructions_add)
        for supermorph in blacklist:
            self._annot_coding.add_to_blacklist(supermorph)
            self._annot_coding.set_morph_count(supermorph,
                self._morph_usage.count(supermorph))
        for morph in constructions_add:
            count = self._morph_usage.count(morph)
            self._annot_coding.set_morph_count(morph, count)

    def add_annotations(self, annotations, annotatedcorpusweight=None,
        multiplier=1, penalty=-999999, blacklist_penalty=-999999):
        self._supervised = True
        self._annotations_tagged = True
        for (word, alternatives) in annotations.items():
            if alternatives[0][0].category == CategorizedMorph.no_category:
                self._annotations_tagged = False
            # The fist entries in self.segmentations are the currently active
            # annotations, in the same order as in self.annotations
            self.segmentations.insert(
                len(self.annotations),
                WordAnalysis(multiplier, alternatives[0]))
            self.annotations.append((word, alternatives))
        self._calculate_morph_backlinks()
        self._annot_coding = CatmapAnnotatedCorpusEncoding(
                                self,
                                weight=annotatedcorpusweight,
                                penalty=penalty,
                                blacklist_penalty=blacklist_penalty)
        self._annot_coding.boundaries = len(self.annotations)

    def get_corpus_coding_weight(self):
        return self._corpus_coding.weight

    def set_corpus_coding_weight(self, weight):
        self._corpus_coding.weight = weight

    def training_focus_filter(self):
        if self.training_focus is None:
            for seg in self.segmentations:
                yield seg
        else:
            ordered = sorted(self.training_focus)
            for i in ordered:
                yield self.segmentations[i]

    def set_focus_sample(self, num_samples):
        if num_samples > len(self.segmentations):
            # No point sampling a larger set than the corpus
            self.training_focus = None
        else:
            self.training_focus = set(utils.weighted_sample(
                self.segmentations, num_samples))

    def best_analysis(self, choices):
        """Choose the best analysis of a set of choices.

        Observe that the call and return signatures are different
        from baseline: this method is more versatile.

        Arguments:
            choices -- a sequence of AnalysisAlternative(analysis, penalty)
                       namedtuples.
                       The analysis must be a sequence of CategorizedMorphs,
                       (segmented and tagged).
                       The penalty is a float that is added to the cost
                       for this choice. Use 0 to disable.
        Returns:
            A sorted (by cost, ascending) list of
            SortedAnalysis(cost, analysis, index, breakdown) namedtuples.
                cost -- the contribution of this analysis to the corpus cost.
                analysis -- as in input.
                breakdown -- A CostBreakdown object, for diagnostics
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
        if all(isinstance(s, basestring) for s in segmentations):
            segmentations = [segmentations]
        if retag is not None:
            assert isinstance(retag, bool)
            tagged = []
            for seg in segmentations:
                tagged.append(AnalysisAlternative(self.viterbi_tag(seg), 0))
        else:
            tagged = [AnalysisAlternative(x, 0) for x in segmentations]
        return self.best_analysis(tagged)

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

    def violated_annotations(self):
        """Yields all segmentations which have an associated annotation,
        but are currently segmented in a way that is not included in the
        annotation alternatives."""
        for (i, anno) in enumerate(self.annotations):
            (_, alternatives) = anno
            alts_de = [self.detag_word(alt) for alt in alternatives]
            seg_de = self.detag_word(self.segmentations[i].analysis)
            if seg_de not in alts_de:
                yield (seg_de, alts_de)

    def toggle_callbacks(self, callbacks=None):
        """Callbacks are not saved in the pickled model, because pickle is
        unable to restore instance methods. If you need callbacks in a loaded
        model, you have to readd them after loading.
        """
        out = (self.operation_callbacks, self.epoch_callbacks)
        if callbacks is None:
            self.operation_callbacks = []
            self.epoch_callbacks = []
        else:
            (self.operation_callbacks, self.epoch_callbacks) = callbacks
        return out

    def get_learned_params(self):
        """Returns a dict of learned and estimated parameters."""
        params = {'corpusweight': self.get_corpus_coding_weight()}
        if self._supervised:
            params['annotationweight'] = self._annot_coding.weight
        return params

    def set_learned_params(self, params):
        if 'corpusweight' in params:
            _logger.info('Setting corpus coding weight to {}'.format(
                params['corpusweight']))
            self.set_corpus_coding_weight(float(params['corpusweight']))
        if self._supervised and 'annotationweight' in params:
            _logger.info('Setting annotation weight to {}'.format(
                params['annotationweight']))
            self._annot_coding.weight = float(params['annotationweight'])

    def _cost_field_fmt(self, cost):
        current = len(str(int(cost))) + self._cost_field_precision + 1
        if current > self._cost_field_width:
            self._cost_field_width = current
        return '{}.{}'.format(self._cost_field_width,
                              self._cost_field_precision)

    @staticmethod
    def get_categories(wb=False):
        """The category tags supported by this model.
        Argumments:
            wb -- If True, the word boundary will be included. Default: False.
        """
        return get_categories(wb)

    @staticmethod
    def _detag_morph(morph):
        if isinstance(morph, CategorizedMorph):
            return morph.morph
        return morph

    @staticmethod
    def detag_word(segments):
        return [CatmapModel._detag_morph(x) for x in segments]

    @staticmethod
    def detag_list(segmentations):
        """Removes category tags from a segmented corpus."""
        for rcount, segments in segmentations:
            yield ((rcount, [CatmapModel._detag_morph(x) for x in segments]))

    @property
    def word_tokens(self):
        return self._corpus_coding.boundaries

    @property
    def morph_tokens(self):
        return sum(self._morph_usage.category_token_count)


class ChangeCounts(object):
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
            analysis -- A tuple of CategorizedMorphs.
            count -- The occurence count of the analyzed word.
                     A negative count removes the effects of the analysis.
            corpus_index -- If not None, the mapping between the morphs
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
        self.annot_counts = ChangeCounts()

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.rule, self.result)

    def apply(self, word, model, corpus_index=None, is_annot=False):
        """Tries to apply this transformation to an analysis.
        If the transformation doesn't match, the input is returned unchanged.
        If the transformation matches, changes are made greedily from the
        beginning of the analysis, until the whole sequence has been
        processed. After this the segmentation is retagged using viterbi
        tagging with the given model.

        Arguments:
            word -- A WordAnalysis object.
            model -- The current model to use for tagging.
            corpus_index -- Index of the word in the corpus, or None if
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
            out = model.viterbi_tag(out)

            self.change_counts.update(word.analysis, -word.count,
                                      corpus_index)
            self.change_counts.update(out, word.count, corpus_index)
            if is_annot:
                self.annot_counts.update(word.analysis, -word.count,
                                        corpus_index)
                self.annot_counts.update(out, word.count, corpus_index)

        return WordAnalysis(word.count, out)

    def reset_counts(self):
        self.change_counts = ChangeCounts()
        self.annot_counts = ChangeCounts()


class ViterbiResegmentTransformation(object):
    """Special transformation that resegments and tags
    words in the corpus using viterbi_segment.
    """

    def __init__(self, word, model):
        self.rule = TransformationRule(tuple(word.analysis))
        self.result, _ = model.viterbi_segment(word.analysis)
        self.change_counts = ChangeCounts()
        self.annot_counts = ChangeCounts()

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.rule, self.result)

    def apply(self, word, model, corpus_index=None):
        if self.rule.num_matches(word.analysis) == 0:
            return word
        self.change_counts.update(word.analysis, -word.count,
                                    corpus_index)
        self.change_counts.update(out, word.count, corpus_index)
        if is_annot:
            self.annot_counts.update(word.analysis, -word.count,
                                    corpus_index)
            self.annot_counts.update(out, word.count, corpus_index)
        return WordAnalysis(word.count, self.result)

    def reset_counts(self):
        self.change_counts = ChangeCounts()
        self.annot_counts = ChangeCounts()


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
            if self._rule[j].category != CategorizedMorph.no_category:
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


ByPartition = collections.namedtuple('ByPartition',
                                     ['total', 'corpus', 'annotations'])
WithTotal = collections.namedtuple('WithTotal', ['total', 'by_cat'])

class TokenCount(object):
    def __init__(self, morph_usage):
        self._morph_usage = morph_usage
        self._emission_counts = utils.Sparse(
            default=lambda: ByPartition(
                total=0,
                corpus=WithTotal(total=0,
                                 by_cat=utils._nt_zeros(ByCategory)),
                annotations=None))
        self._transition_counts = utils.Sparse(
            default=lambda: ByPartition(
                total=0,
                corpus=WithTotal(total=0,
                                 by_cat=collections.Counter()),
                annotations=None))


class CatmapLexiconEncoding(baseline.LexiconEncoding):
    """Extends LexiconEncoding to include the coding costs of the
    encoding cost of morph usage (context) features.
    """

    def __init__(self, morph_usage):
        super(CatmapLexiconEncoding, self).__init__()
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
        super(CatmapLexiconEncoding, self).add(morph)
        self.logfeaturesum += self._morph_usage.feature_cost(morph)

    def remove(self, morph):
        super(CatmapLexiconEncoding, self).remove(morph)
        self.logfeaturesum -= self._morph_usage.feature_cost(morph)

    def get_cost(self):
        assert self.boundaries >= 0
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return  ((n * math.log(n)
                  - self.boundaries * math.log(self.boundaries)
                  - self.logtokensum
                  + self.permutations_cost()
                  + self.logfeaturesum
                 ) * self.weight
                 + self.frequency_distribution_cost())

    def get_codelength(self, morph):
        cost = super(CatmapLexiconEncoding, self).get_codelength(morph)
        cost += self._morph_usage.feature_cost(morph)
        return cost


class CatmapEncoding(baseline.CorpusEncoding):
    """Class for calculating the encoding costs of the grammar and the
    corpus. Also stores the HMM parameters.

    tokens: the number of emissions observed.
    boundaries: the number of word tokens observed.
    """

    def __init__(self, morph_usage, lexicon_encoding, weight=1.0):
        self._morph_usage = morph_usage
        super(CatmapEncoding, self).__init__(lexicon_encoding, weight)

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

        self.logcondprobsum = 0.0

    # Transition count methods

    def get_transition_count(self, prev_cat, next_cat):
        return self._transition_counts[(prev_cat, next_cat)]

    def log_transitionprob(self, prev_cat, next_cat):
        pair = (prev_cat, next_cat)
        if pair not in self._log_transitionprob_cache:
            if self._cat_tagcount[prev_cat] == 0:
                self._log_transitionprob_cache[pair] = LOGPROB_ZERO
            else:
                self._log_transitionprob_cache[pair] = (
                    zlog(self._transition_counts[(prev_cat, next_cat)]) -
                    zlog(self._cat_tagcount[prev_cat]))
        return self._log_transitionprob_cache[pair]

    def update_transition_count(self, prev_cat, next_cat, diff_count):
        """Updates the number of observed transitions between
        categories.

        Arguments:
            prev_cat -- The name (not index) of the category
                        transitioned from.
            next_cat -- The name (not index) of the category
                        transitioned to.
            diff_count -- The change in the number of transitions.
        """

        msg = 'update_transition_count needs category names, not indices'
        assert not isinstance(prev_cat, int), msg
        assert not isinstance(next_cat, int), msg
        pair = (prev_cat, next_cat)

        self._transition_counts[pair] += diff_count
        self._cat_tagcount[prev_cat] += diff_count

        if self._transition_counts[pair] > 0:
            assert pair not in MorphUsageProperties.zero_transitions

        msg = 'subzero transition count for {}'.format(pair)
        assert self._transition_counts[pair] >= 0, msg
        assert self._cat_tagcount[prev_cat] >= 0

        # invalidate cache
        self._log_transitionprob_cache.clear()

    def clear_transition_counts(self):
        """Resets transition counts, costs and cache.
        Use before fully reprocessing a tagged segmented corpus."""
        self._transition_counts.clear()
        self._cat_tagcount.clear()
        self._log_transitionprob_cache.clear()

    # Emission count methods

    def get_emission_counts(self, morph):
        return self._emission_counts[morph]

    def log_emissionprob(self, category, morph):
        """-Log of posterior emission probability P(morph|category)"""
        pair = (category, morph)
        if pair not in self._log_emissionprob_cache:
            cat_index = get_categories().index(category)
            # Not equal to what you get by:
            # zlog(self._emission_counts[morph][cat_index]) +
            if self._cat_tagcount[category] == 0:
                self._log_emissionprob_cache[pair] = LOGPROB_ZERO
            else:
                self._log_emissionprob_cache[pair] = (
                    zlog(self._morph_usage.count(morph)) +
                    zlog(self._morph_usage.condprobs(morph)[cat_index]) -
                    zlog(self._morph_usage.category_token_count[cat_index]))
        msg = 'emission {} -> {} has probability > 1'.format(category, morph)
        assert self._log_emissionprob_cache[pair] >= 0, msg
        return self._log_emissionprob_cache[pair]

    def update_emission_count(self, category, morph, diff_count):
        """Updates the number of observed emissions of a single morph from a
        single category, and the logtokensum (which is category independent).
        Updates logcondprobsum.

        Arguments:
            category -- name of category from which emission occurs.
            morph -- string representation of the morph.
            diff_count -- the change in the number of occurences.
        """
        cat_index = get_categories().index(category)
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

        # invalidate cache
        self._log_emissionprob_cache.clear()

    def _set_emission_counts(self, morph, new_counts):
        """Set the number of emissions of a morph from all categories
        simultaneously.
        Does not update logcondprobsum.

        Arguments:
            morph -- string representation of the morph.
            new_counts -- ByCategory object with new counts.
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

        # invalidate cache
        self._log_emissionprob_cache.clear()

    def clear_emission_counts(self):
        """Resets emission counts and costs.
        Use before fully reprocessing a tagged segmented corpus."""
        self.tokens = 0
        self.logtokensum = 0.0
        self.logcondprobsum = 0.0
        self._emission_counts.clear()
        self._log_emissionprob_cache.clear()

    def clear_emission_cache(self):
        """Clears the cache for emission probability values.
        Use if an incremental change invalidates cached values."""
        self._log_emissionprob_cache.clear()

    # General methods

    def transit_emit_cost(self, prev_cat, next_cat, morph):
        """Cost of transitioning from prev_cat to next_cat and emitting
        the morph."""
        if (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
            return LOGPROB_ZERO
        return (self.log_transitionprob(prev_cat, next_cat) +
                self.log_emissionprob(next_cat, morph))

    def update_count(self, construction, old_count, new_count):
        raise Exception('Inherited method not appropriate for CatmapEncoding')

    def logtransitionsum(self):
        categories = get_categories(wb=True)
        t_cost = 0.0
        # FIXME: this can be optimized when getting rid of the assertions
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
        return  ((self.tokens * math.log(self.tokens)
                  - self.logtokensum
                  - self.logcondprobsum
                  - self.logtransitionsum()
                  + n * math.log(n)
                 ) * self.weight
                 + self.frequency_distribution_cost()
                )


class CatmapAnnotatedCorpusEncoding(baseline.AnnotatedCorpusEncoding):
    def __init__(self, model, weight=None, multiplier=1,
                 penalty=-999999, blacklist_penalty=-999999):
        super(CatmapAnnotatedCorpusEncoding, self).__init__(
            model._corpus_coding,
            weight=weight,
            penalty=penalty)
        self.model = model
        self.multiplier = multiplier
        self.blacklist = set()
        self.blacklist_penalty = blacklist_penalty

    def add_to_blacklist(self, morph):
        """Blacklist to prevent supermorphs of annotation parts from
        being added to the lexicon, unless they are separately included
        in the annotations.
        """
        if self.blacklist_penalty != 0 and morph not in self.constructions:
            self.blacklist.add(morph)

    def set_constructions(self, constructions):
        super(CatmapAnnotatedCorpusEncoding, self).set_constructions(
            constructions)
        self.blacklist.clear()

    def set_morph_count(self, construction, count):
        super(CatmapAnnotatedCorpusEncoding, self).set_count(construction,
                                                             count)
        if count > 0 and construction in self.blacklist:
            self.logtokensum += self.blacklist_penalty

    def update_morph_count(self, construction, old_count, new_count):
        super(CatmapAnnotatedCorpusEncoding, self).update_count(construction,
                                                                old_count,
                                                                new_count)
        assert new_count >= 0
        if construction in self.blacklist:
            if old_count > 0:
                self.logtokensum -= self.blacklist_penalty
            if new_count > 0:
                self.logtokensum += self.blacklist_penalty

    def update_counts(self, change_counts, multiplier):
        """Updates the emission and transition counts
        according to the pre-calculated
        ChangeCounts object (e.g. calculated in Transformation).

        Arguments:
            change_counts -- A ChangeCounts object
            multiplier -- +1 to apply the change, -1 to revert it.
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

    def set_multiplier(self, multiplier):
        for i in range(len(self.model.annotations)):
            self.model.segmentations[i] = WordAnalysis(
                self.multiplier, self.model.segmentations[i].analysis)
        self.model.reestimate_probabilities()

    def update_weight(self):
        """Update the weight of the Encoding by taking the ratio of the
        corpus boundaries and annotated boundaries.
        Does not scale by corpus weight, unlike Morfessor Baseline.
        """
        if not self.do_update_weight:
            return
        old = self.weight
        annotation_counts = 0
        for i in range(len(self.model.annotations)):
            annotation_counts += self.model.segmentations[i].count
        self.weight = (float(self.corpus_coding.boundaries - annotation_counts)
            / self.boundaries)
        #self.set_multiplier(int(math.ceil(math.log(self.weight))))
        self.weight /= float(self.multiplier)
        if self.weight != old:
            _logger.info(
                'Corpus weight of annotated data set to {} * {}'.format(
                    self.multiplier, self.weight))


class CorpusWeightUpdater(object):
    def __init__(self, annotations, heuristic, io, checkpointfile,
                 max_epochs=2, threshold=0.01):
        self.annotations = annotations
        self.heuristic = heuristic
        self.io = io
        self.checkpointfile = checkpointfile
        self.max_epochs = max_epochs
        self.threshold = threshold

    def calculate_update(self, model):
        """Tune model corpus weight based on the precision and
        recall of the development data, trying to keep them equal"""
        tmp = self.annotations.items()
        wlist, annotations = zip(*tmp)

        if self.heuristic is None:
            heuristic_func = lambda x: x
        else:
            heuristic_func = lambda x: self.heuristic.remove_nonmorfemes(x,
                                                                     model)

        segments = [heuristic_func(model.viterbi_segment(w)[0])
                    for w in wlist]
        pre, rec, f = baseline.AnnotationsModelUpdate._bpr_evaluation(
                         [[x] for x in segments], annotations)
        if abs(pre - rec) < self.threshold:
            direction = 0
        elif rec > pre:
            direction = 1
        else:
            direction = -1
        return (f, direction)

    def update_model(self, model, epochs, direction):
        if direction != 0:
            weight = model.get_corpus_coding_weight()
            if direction > 0:
                weight *= 1 + 2.0 / epochs
            else:
                weight *= 1.0 / (1 + 2.0 / epochs)
            model.set_corpus_coding_weight(weight)
        return weight

    def weight_learning(self, model, max_epochs=None):
        if max_epochs is None:
            max_epochs = self.max_epochs
        real_iteration_number = model._iteration_number
        callbacks = model.toggle_callbacks(None)
        model._iteration_update(no_increment=True)
        self.io.write_binary_model_file(self.checkpointfile, model)
        first_weight = model.get_corpus_coding_weight()
        prev_weight = first_weight
        model.weightlearn_probe()
        (f_prev, direction) = self.calculate_update(model)

        _logger.info(
            'Initial corpus weight: ' +
            '{}, f-measure: {}, direction: {}'.format(
                prev_weight, f_prev, direction))

        for i in range(max_epochs):
            # Revert the changes by reloading the checkpoint model.
            # Good steps are also reverted, to prevent accumulated gains
            # and make the comparison fair.
            model = self.io.read_binary_model_file(self.checkpointfile)
            model.set_corpus_coding_weight(prev_weight)
            if direction == 0:
                break
            next_weight = self.update_model(model,
                                            i + real_iteration_number,
                                            direction)
            model.weightlearn_probe()
            prev_direction = direction
            (f, direction) = self.calculate_update(model)
            _logger.info(
                'Weight learning iteration {}: '.format(i) +
                'corpus weight {}, f-measure: {}, direction: {}'.format(
                    next_weight, f, direction))
            if f > f_prev:
                # Accept the step
                _logger.info('Accepted the step to {}'.format(next_weight))
                prev_weight = next_weight
                f_prev = f
            else:
                _logger.info('Rejected the step, ' +
                    'reverting to weight {}'.format(prev_weight))
                # Discard the step and try again with a smaller step
                direction = prev_direction
        # Start normal training from the checkpoint using the optimized weight
        model = self.io.read_binary_model_file(self.checkpointfile)
        model._iteration_number = real_iteration_number
        model.training_focus = None
        model.toggle_callbacks(callbacks)
        model.set_corpus_coding_weight(prev_weight)
        _logger.info('Final learned corpus weight {}'.format(prev_weight))

        return (model, model.get_corpus_coding_weight() != first_weight)


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
        return list(segments) + [wb]
    else:
        return [wb] + segments + [wb]


class CostBreakdown(object):
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
