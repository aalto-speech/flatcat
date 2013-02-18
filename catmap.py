#!/usr/bin/env python
"""
Morfessor 2.0 Categories-MAP variant.
"""

__all__ = ['CatmapIO', 'CatmapModel']

import collections
import logging
import numpy as np

import morfessor

_logger = logging.getLogger(__name__)
_logger.level = logging.DEBUG   # FIXME development convenience

LOGPROB_ZERO = 1000


class CatmapIO(morfessor.MorfessorIO):
    """Extends data file formats to include category tags."""
    # FIXME unimplemented

    def __init__(self, encoding=None, construction_separator=' + ',
                 comment_start='#', compound_separator='\s+',
                 atom_separator=None, category_separator='/'):
        morfessor.MorfessorIO.__init__(
            self, encoding=encoding,
            construction_separator=construction_separator,
            comment_start=comment_start, compound_separator=compound_separator,
            atom_separator=atom_separator)
        self.category_separator = category_separator


CatProbs = collections.namedtuple('CatProbs', ['PRE', 'STM', 'SUF', 'ZZZ'])
ProbN = collections.namedtuple('ProbN', ['PROB', 'N'])


class WordBoundary:
    def __repr__(self):
        return '#'


class CatmapModel:
    """Morfessor Categories-MAP model class."""

    word_boundary = WordBoundary()

    # These transitions are impossible
    zero_transitions = ((word_boundary, word_boundary),
                        ('PRE', word_boundary),
                        ('PRE', 'SUF'),
                        (word_boundary, 'SUF'))

    def __init__(self, ppl_treshold=100, ppl_slope=None, length_treshold=3,
                 length_slope=2, use_word_tokens=True,
                 min_perplexity_length=4, transition_cutoff=0.00000000001):
        """Initialize a new model instance.

        Arguments:
            ppl_treshold -- Treshold value for sigmoid used to calculate
                            probabilities from left and right perplexities.
            ppl_slope -- Slope value for sigmoid used to calculate
                         probabilities from left and right perplexities.
            length_treshold -- Treshold value for sigmoid used to calculate
                               probabilities from length of morph.
            length_slope -- Slope value for sigmoid used to calculate
                            probabilities from length of morph.
            use_word_tokens -- If true, perplexity is based on word tokens.
                               If false, perplexity is based on word types.
            min_perplexity_length -- Morphs shorter than this length are
                                     ignored when calculating perplexity.
            transition_cutoff -- FIXME
        """

        self._ppl_treshold = float(ppl_treshold)
        self._length_treshold = float(length_treshold)
        self._length_slope = float(length_slope)
        self._use_word_tokens = bool(use_word_tokens)
        self._min_perplexity_length = int(min_perplexity_length)
        self._transition_cutoff = float(transition_cutoff)
        if ppl_slope is not None:
            self._ppl_slope = float(ppl_slope)
        else:
            self._ppl_slope = 10.0 / self._ppl_treshold

        # Counts of different contexts in which a morph occurs
        self._contexts = collections.defaultdict(MorphContext)

        # Conditional probabilities P(Category|Morph).
        # A dict of CatProbs objects. Actual probabilities.
        self._condprobs = dict()

        # Priors for categories P(Category).
        # Single CatProbs object. Log-probabilities.
        self._log_catpriors = None

        # Posterior emission probabilities P(Morph|Category).
        # A dict of CatProbs objects. Log-probabilities.
        self._log_emissionprobs = dict()

        # Probabilities of transition between categories.
        #P(Category -> Category). A dict of ProbN objects. Log-probabilities.
        self._log_transitionprobs = dict()

    def train(self, segmentations):
        """Perform Cat-MAP training on the model.

        Arguments:
            segmentations -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
        """
        self.load_baseline(segmentations)
        until_convergence(self._estimate_transition_probs, segmentations)

    def load_baseline(self, segmentations):
        """Initialize the model using the segmentation produced by a morfessor
        baseline model.

        Arguments:
            segmentations -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
        """

        self._estimate_probabilities(segmentations)

        self._unigram_transition_probs(self._category_totals,
                                       self._num_word_tokens)

    def _estimate_probabilities(self, segmentations):
        """Estimates P(Category|Morph), P(Category) and P(Morph|Category).
        """

        self._total_morph_tokens = 0
        self._num_word_types = 0
        self._num_word_tokens = 0

        for rcount, segments in segmentations:
            # Category tags are not needed for these calculations
            segments = [CatmapModel._detag_morph(x) for x in segments]

            self._num_word_types += 1
            self._num_word_tokens += rcount
            if self._use_word_tokens:
                pcount = rcount
            else:
                # pcount used for perplexity, rcount is real count
                pcount = 1
            self._total_morph_tokens += len(segments)
            # Collect information about the contexts in which the morphs occur
            for (i, morph) in enumerate(segments):
                # Previous morph
                if i == 0:
                    # Word boundaries are counted as separate contexts
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i - 1]
                    # Contexts shorter than treshold don't affect perplexity
                    if len(neighbour) < self._min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self._contexts[morph].left[neighbour] += pcount

                # Next morph
                if i == len(segments) - 1:
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i + 1]
                    if len(neighbour) < self._min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self._contexts[morph].right[neighbour] += pcount

                self._contexts[morph].rcount += rcount

        # Calculate conditional probabilities from the encountered contexts
        marginalizer = Marginalizer()
        for morph in sorted(self._contexts, cmp=lambda x, y: len(x) < len(y)):
            self._condprobs[morph] = self._context_to_probability(morph,
                self._contexts[morph])
            # Marginalize (scale by frequency and accumulate elementwise)
            marginalizer.add(self._contexts[morph].rcount,
                           self._condprobs[morph])
        # Category priors from marginalization
        self._catpriors = _log_catprobs(marginalizer.normalized())

        # Calculate posterior emission probabilities
        self._category_totals = marginalizer.category_token_count
        for morph in self._contexts:
            tmp = []
            for (i, total) in enumerate(self._category_totals):
                tmp.append(self._condprobs[morph][i] *
                           self._contexts[morph].rcount /
                           self._category_totals[i])
            self._log_emissionprobs[morph] = _log_catprobs(CatProbs(*tmp))

    def _context_to_probability(self, morph, context):
        """Calculate conditional probabilities P(Category|Morph) from the
        contexts in which the morphs occur.

        Arguments:
            morph -- A string representation of the morph type.
            context -- MorphContext object for morph type.
        """

        prelike = sigmoid(context.right_perplexity, self._ppl_treshold,
                          self._ppl_slope)
        suflike = sigmoid(context.left_perplexity, self._ppl_treshold,
                          self._ppl_slope)
        stmlike = sigmoid(len(morph), self._length_treshold,
                          self._length_slope)

        p_nonmorpheme = (1. - prelike) * (1. - suflike) * (1. - stmlike)

        if p_nonmorpheme == 1:
            p_pre = 0.0
            p_suf = 0.0
            p_stm = 0.0
        else:
            if p_nonmorpheme < 0.001:
                p_nonmorpheme = 0.001

            normcoeff = ((1.0 - p_nonmorpheme) /
                         ((prelike ** 2) + (suflike ** 2) + (stmlike ** 2)))
            p_pre = (prelike ** 2) * normcoeff
            p_suf = (suflike ** 2) * normcoeff
            p_stm = 1.0 - p_pre - p_suf - p_nonmorpheme

        return CatProbs(p_pre, p_stm, p_suf, p_nonmorpheme)

    def _unigram_transition_probs(self, category_token_count,
                                  num_word_tokens):
        """Initial transition probabilities based on unigram distribution.

        Each tag is presumed to be succeeded by the expectation over all data
        of the number of prefixes, suffixes, stems, non-morphemes and word
        boundaries.

        Arguments:
            category_token_count -- A CatProbs with unnormalized
                                    morph token counts.
            num_word_tokens -- Total number of word tokens, for word boundary
                               probability.
        """

        transitions = dict()
        nclass = {CatmapModel.word_boundary: num_word_tokens}
        for (i, category) in enumerate(CatmapModel.get_categories()):
            nclass[category] = float(category_token_count[i])

        num_tokens_tagged = collections.defaultdict(int)
        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in CatmapModel.zero_transitions:
                    continue
                # count all possible valid transitions
                num_tokens_tagged[cat1] += nclass[cat2]

        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in CatmapModel.zero_transitions:
                    continue
                transitions[(cat1, cat2)] = ProbN(_zlog(nclass[cat2] /
                                                  num_tokens_tagged[cat1]),
                                                  nclass[cat2])

        for pair in CatmapModel.zero_transitions:
            transitions[pair] = ProbN(LOGPROB_ZERO, 0)
        self._log_transitionprobs = transitions

    def _estimate_transition_probs(self, segmentations):
        """Estimate transition probabilities from a category-tagged segmented
        corpus.

        Arguments:
            segmentations -- Category-tagged segmented words.
                List of format:
                (count, (CategorizedMorph1, CategorizedMorph2, ...)), ...
        """
        total_transitions_from = collections.defaultdict(int)
        num_transitions = collections.defaultdict(int)
        for rcount, segments in segmentations:
            # Only the categories matter
            categories = [x.category for x in segments]
            # Include word boundaries
            categories.insert(0, CatmapModel.word_boundary)
            categories.append(CatmapModel.word_boundary)
            for (prev_cat, next_cat) in ngrams(categories, 2):
                num_transitions[(prev_cat, next_cat)] += float(rcount)
                total_transitions_from[prev_cat] += float(rcount)

        transitions = dict()
        for prev_cat in CatmapModel.get_categories(wb=True):
            for next_cat in CatmapModel.get_categories(wb=True):
                pair = (prev_cat, next_cat)
                if pair not in num_transitions:
                    transitions[pair] = ProbN(LOGPROB_ZERO, 0)
                    continue
                if pair in CatmapModel.zero_transitions:
                    if num_transitions[pair] > 0:
                        _logger.warning('Impossible transition ' +
                                        '%s -> %s had nonzero count' % pair)
                    transitions[pair] = ProbN(LOGPROB_ZERO, 0)
                else:
                    transitions[pair] = ProbN(_zlog(num_transitions[pair] /
                                            total_transitions_from[prev_cat]),
                                            num_transitions[pair])
        self._log_transitionprobs = transitions

    def viterbi_tag(self, segments):
        """Tag a pre-segmented word using the learned model.

        Arguments:
            segments -- A list of morphs to tag.
                        Raises KeyError if morph is not present in the
                        training data.
                        For segmenting and tagging new words,
                        use viterbi_segment(compound).
        """

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = CatmapModel.get_categories(wb=True)
        # Index of word boundary.
        wb = categories.index(CatmapModel.word_boundary)

        # The lowest accumulated cost ending in each possible state.
        # Initialized to pseudo-zero for all states
        delta = LOGPROB_ZERO * np.ones(len(categories))
        # Back pointers that indicate the best path
        psi = -1 * np.ones((len(segments) + 1,
                            len(categories)), dtype='int')
        # First row of back pointers point to word boundary
        #psi[0] = wb * np.ones(len(categories))

        # Probability one that first state is a word boundary
        delta[wb] = 0

        # Cumulative costs for each category at current time step
        cost = LOGPROB_ZERO * np.ones(len(categories))
        # Temporaries
        best_cat = -1 * np.ones(len(categories), dtype='int')
        best_cost = LOGPROB_ZERO * np.ones(len(categories))

        for (i, morph) in enumerate(segments):
            for next_cat in range(len(categories) - 1):
                for prev_cat in range(len(categories)):
                    name_pair = (categories[prev_cat], categories[next_cat])
                    # Cost of selecting prev_cat as previous state
                    # if now at next_cat
                    cost[prev_cat] = (delta[prev_cat] +
                        self._log_transitionprobs[name_pair].PROB +
                        self._log_emissionprobs[morph][next_cat])
                best_cat[next_cat] = np.argmin(cost)
                best_cost[next_cat] = cost[best_cat[next_cat]]
            # Update delta and psi to prepare for next iteration
            delta = best_cost
            psi[i] = best_cat
            best_cost = LOGPROB_ZERO * np.ones(len(categories))

        # Last transition must be to word boundary
        for prev_cat in range(len(categories)):
            name_pair = (categories[prev_cat], CatmapModel.word_boundary)
            cost[prev_cat] = (delta[prev_cat] +
                              self._log_transitionprobs[name_pair].PROB)
            backtrace = np.argmin(cost)

        # Backtrace for the best category sequence
        result = [CategorizedMorph(segments[-1], categories[backtrace])]
        for i in range(len(segments) - 1, 0, -1):
            backtrace = psi[i, backtrace]
            result.insert(0, CategorizedMorph(segments[i - 1],
                                                categories[backtrace]))
        return result

    def viterbi_tag_segmentations(self, segmentations):
        """Convenience wrapper around viterbi_tag for a list of segmentations
        with attached counts."""
        tagged = []
        for (count, segmentation) in segmentations:
            tagged.append((count, self.viterbi_tag(segmentation)))
        return tagged

    def until_convergence(self, func, segmentations, max_differences=0,
                          max_iterations=15):
        """Iterates the specified training function until the segmentations
        produced by the model for the given input no longer change more than
        the specified treshold, or until maximum number of iterations is
        reached.

        Arguments:
            func -- A method of CatmapModel that takes one argument:
                    segmentations, and which causes some aspect of the model
                    to be trained.
            segmentations -- list of (count, segmentation) pairs. Can be
                             either tagged or untagged.
            max_differences -- Maximum number of changed category tags in
                               the final iteration. Default 0.
            max_iterations -- Maximum number of iterations. Default 15.
        """

        detagged = CatmapModel._detag_segmentations(segmentations)
        previous_segmentation = self.viterbi_tag_segmentations(detagged)
        for iteration in range(max_iterations):
            _logger.info('Iteration number %d/%d.' % (iteration,
                                                      max_iterations))
            # perform the optimization
            func(previous_segmentation)

            current_segmentation = self.viterbi_tag_segmentations(detagged)
            differences = 0
            for (r, o) in zip(previous_segmentation, current_segmentation):
                if r != o:
                    differences += 1
            if differences <= max_differences:
                _logger.info('Converged, with ' +
                    '%d differences in final iteration.' % (differences,))
                break
            _logger.info('%d differences.' % (differences,))
            previous_segmentation = current_segmentation

    @staticmethod
    def get_categories(wb=False):
        """The category tags supported by this model.
        Argumments:
            wb -- If True, the word boundary will be included. Default: False.
        """
        categories = list(CatProbs._fields)
        if wb:
            categories.append(CatmapModel.word_boundary)
        return tuple(categories)

    @staticmethod
    def _detag_morph(morph):
        if isinstance(morph, CategorizedMorph):
            return morph.morph
        return morph

    @staticmethod
    def _detag_segmentations(segmentations):
        detagged = []
        for rcount, segments in segmentations:
            detagged.append((rcount,
                             [CatmapModel._detag_morph(x) for x in segments]))
        return detagged


class CategorizedMorph:
    """Represents a morph with attached category information."""
    no_category = object()

    def __init__(self, morph, category=None):
        self.morph = morph
        if category is not None:
            self.category = category
        else:
            self.category = CategorizedMorph.no_category

    def __repr__(self):
        if self.category == CategorizedMorph.no_category:
            return u'%s' % (self.morph,)
        return u'%s/%s' % (self.morph, self.category)

    def __eq__(self, other):
        return (self.morph == other.morph and
                self.category == other.category)


class MorphContext:
    """Represents the different contexts in which a morph has been
    encountered.
    """

    def __init__(self):
        self.rcount = 0
        self.left = collections.defaultdict(int)
        self.right = collections.defaultdict(int)

    @property
    def left_perplexity(self):
        return MorphContext._perplexity(self.left)

    @property
    def right_perplexity(self):
        return MorphContext._perplexity(self.right)

    @staticmethod
    def _perplexity(contexts):
        entropy = 0
        total_tokens = float(sum(contexts.values()))
        for c in contexts:
            p = float(contexts[c]) / total_tokens
            entropy -= p * np.log(p)
        return np.exp(entropy)


class Marginalizer:
    """An accumulator for marginalizing the class probabilities
    P(Category) from all the individual conditional probabilities
    P(Category|Morph) and observed morph probabilities P(Morph).

    First the unnormalized distribution is obtained by summing over
    #(Morph) * P(Category|Morph) over each morph, separately for each
    category. P(Category) is then obtained by normalizing the
    distribution.
    """

    def __init__(self):
        self._counts = [0.0] * len(CatProbs._fields)

    def add(self, rcount, condprobs):
        """Add the products #(Morph) * P(Category|Morph)
        for one observed morph."""
        for i, x in enumerate(condprobs):
            self._counts[i] += float(rcount) * float(x)

    def normalized(self):
        """Returns the marginal probabilities for all categories."""
        total = self.total_token_count
        return CatProbs(*[x / total for x in self._counts])

    @property
    def total_token_count(self):
        """Total number of tokens seen."""
        return sum(self._counts)

    @property
    def category_token_count(self):
        """Tokens seen per category."""
        return CatProbs(*self._counts)


def sigmoid(value, treshold, slope):
    return 1.0 / (1.0 + np.exp(-slope * (value - treshold)))


def ngrams(sequence, n=2):
    window = []
    for item in sequence:
        window.append(item)
        if len(window) > n:
            # trim back to size
            window = window[-n:]
        if len(window) == n:
            yield(tuple(window))


def _zlog(x):
    """Logarithm which uses constant value for log(0) instead of -inf"""
    # FIXME not sure if this is needed anymore when using numpy

    if x == 0:
        return LOGPROB_ZERO
    return -np.log(x)


def _log_catprobs(probs):
    """Convenience function to convert a CatProbs object containing actual
    probabilities into one with log probabilities"""

    return CatProbs(*[_zlog(x) for x in probs])
