#!/usr/bin/env python
"""
Morfessor 2.0 Categories-MAP variant.
"""

__all__ = ['CatmapIO', 'CatmapModel']

import collections
import logging
import math

import morfessor

_logger = logging.getLogger(__name__)
_logger.level = logging.DEBUG   # FIXME development convenience

LOGPROB_ZERO = 1000000


class WordBoundary:
    def __repr__(self):
        return '#'

WORD_BOUNDARY = WordBoundary()

##################################
### Categorization-dependent code:
### to change the categories, only code in this section needs to be changed.


class ByCategory:
    """A data structure with one value for each category.

    Commonly used for counts or probabilities, which have a usage pattern
    in which values for all of the categories are needed at the same time.
    """

    # This defines the set of possible categories
    __slots__ = ('PRE', 'STM', 'SUF', 'ZZZ')

    def __init__(self, *args):
        """Initialize the structure.
        You must either give as arguments values to all categories,
        or none (which sets all values to zero)."""
        if len(args) == 0:
            args = [0.0] * len(self.__slots__)
        tmp = tuple(args)
        msg = 'ByCategory initialized with {} values (expecting {})'.format(
            len(tmp), len(self.__slots__))
        assert len(tmp) == len(self.__slots__), msg
        for (key, val) in zip(self.__slots__, args):
            setattr(self, key, val)

    def __len__(self):
        return len(self.__slots__)

    def __iter__(self):
        for key in self.__slots__:
            yield getattr(self, key)

    def __getitem__(self, index):
        if isinstance(index, int):
            return getattr(self, self.__slots__[index])
        return getattr(self, index)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            return setattr(self, self.__slots__[index], value)
        return setattr(self, index, value)

    def copy(self):
        tmp = tuple(iter(self))
        return ByCategory(*tmp)


class MorphContext:
    """Represents the different contexts in which a morph has been
    encountered.
    """

    def __init__(self):
        self.rcount = 0
        self.left = collections.Counter()
        self.right = collections.Counter()

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
            entropy -= p * math.log(p)
        return math.exp(entropy)


class MorphUsageProperties:
    """This class describes how the prior probabilities are calculated
    from the usage of morphs.
    """

    # These transitions are impossible
    zero_transitions = ((WORD_BOUNDARY, WORD_BOUNDARY),
                        ('PRE', WORD_BOUNDARY),
                        ('PRE', 'SUF'),
                        (WORD_BOUNDARY, 'SUF'))

    def __init__(self, ppl_treshold=100, ppl_slope=None, length_treshold=3,
                 length_slope=2, use_word_tokens=True,
                 min_perplexity_length=4):
        """Initialize the model parameters describing morph usage.

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
        """

        self._ppl_treshold = float(ppl_treshold)
        self._length_treshold = float(length_treshold)
        self._length_slope = float(length_slope)
        self.use_word_tokens = bool(use_word_tokens)
        self._min_perplexity_length = int(min_perplexity_length)
        if ppl_slope is not None:
            self._ppl_slope = float(ppl_slope)
        else:
            self._ppl_slope = 10.0 / self._ppl_treshold

        # Counts of different contexts in which a morph occurs
        self._contexts = collections.defaultdict(MorphContext)

    def add_context(self, morph, pcount, rcount, i, segments):
        """Collect information about the contexts in which the morph occurs"""
        # Previous morph.
        if i == 0:
            # Word boundaries are counted as separate contexts
            neighbour = WORD_BOUNDARY
        else:
            neighbour = segments[i - 1]
            # Contexts shorter than treshold don't affect perplexity
            if len(neighbour) < self._min_perplexity_length:
                neighbour = None
        if neighbour is not None:
            self._contexts[morph].left[neighbour] += pcount

        # Next morph.
        if i == len(segments) - 1:
            neighbour = WORD_BOUNDARY
        else:
            neighbour = segments[i + 1]
            if len(neighbour) < self._min_perplexity_length:
                neighbour = None
        if neighbour is not None:
            self._contexts[morph].right[neighbour] += pcount

        self._contexts[morph].rcount += rcount

    def condprob(self, morph):
        """Calculate conditional probabilities P(Category|Morph) from the
        contexts in which the morphs occur.

        Arguments:
            morph -- A string representation of the morph type.
        """
        context = self._contexts[morph]

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

        return ByCategory(p_pre, p_stm, p_suf, p_nonmorpheme)

    def seen_morphs(self):
        """All morphs that have defined contexts."""
        return self._contexts.keys()

    def rcount(self, morph):
        """The real counts in the corpus of morphs with contexts."""
        return self._contexts[morph].rcount

### End of categorization-dependent code
########################################


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


class CatmapModel:
    """Morfessor Categories-MAP model class."""

    word_boundary = WORD_BOUNDARY

    def __init__(self, morph_usage, transition_cutoff=0.00000000001):
        """Initialize a new model instance.

        Arguments:
            morph_usage -- A MorphUsageProperties object describing how
                           the usage of a morph affects the category.
            transition_cutoff -- FIXME
        """

        self._morph_usage = morph_usage
        self._transition_cutoff = float(transition_cutoff)

        # Cost variables
        self._lexicon_coding = morfessor.LexiconEncoding()
        # Catmap encoding also stores the HMM parameters
        self._catmap_coding = CatmapEncoding(self._lexicon_coding)

        # Conditional probabilities P(Category|Morph).
        # A dict of ByCategory objects indexed by morph. Actual probabilities.
        self._condprobs = dict()

        # Priors for categories P(Category)
        # Single ByCategory object. Log-probabilities.
        self._log_catpriors = None

        self._num_word_tokens = 0

    def train(self, segmentations):
        """Perform Cat-MAP training on the model.

        Arguments:
            segmentations -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
        """
        self.load_baseline(segmentations)
        self.until_convergence(self._calculate_transition_counts,
                               segmentations)

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

        num_letter_tokens = collections.Counter()
        self._num_word_tokens = 0

        for rcount, segments in segmentations:
            self._num_word_tokens += rcount
            # Category tags are not needed for these calculations
            segments = [CatmapModel._detag_morph(x) for x in segments]

            if self._morph_usage.use_word_tokens:
                pcount = rcount
            else:
                # pcount used for perplexity, rcount is real count
                pcount = 1
            num_letter_tokens[WORD_BOUNDARY] += pcount

            for (i, morph) in enumerate(segments):
                # Add previously unseen morph to lexicon cost
                if morph not in self._morph_usage.seen_morphs():
                    self._lexicon_coding.add(morph)

                # Collect information about the contexts in which
                # the morphs occur.
                self._morph_usage.add_context(morph, pcount, rcount,
                                              i, segments)

                for letter in morph:
                    num_letter_tokens[letter] += pcount

        # Calculate conditional probabilities from the encountered contexts
        marginalizer = Marginalizer()
        for morph in self._morph_usage.seen_morphs():
            self._condprobs[morph] = self._morph_usage.condprob(morph)
            # Marginalize (scale by frequency and accumulate elementwise)
            marginalizer.add(self._morph_usage.rcount(morph),
                             self._condprobs[morph])
        # Category priors from marginalization
        self._log_catpriors = _log_catprobs(marginalizer.normalized())

        # Calculate posterior emission probabilities
        self._category_totals = marginalizer.category_token_count
        for morph in self._morph_usage.seen_morphs():
            tmp = []
            for (i, total) in enumerate(self._category_totals):
                tmp.append(self._condprobs[morph][i] *
                           self._morph_usage.rcount(morph) /
                           total)
            lep = _log_catprobs(ByCategory(*tmp))
            self._catmap_coding.set_log_emissionprobs(morph, lep)

        # Calculate letter log probabilities
        self._total_letter_tokens = sum(num_letter_tokens.values())
        log_tlt = math.log(self._total_letter_tokens)
        self._log_letterprobs = dict()
        for letter in num_letter_tokens:
            self._log_letterprobs[letter] = (log_tlt -
                math.log(num_letter_tokens[letter]))

    def _unigram_transition_probs(self, category_token_count,
                                  num_word_tokens):
        """Initial transition probabilities based on unigram distribution.

        Each tag is presumed to be succeeded by the expectation over all data
        of the number of prefixes, suffixes, stems, non-morphemes and word
        boundaries.

        Arguments:
            category_token_count -- A ByCategory with unnormalized
                                    morph token counts.
            num_word_tokens -- Total number of word tokens, for word boundary
                               probability.
        """

        transitions = collections.Counter()
        nclass = {WORD_BOUNDARY: num_word_tokens}
        for (i, category) in enumerate(CatmapModel.get_categories()):
            nclass[category] = float(category_token_count[i])

        num_tokens_tagged = collections.Counter()
        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in MorphUsageProperties.zero_transitions:
                    continue
                # count all possible valid transitions
                num_tokens_tagged[cat1] += nclass[cat2]

        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in MorphUsageProperties.zero_transitions:
                    continue
                transitions[(cat1, cat2)] = nclass[cat2]

        for pair in MorphUsageProperties.zero_transitions:
            transitions[pair] = 0
        # FIXME: ugly using privates of delegate class
        self._catmap_coding._transition_counts = transitions
        self._catmap_coding._cat_tagcount = num_tokens_tagged

    def _calculate_transition_counts(self, segmentations):
        """Count the number of transitions of each type.
        Can be used to estimate transition probabilities from
        a category-tagged segmented corpus.

        Arguments:
            segmentations -- Category-tagged segmented words.
                List of format:
                (count, (CategorizedMorph1, CategorizedMorph2, ...)), ...
        """
        self._catmap_coding.clear_transitions()
        for rcount, segments in segmentations:
            # Only the categories matter
            categories = [x.category for x in segments]
            # Include word boundaries
            categories.insert(0, WORD_BOUNDARY)
            categories.append(WORD_BOUNDARY)
            for (prev_cat, next_cat) in ngrams(categories, 2):
                pair = (prev_cat, next_cat)
                if pair in MorphUsageProperties.zero_transitions:
                        _logger.warning('Impossible transition ' +
                                        '{!r} -> {!r}'.format(*pair))
                self._catmap_coding.add_transitions(prev_cat, next_cat,
                                                    rcount)

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
        wb = categories.index(WORD_BOUNDARY)

        # Grid consisting of
        # the lowest accumulated cost ending in each possible state.
        # and back pointers that indicate the best path.
        # Initialized to pseudo-zero for all states
        ViterbiNode = collections.namedtuple('ViterbiNode',
                                             ['cost', 'backpointer'])
        grid = [[ViterbiNode(LOGPROB_ZERO, None)] * len(categories)]
        # Except probability one that first state is a word boundary
        grid[0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = []
        best = []

        for (i, morph) in enumerate(segments):
            for next_cat in range(len(categories)):
                if next_cat == wb:
                    # Impossible to visit boundary in the middle of the
                    # sequence
                    best.append(ViterbiNode(LOGPROB_ZERO, None))
                    continue
                for prev_cat in range(len(categories)):
                    pair = (categories[prev_cat], categories[next_cat])
                    # Cost of selecting prev_cat as previous state
                    # if now at next_cat
                    cost.append(grid[i][prev_cat].cost +
                                self._catmap_coding.transit_emit_cost(
                                pair[0], pair[1], morph))
                best.append(ViterbiNode(*_minargmin(cost)))
                cost = []
            # Update grid to prepare for next iteration
            grid.append(best)
            best = []

        # Last transition must be to word boundary
        for prev_cat in range(len(categories)):
            pair = (categories[prev_cat], WORD_BOUNDARY)
            cost = (grid[-1][prev_cat].cost +
                    self._catmap_coding.log_transitionprob(*pair))
            best.append(cost)
        backtrace = ViterbiNode(*_minargmin(best))

        # Backtrace for the best category sequence
        result = [CategorizedMorph(segments[-1],
                  categories[backtrace.backpointer])]
        for i in range(len(segments) - 1, 0, -1):
            backtrace = grid[i + 1][backtrace.backpointer]
            result.insert(0, CategorizedMorph(segments[i - 1],
                categories[backtrace.backpointer]))
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
            _logger.info('Iteration number {}/{}.'.format(iteration,
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
                    '{} differences in final iteration.'.format(differences))
                break
            _logger.info('{} differences.'.format(differences))
            previous_segmentation = current_segmentation

    def log_emissionprobs(self, morph):
        return self._catmap_coding.log_emissionprobs(morph)

    @staticmethod
    def get_categories(wb=False):
        """The category tags supported by this model.
        Argumments:
            wb -- If True, the word boundary will be included. Default: False.
        """
        categories = list(ByCategory.__slots__)
        if wb:
            categories.append(WORD_BOUNDARY)
        return tuple(categories)

    @staticmethod
    def _detag_morph(morph):
        if isinstance(morph, CategorizedMorph):
            return morph.morph
        return morph

    @staticmethod
    def _detag_segmentations(segmentations):
        for rcount, segments in segmentations:
            yield ((rcount, [CatmapModel._detag_morph(x) for x in segments]))

    @property
    def _log_unknownletterprob(self):
        """The probability of an unknown letter is defined to be the squared
        probability of the rarest known letter"""
        return 2 * max(self._log_letterprobs.values())


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
            return unicode(self.morph)
        return '{}/{}'.format(self.morph, self.category)

    def __eq__(self, other):
        return (self.morph == other.morph and
                self.category == other.category)


class Marginalizer(ByCategory):
    """An accumulator for marginalizing the class probabilities
    P(Category) from all the individual conditional probabilities
    P(Category|Morph) and observed morph probabilities P(Morph).

    First the unnormalized distribution is obtained by summing over
    #(Morph) * P(Category|Morph) over each morph, separately for each
    category. P(Category) is then obtained by normalizing the
    distribution.
    """

    def __init__(self):
        ByCategory.__init__(self)

    def add(self, rcount, condprobs):
        """Add the products #(Morph) * P(Category|Morph)
        for one observed morph."""
        for i, x in enumerate(condprobs):
            self[i] += float(rcount) * float(x)

    def normalized(self):
        """Returns the marginal probabilities for all categories."""
        total = self.total_token_count
        return ByCategory(*[x / total for x in self])

    @property
    def total_token_count(self):
        """Total number of tokens seen."""
        return sum(self)

    @property
    def category_token_count(self):
        """Tokens seen per category."""
        tmp = tuple(self)
        return ByCategory(*tmp)


class CatmapEncoding(morfessor.Encoding):
    """Class for calculating the encoding costs of the grammar and the
    corpus. Also stores the HMM parameters.
    """
    # can inherit without change: frequency_distribution_cost,

    def __init__(self, lexicon_encoding, weight=1.0):
        super(CatmapEncoding, self).__init__(weight)
        self.lexicon_encoding = lexicon_encoding

        # Posterior emission probabilities P(Morph|Category).
        # A dict of ByCategory objects indexed by morph. Log-probabilities.
        self._log_emissionprobs = dict()

        # Counts of emissions observed in the tagged corpus.
        # Not equivalent to _log_emissionprobs (which is the MAP estimate,
        # while these would give the ML estimate)
        # A dict of ByCategory objects indexed by morph. Counts occurences.
        self._emission_counts = dict()

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

        # Counts of observed category tags.
        # Single Counter object (ByCategory is unsuitable, need break also).
        self._cat_tagcount = collections.Counter()

        # Cache for transition logprobs, to avoid wasting effort recalculating.
        self._log_transitionprob_cache = dict()

    def set_log_emissionprobs(self, morph, lep):
        self._log_emissionprobs[morph] = lep

    def clear_transitions(self):
        self._transition_counts.clear()
        self._cat_tagcount.clear()
        self._log_transitionprob_cache.clear()

    def add_transitions(self, prev_cat, next_cat, rcount):
        rcount = float(rcount)
        self._transition_counts[(prev_cat, next_cat)] += rcount
        self._cat_tagcount[prev_cat] += rcount
        # invalidate cache
        self._log_transitionprob_cache.clear()

    def log_transitionprob(self, prev_cat, next_cat):
        pair = (prev_cat, next_cat)
        if pair not in self._log_transitionprob_cache:
            self._log_transitionprob_cache[pair] = (
                _zlog(self._transition_counts[(prev_cat, next_cat)]) -
                _zlog(self._cat_tagcount[prev_cat]))
        return self._log_transitionprob_cache[pair]

    def log_emissionprobs(self, morph):
        return self._log_emissionprobs[morph].copy()

    def transit_emit_cost(self, prev_cat, next_cat, morph):
        """Cost of transitioning from prev_cat to next_cat and emitting
        the morph."""
        return (self.log_transitionprob(prev_cat, next_cat) +
                self._log_emissionprobs[morph][next_cat])

# morfessor.LexiconEncoding can be used without modification


def sigmoid(value, treshold, slope):
    return 1.0 / (1.0 + math.exp(-slope * (value - treshold)))


def ngrams(sequence, n=2):
    window = []
    for item in sequence:
        window.append(item)
        if len(window) > n:
            # trim back to size
            window = window[-n:]
        if len(window) == n:
            yield(tuple(window))


def _minargmin(sequence):
    best = (None, None)
    for (i, value) in enumerate(sequence):
        if best[0] is None or value < best[0]:
            best = (value, i)
    return best


def _zlog(x):
    """Logarithm which uses constant value for log(0) instead of -inf"""
    if x == 0:
        return LOGPROB_ZERO
    return -math.log(x)


def _log_catprobs(probs):
    """Convenience function to convert a ByCategory object containing actual
    probabilities into one with log probabilities"""

    return ByCategory(*[_zlog(x) for x in probs])
