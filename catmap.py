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
            entropy -= p * math.log(p)
        return math.exp(entropy)


CatProbs = collections.namedtuple('CatProbs', ['PRE', 'STM', 'SUF', 'ZZZ'])
ProbN = collections.namedtuple('ProbN', ['PROB', 'N'])


class CatmapModel:
    """Morfessor Categories-MAP model class."""

    word_boundary = object()

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

    def load_baseline(self, segmentations):
        """Initialize the model using the segmentation produced by a morfessor
        baseline model.

        Initialization is required before the model is ready for the Cat-MAP
        learning.

        Arguments:
            segmentations -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
        """
        total_morph_tokens = 0
        num_word_types = 0
        num_word_tokens = 0

        for rcount, segments in segmentations:
            num_word_types += 1
            num_word_tokens += rcount
            # Collect information about the contexts in which the morphs occur
            if self._use_word_tokens:
                pcount = rcount
            else:
                # pcount used for perplexity, rcount is real count
                pcount = 1
            total_morph_tokens += len(segments)
            for (i, morph) in enumerate(segments):
                if i == 0:
                    # word boundaries are counted as separate contexts
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i - 1]
                    # contexts shorter than treshold don't affect perplexity
                    if len(neighbour) < self._min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self._contexts[morph].left[neighbour] += pcount

                if i == len(segments) - 1:
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i + 1]
                    if len(neighbour) < self._min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self._contexts[morph].right[neighbour] += pcount

                self._contexts[morph].rcount += rcount

        class Marginalizer:
            """An accumulator for marginalizing the class probabilities
            P(Category) from all the individual conditional probabilities
            P(Category|Morph)
            """

            def __init__(self):
                self._counts = [0.0] * len(CatProbs._fields)

            def add(self, rcount, condprobs):
                """Add the conditional probabilities P(Category|Morph)
                for one observed morph. Once all observed morphs have been
                added, the marginalization is complete."""
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

        # Calculate conditional probabilities from the encountered contexts
        marginalizer = Marginalizer()
        for morph in sorted(self._contexts, cmp=lambda x, y: len(x) < len(y)):
            self._condprobs[morph] = self._context_to_probability(morph,
                self._contexts[morph])
            # Marginalize (scale by frequency and accumulate elementwise)
            marginalizer.add(self._contexts[morph].rcount,
                           self._condprobs[morph])
        self._catpriors = _log_catprobs(marginalizer.normalized())

        # Calculate posterior emission probabilities
        category_totals = marginalizer.category_token_count
        for morph in self._contexts:
            tmp = []
            for (i, total) in enumerate(category_totals):
                tmp.append(self._condprobs[morph][i] *
                           self._contexts[morph].rcount / category_totals[i])
            self._log_emissionprobs[morph] = _log_catprobs(CatProbs(*tmp))

        self._log_transitionprobs = CatmapModel._unigram_transition_probs(
            category_totals, num_word_tokens)

    def _context_to_probability(self, morph, context):
        """Calculate conditional probabilities P(Category|Morph) from the
        contexts in which the morphs occur.
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

    @staticmethod
    def _unigram_transition_probs(category_token_count, num_word_tokens):
        """Initial transition probabilities based on unigram distribution

        Each tag is presumed to be succeeded by the expectation over all data
        of the number of prefixes, suffixes, stems, non-morphemes and word
        boundaries.
        """

        wb = CatmapModel.word_boundary
        zeros = ((wb, wb), ('PRE', wb), ('PRE', 'SUF'), (wb, 'SUF'))

        transitions = dict()
        nclass = {wb: num_word_tokens}
        for (i, category) in enumerate(CatProbs._fields):
            nclass[category] = float(category_token_count[i])

        num_tokens_tagged = collections.defaultdict(int)
        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in zeros:
                    continue
                # count all possible valid transitions
                num_tokens_tagged[cat1] += nclass[cat2]

        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in zeros:
                    continue
                transitions[(cat1, cat2)] = ProbN(_zlog(nclass[cat2] /
                                                  num_tokens_tagged[cat1]),
                                                  nclass[cat2])

        for pair in zeros:
            transitions[pair] = ProbN(LOGPROB_ZERO, 0)

        return transitions


def sigmoid(value, treshold, slope):
    return 1.0 / (1.0 + math.exp(-slope * (value - treshold)))


def _zlog(x):
    """Logarithm which uses constant value for log(0) instead of
    raising exception"""

    if x == 0:
        return LOGPROB_ZERO
    return -math.log(x)


def _log_catprobs(probs):
    """Convenience function to convert a CatProbs object containing actual
    probabilities into one with log probabilities"""

    return CatProbs(*[_zlog(x) for x in probs])


# Temporary helper function for my common testing setup
# FIXME: remove when command-line configurable main function is written
def debug_trainbaseline():
    baseline = morfessor.BaselineModel()
    io = morfessor.MorfessorIO(encoding='latin-1')
    data = io.read_corpus_list_file('mydata.gz')
    c = baseline.load_data(data)
    e, c = baseline.train_batch('recursive')
    return baseline
