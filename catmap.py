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


class CatmapModel:
    """Morfessor Categories-MAP model class."""

    word_boundary = object()

    def __init__(self, ppl_treshold=100, ppl_slope=None, length_treshold=3,
                 length_slope=2, use_word_tokens=True,
                 min_perplexity_length=4):
        self._ppl_treshold = float(ppl_treshold)
        self._length_treshold = float(length_treshold)
        self._length_slope = float(length_slope)
        self._use_word_tokens = bool(use_word_tokens)
        self._min_perplexity_length = int(min_perplexity_length)
        if ppl_slope is not None:
            self._ppl_slope = float(ppl_slope)
        else:
            self._ppl_slope = 10.0 / self._ppl_treshold

        # Counts of different contexts in which a morph occurs
        self._contexts = collections.defaultdict(MorphContext)

        # Conditional probabilities P(Category|Morph)
        self._condprobs = dict()

        # Priors for categories P(Category)
        self_catpriors = None

        # Posterior emission probabilities P(Morph|Category)
        self._emissionprobs = dict()

    def load_baseline(self, segmentations):
        """Initialize the model using the segmentation produced by a morfessor
        baseline model.

        Initialization is required before the model is ready for the Cat-MAP
        learning.
        """
        total_morph_tokens = 0

        for rcount, segments in segmentations:
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
                self.probs = [0.0] * len(CatProbs._fields)

            def add(self, rcount, condprobs):
                for i, x in enumerate(condprobs):
                    self.probs[i] += float(rcount) * float(x)

            def normalized(self):
                total = sum(self.probs)
                return CatProbs(*[x / total for x in self.probs])

            def get(self):
                return CatProbs(*self.probs)

        # Calculate conditional probabilities from the encountered contexts
        classprobs = Marginalizer()
        for morph in sorted(self._contexts, cmp=lambda x, y: len(x) < len(y)):
            self._condprobs[morph] = self._contextToProbability(morph,
                self._contexts[morph])
            # Marginalize (scale by frequency and accumulate elementwise)
            classprobs.add(self._contexts[morph].rcount,
                           self._condprobs[morph])
        self._catpriors = classprobs.normalized()

        # Calculate posterior emission probabilities
        category_totals = classprobs.get()
        for morph in self._contexts:
            tmp = []
            for (i, total) in enumerate(category_totals):
                tmp.append(self._condprobs[morph][i] *
                           self._contexts[morph].rcount / category_totals[i])
            self._emissionprobs[morph] = CatProbs(*tmp)

    def _contextToProbability(self, morph, context):
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


def sigmoid(value, treshold, slope):
    return 1.0 / (1.0 + math.exp(-slope * (value - treshold)))


# Temporary helper function for my common testing setup
# FIXME: remove when command-line configurable main function is written
def debug_trainbaseline():
    baseline = morfessor.BaselineModel()
    io = morfessor.MorfessorIO(encoding='latin-1')
    data = io.read_corpus_list_file('mydata.gz')
    c = baseline.load_data(data)
    e, c = baseline.train_batch('recursive')
    return baseline
