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
        self.ppl_treshold = float(ppl_treshold)
        self.length_treshold = float(length_treshold)
        self.length_slope = float(length_slope)
        self.use_word_tokens = bool(use_word_tokens)
        self.min_perplexity_length = int(min_perplexity_length)
        if ppl_slope is not None:
            self.ppl_slope = float(ppl_slope)
        else:
            self.ppl_slope = 10.0 / self.ppl_treshold

    def load_baseline(self, segmentations):
        """Initialize the model using the segmentation produced by a morfessor
        baseline model.

        Initialization is required before the model is ready for the Cat-MAP
        learning.
        """
        self.contexts = collections.defaultdict(MorphContext)
        self.condprobs = dict()
        total_morph_tokens = 0

        for rcount, segments in segmentations:
            # Collect information about the contexts in which the morphs occur
            if self.use_word_tokens:
                pcount = rcount
            else:
                # pcount used for perplexity, rcount is real count
                pcount = 1
            total_morph_tokens += len(segments)
            for (i, morph) in enumerate(segments):
                if i == 0:
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i - 1]
                    if len(neighbour) < self.min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self.contexts[morph].left[neighbour] += pcount

                if i == len(segments) - 1:
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i + 1]
                    if len(neighbour) < self.min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self.contexts[morph].right[neighbour] += pcount

                self.contexts[morph].rcount += rcount

        class Marginalizer:
            """An accumulator for marginalizing the class probabilities
            P(Category) from all the individual conditional probabilities
            P(Category|Morph)
            """

            def __init__(self, total_morph_tokens):
                self.total_morph_tokens = float(total_morph_tokens)
                self.probs = [0.0] * len(CatProbs._fields)

            def add(self, rcount, condprobs):
                freq = float(rcount) / self.total_morph_tokens
                for i, x in enumerate(condprobs):
                    self.probs[i] += freq * float(x)

            def normalize(self):
                total = float(sum(self.probs))
                self.probs = [x / total for x in self.probs]

            def get(self):
                return CatProbs(*self.probs)

        # Calculate conditional probabilities from the encountered contexts
        classprobs = Marginalizer(total_morph_tokens)
        for morph in sorted(self.contexts, cmp=lambda x, y: len(x) < len(y)):
            self.condprobs[morph] = self._contextToProbability(morph,
                self.contexts[morph])
            # Marginalize (scale by frequency and accumulate elementwise)
            classprobs.add(self.contexts[morph].rcount, self.condprobs[morph])
        classprobs.normalize()
        self.catpriors = classprobs.get()

    def _contextToProbability(self, morph, context):
        """Calculate conditional probabilities P(Category|Morph) from the
        contexts in which the morphs occur.
        """

        prelike = sigmoid(context.right_perplexity, self.ppl_treshold,
                            self.ppl_slope)
        suflike = sigmoid(context.left_perplexity, self.ppl_treshold,
                            self.ppl_slope)
        stmlike = sigmoid(len(morph), self.length_treshold,
                            self.length_slope)

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
