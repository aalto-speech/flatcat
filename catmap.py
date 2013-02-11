#!/usr/bin/env python
"""
Morfessor 2.0 Categories-MAP variant.
"""

import collections
import logging
import math

import morfessor

_logger = logging.getLogger(__name__)


class CatmapIO(morfessor.MorfessorIO):
    """Extends data file formats to include category tags."""

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
        for c in contexts:
            p = float(contexts[c]) / float(len(contexts))
            entropy -= p * math.log(p)
        return math.exp(entropy)


CatProbs = collections.namedtuple('CatProbs', ['PRE', 'STM', 'SUF', 'ZZZ'])


class ClassProbs:
    def __init__(self, total_morph_tokens):
        self.total_morph_tokens = float(total_morph_tokens)
        self.probs = None

    def add(self, rcount, catprobs):
        if self.probs is None:
            self.probs = [0.0] * len(catprobs)
        freq = float(rcount) / self.total_morph_tokens
        for i, x in enumerate(catprobs):
            self.probs[i] += freq * float(x)

    def get(self):
        return CatProbs(self.probs)


class CatmapModel:
    """Morfessor Categories-MAP model class."""

    word_boundary = object()

    def __init__(self, ppl_treshold=100, ppl_slope=None, length_treshold=3,
                 length_slope=2):
        self.ppl_treshold = float(ppl_treshold)
        self.length_treshold = float(length_treshold)
        self.length_slope = float(length_slope)
        if ppl_slope is not None:
            self.ppl_slope = float(ppl_slope)
        else:
            self.ppl_slope = 10.0 / self.ppl_treshold

    def load_baseline(self, segmentations):
        self.contexts = collections.defaultdict(MorphContext)
        total_morph_tokens = 0
        for rcount, segments in segmentations:
            total_morph_tokens += len(segments)
            for (i, morph) in enumerate(segments):
                if i == 0:
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i - 1]
                self.contexts[morph].left[neighbour] += rcount
                if i == len(segments) - 1:
                    neighbour = segments[i - 1]
                else:
                    neighbour = CatmapModel.word_boundary
                self.contexts[morph].right[neighbour] += rcount
                self.contexts[morph].rcount += rcount

        classprobs = ClassProbs(total_morph_tokens)
        for morph in self.contexts:
            catprobs = self._contextToProbability(morph, self.contexts[morph])
            # Scale by frequency and accumulate elementwise
            classprobs.add(self.contexts[morph].rcount, catprobs)
            print("%s: %s" % (morph, catprobs))         # FIXME debug
        print("classprobs: %s" % (classprobs.probs,))   # FIXME debug

    def _contextToProbability(self, morph, context):
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
