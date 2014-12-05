#!/usr/bin/env python
"""
Reduced models for segmenting with less memory overhead
"""
from __future__ import unicode_literals

import collections
import logging
import math
import random
import re
import sys

from . import utils
from .categorizationscheme import ByCategory, get_categories, CategorizedMorph
from .categorizationscheme import MorphUsageProperties
from .flatcat import AbstractSegmenter, FlatcatEncoding
from .flatcat import FlatcatAnnotatedCorpusEncoding
from .utils import LOGPROB_ZERO, zlog

_logger = logging.getLogger(__name__)

class FlatcatSegmenter(AbstractSegmenter):
    def __init__(self, morph_usage, lexicon_coding, corpus_coding,
                 num_compounds, num_constructions,
                 annotations=None, annotatedcorpusweight=None):
        super(FlatcatSegmenter, self).__init__(morph_usage,
                                               corpusweight=1.0)
        self._lexicon_coding = lexicon_coding
        self._corpus_coding = corpus_coding
        self._segment_only = True

        self.annotations = None
        if annotations is not None:
            self._supervised = True
            self.annotations = annotations
            self._annotations_tagged = True
            for (word, annot) in annotations.items():
                if annot.alternatives[0][0].category is None:
                    self._annotations_tagged = False
            self._annot_coding = FlatcatAnnotatedCorpusEncoding(
                                    self._corpus_coding,
                                    weight=annotatedcorpusweight)
            self._annot_coding.boundaries = len(self.annotations)
        self._num_compounds = num_compounds
        self._num_constructions = num_constructions

    @property
    def num_compounds(self):
        """Compound (word) types"""
        return self._num_compounds

    @property
    def num_constructions(self):
        """Construction (morph) types"""
        return self._num_constructions


class ReducedEncoding(object):
    """Reduced variant of FlatcatEncoding """

    def __init__(self, corpus_encoding):
        # Transition and emission logprobs,
        # The reduced model only stores these
        # FIXME: not working (copied as is, not guaranteed to be filled)
        # FIXME: also, the format results in a huge number of keys
        self._log_transitionprob_cache = dict(
            corpus_encoding._log_transitionprob_cache)
        self._log_emissionprob_cache = dict(
            corpus_encoding._log_emissionprob_cache)

        self.cost = corpus_encoding.get_cost()
        self.boundaries = corpus_encoding.boundaries

    # Transition count methods

    def log_transitionprob(self, prev_cat, next_cat):
        """-Log of transition probability P(next_cat|prev_cat)"""
        pair = (prev_cat, next_cat)
        return self._log_transitionprob_cache[pair]

    def log_emissionprob(self, category, morph, extrazero=False):
        """-Log of posterior emission probability P(morph|category)"""
        pair = (category, morph)
        tmp = self._log_emissionprob_cache[pair]
        if extrazero and tmp >= LOGPROB_ZERO:
            return tmp ** 2
        return tmp

    def transit_emit_cost(self, prev_cat, next_cat, morph):
        """Cost of transitioning from prev_cat to next_cat and emitting
        the morph."""
        if (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
            return LOGPROB_ZERO
        return (self.log_transitionprob(prev_cat, next_cat) +
                self.log_emissionprob(next_cat, morph))

    def get_cost(self):
        """
        This is P( D_W | theta, Y )
        """
        return self.cost
