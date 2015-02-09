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

from .categorizationscheme import ByCategory, get_categories, CategorizedMorph
from .categorizationscheme import MorphUsageProperties
from .flatcat import AbstractSegmenter, FlatcatAnnotatedCorpusEncoding
from .utils import LOGPROB_ZERO, zlog

_logger = logging.getLogger(__name__)


class FlatcatSegmenter(AbstractSegmenter):
    def __init__(self, model):
        self._corpus_coding = ReducedEncoding(
            model._corpus_coding, model._morph_usage)
        super(FlatcatSegmenter, self).__init__(self._corpus_coding,
                                               model.nosplit_re)
        self._segment_only = True
        self._initialized = True
        self._corpus_tagging_level = 'full'

        if model.annotations is None:
            self.annotations = None
        else:
            self._supervised = True
            self.annotations = model.annotations
            self._annotations_tagged = True
            for (word, annot) in self.annotations.items():
                if annot.alternatives[0][0].category is None:
                    self._annotations_tagged = False
            self._annot_coding = model._annot_coding
        self._num_compounds = model.num_compounds
        self._num_constructions = model.num_constructions
        self._all_chars = dict(model._lexicon_coding.atoms)

    def __contains__(self, morph):
        return morph in self._corpus_coding._log_emissionprob_cache

    def __setstate__(self, d):
        """Temporary hack to allow loading old reduced models"""
        self.__dict__ = d
        self._initialized = True
        self._corpus_tagging_level = 'full'
        if 'forcesplit' not in self.__dict__:
            self.forcesplit = [':', '-']

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

    def __init__(self, corpus_encoding, morph_usage):
        # Transition and emission logprobs,
        # The reduced model only stores these
        self._log_transitionprob_cache = self._populate_transitions(
            corpus_encoding)
        self._log_emissionprob_cache = self._populate_emissions(
            corpus_encoding, morph_usage)

        self.weight = corpus_encoding.weight
        self.cost = corpus_encoding.get_cost()
        self.boundaries = corpus_encoding.boundaries

    # Transition count methods

    def _populate_transitions(self, corpus_encoding):
        out = {}
        categories = get_categories(wb=True)
        for prev_cat in categories:
            for next_cat in categories:
                out[(prev_cat, next_cat)] = (
                    corpus_encoding.log_transitionprob(prev_cat, next_cat))
        return out

    def _populate_emissions(self, corpus_encoding, morph_usage):
        out = {}
        categories = get_categories(wb=False)
        for morph in morph_usage.seen_morphs():
            out[morph] = ByCategory(
                *[corpus_encoding.log_emissionprob(cat, morph)
                  for cat in categories])
            corpus_encoding.clear_emission_cache()
        return out

    def log_transitionprob(self, prev_cat, next_cat):
        """-Log of transition probability P(next_cat|prev_cat)"""
        pair = (prev_cat, next_cat)
        return self._log_transitionprob_cache[pair]

    def log_emissionprob(self, category, morph, extrazero=False):
        """-Log of posterior emission probability P(morph|category)"""
        categories = get_categories(wb=False)
        if morph not in self._log_emissionprob_cache:
            # The morph is not present in this reduced model
            return LOGPROB_ZERO
        tmp = self._log_emissionprob_cache[morph][categories.index(category)]
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
