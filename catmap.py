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


class CatmapModel:
    """Morfessor Categories-MAP model class."""

    def __init__(self):
        pass

    def load_baseline(self, segmentations):
        self.contexts = collections.defaultdict(MorphContext)
        for rcount, segments in segmentations:
            for (i, morph) in enumerate(segments):
                if i > 0:
                    self.contexts[morph].left[segments[i - 1]] += rcount
                if i < len(segments) - 1:
                    self.contexts[morph].right[segments[i + 1]] += rcount
                self.contexts[morph].rcount += rcount

        # debug
        for morph in self.contexts:
            context = self.contexts[morph]
            print("morph '%s', rcount %d, ppr %f, ppl %f" % (morph,
                context.rcount, context.right_perplexity,
                context.left_perplexity))
