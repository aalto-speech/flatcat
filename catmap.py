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
        total_tokens = float(sum(contexts.values()))
        for c in contexts:
            p = float(contexts[c]) / total_tokens
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
        return CatProbs(*self.probs)


class CatmapModel:
    """Morfessor Categories-MAP model class."""

    word_boundary = object()

    def __init__(self, ppl_treshold=100, ppl_slope=None, length_treshold=3,
                 length_slope=2, use_word_tokens=True, min_perplexity_length=4):
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
        self.contexts = collections.defaultdict(MorphContext)
        total_morph_tokens = 0
        for rcount, segments in segmentations:
            if not self.use_word_tokens:
                rcount = 1
            total_morph_tokens += len(segments)
            for (i, morph) in enumerate(segments):
                if i == 0:
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i - 1]
                    if len(neighbour) < self.min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self.contexts[morph].left[neighbour] += rcount

                if i == len(segments) - 1:
                    neighbour = CatmapModel.word_boundary
                else:
                    neighbour = segments[i + 1]
                    if len(neighbour) < self.min_perplexity_length:
                        neighbour = None
                if neighbour is not None:
                    self.contexts[morph].right[neighbour] += rcount

                self.contexts[morph].rcount += rcount

        classprobs = ClassProbs(total_morph_tokens)
        for morph in sorted(self.contexts, cmp=lambda x, y: len(x) < len(y)):
            catprobs = self._contextToPrior(morph, self.contexts[morph])
            # Scale by frequency and accumulate elementwise
            classprobs.add(self.contexts[morph].rcount, catprobs)
            print(u'#P(Tag|"{0:s}")\t{1:.10f}\t{2:.10f}\t{3:.10f}\t{4:.10f}'.format(morph, *catprobs))  # FIXME debug
        tmp = classprobs.get()
        print('#PTag("PRE")\t{0:.10f}'.format(tmp.PRE))   # FIXME debug
        print('#PTag("STM")\t{0:.10f}'.format(tmp.STM))   # FIXME debug
        print('#PTag("SUF")\t{0:.10f}'.format(tmp.SUF))   # FIXME debug
        print('#PTag("ZZZ")\t{0:.10f}'.format(tmp.ZZZ))   # FIXME debug

    def _contextToPrior(self, morph, context):
        crap = u'#Features("{0:s}")\t{1:.4f}\t{2:.4f}\t{3:d}'.format(morph, context.right_perplexity, context.left_perplexity, len(morph)) # FIXME debug
        print(crap) # FIXME debug
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


def debug_trainbaseline():
    baseline = morfessor.BaselineModel()
    io = morfessor.MorfessorIO(encoding='latin-1')
    data = io.read_corpus_list_file('mydata.gz')
    c = baseline.load_data(data)
    e, c = baseline.train_batch('recursive')
    return baseline

# FIXME temporary rough test against old morfessor

baseline = morfessor.BaselineModel()
io = morfessor.MorfessorIO(encoding='latin-1')

baseline.load_segmentations(io.read_segmentation_file('/akulabra/home/t40511/sgronroo/Downloads/morfessor_catmap0.9.2/train/baselineseg.final.gz'))
model = CatmapModel(ppl_treshold=10, ppl_slope=1, length_treshold=3, length_slope=2, use_word_tokens=False)
model.load_baseline(baseline.get_segmentations())

