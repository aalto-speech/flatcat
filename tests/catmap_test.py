#!/usr/bin/env python
"""
Tests for Morfessor 2.0 Categories-MAP variant.
"""

import unittest
import re

import morfessor
import catmap

# Directory for reference input and output files
REFERENCE_DIR = 'reference_data/'
# A baseline segmentation, e.g. produced by morphessor 0.9.2
REFERENCE_BASELINE_SEGMENTATION = REFERENCE_DIR + 'baselineseg.final.gz'
# Probabilities estimated from the above baseline segmentation
REFERENCE_PROBS = REFERENCE_DIR + 'baseline.probs.gz'


class TestProbabilityEstimation(unittest.TestCase):
    def setUp(self):
        self.perplexities = dict()
        self.condprobs = dict()
        self.posteriors = dict()
        self.transitions = dict()
        catpriors_tmp = dict()

        baseline = morfessor.BaselineModel()
        io = morfessor.MorfessorIO(encoding='latin-1')

        baseline.load_segmentations(io.read_segmentation_file(
            REFERENCE_BASELINE_SEGMENTATION))
        self.model = catmap.CatmapModel(ppl_treshold=10, ppl_slope=1,
                                   length_treshold=3, length_slope=2,
                                   use_word_tokens=False)
        self.model.load_baseline(baseline.get_segmentations())

        comments_io = morfessor.MorfessorIO(encoding='latin-1',
                                            comment_start='++++++++++')

        pattern_float = r'([0-9.]+)'
        pattern_int = r'([0-9]+)'
        pattern_quoted = r'"([^"]*)"'
        ppl_re = re.compile(r'^#Features\(' + pattern_quoted + r'\)\s+' +
            pattern_float + r'\s+' + pattern_float + r'\s+' + pattern_int)
        condprobs_re = re.compile(r'^#P\(Tag\|' + pattern_quoted + r'\)\s+' +
            pattern_float + r'\s+' + pattern_float + r'\s+' +
            pattern_float + r'\s+' + pattern_float)
        catpriors_re = re.compile(r'^#PTag\(' + pattern_quoted + r'\)\s+' +
                                  pattern_float)
        posteriors_re = re.compile(r'^(\S*)\s+' +
            pattern_float + r'\s+' + pattern_float + r'\s+' +
            pattern_float + r'\s+' + pattern_float)
        transitions_re = re.compile(r'^P\((\S+) .. ([^\)]+)\) = ' +
             pattern_float + r' \(N = ' + pattern_int + '\)')

        for line in comments_io._read_text_file(REFERENCE_PROBS):
            m = ppl_re.match(line)
            if m:
                self.perplexities[m.group(1)] = (float(m.group(2)),
                                                 float(m.group(3)),
                                                 int(m.group(4)))
                continue

            m = condprobs_re.match(line)
            if m:
                self.condprobs[m.group(1)] = (float(m.group(2)),
                                              float(m.group(3)),
                                              float(m.group(4)),
                                              float(m.group(5)))
                continue

            m = catpriors_re.match(line)
            if m:
                catpriors_tmp[m.group(1)] = float(m.group(2))
                continue

            m = posteriors_re.match(line)
            if m:
                self.posteriors[m.group(1)] = catmap.CatProbs(
                    float(m.group(2)), float(m.group(3)),
                    float(m.group(4)), float(m.group(5)))
                continue

            m = transitions_re.match(line)
            if m:
                def _tr_wb(x):
                    if x == '#':
                        return catmap.CatmapModel.word_boundary
                    else:
                        return x

                cats = tuple([_tr_wb(x) for x in (m.group(1), m.group(2))])
                self.transitions[cats] = (float(m.group(3)), int(m.group(4)))

        self.catpriors = catmap.CatProbs(*(catpriors_tmp[x] for x in
                                           catmap.CatProbs._fields))

    def test_perplexities(self):
        for morph in self.perplexities:
            reference = self.perplexities[morph]
            if morph not in self.model._contexts:
                raise KeyError('%s not in observed morphs' % (morph,))
            observed = self.model._contexts[morph]
            msg = '%s perplexity of %s, %s not almost equal to %s'
            tmp = observed.right_perplexity
            self.assertAlmostEqual(tmp, reference[0], places=3,
                                 msg=msg % ('right', morph, tmp, reference[0]))
            tmp = observed.left_perplexity
            self.assertAlmostEqual(tmp, reference[1], places=3,
                                 msg=msg % ('left', morph, tmp, reference[1]))
            # checking lenght of morph is useless,
            # when we know it already was found

    def test_condprobs(self):
        for morph in self.condprobs:
            reference = self.condprobs[morph]
            if morph not in self.model._condprobs:
                raise KeyError('%s not in observed morphs' % (morph,))
            observed = self.model._condprobs[morph]
            msg = 'P(%s | "%s"), %s not almost equal to %s'

            for (i, category) in enumerate(catmap.CatProbs._fields):
                self.assertAlmostEqual(observed[i], reference[i], places=9,
                    msg=msg % (category, morph, observed[i], reference[i]))

    def test_catpriors(self):
        for (i, category) in enumerate(catmap.CatProbs._fields):
            reference = self.catpriors
            observed = self.model._catpriors
            msg = 'P(%s), %s not almost equal to %s'
            self.assertAlmostEqual(observed[i], reference[i], places=9,
                msg=msg % (category, observed[i], reference[i]))

    def test_posterior_emission_probs(self):
        for morph in self.posteriors:
            reference = self.posteriors[morph]
            if morph not in self.model._emissionprobs:
                raise KeyError('%s not in observed morphs' % (morph,))
            observed = self.model._emissionprobs[morph]
            msg = 'P(%s | "%s"), %s not almost equal to %s'

            for (i, category) in enumerate(catmap.CatProbs._fields):
                self.assertAlmostEqual(observed[i], reference[i], places=9,
                    msg=msg % (morph, category, observed[i], reference[i]))

    def test_transitions(self):
        categories = list(catmap.CatProbs._fields)
        categories.append(catmap.CatmapModel.word_boundary)
        msg = 'P(%s -> %s), %s not almost equal to %s'
        reference = self.transitions
        observed = self.model._transitionprobs
        for cat1 in categories:
            for cat2 in categories:
                pair = (cat1, cat2)
                self.assertAlmostEqual(observed[pair][0], reference[pair][0],
                                       places=9,
                                       msg=msg % (cat1, cat2,
                                                  observed[pair][0],
                                                  reference[pair][0]))

if __name__ == '__main__':
    unittest.main()
