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

        pattern_float = r'([0-9.]*)'
        pattern_int = r'([0-9]*)'
        pattern_quoted = r'"([^"]*)"'
        ppl_re = re.compile(r'^#Features\(' + pattern_quoted + r'\)\s+' +
            pattern_float + r'\s+' + pattern_float + r'\s+' + pattern_int)
        condprobs_re = re.compile(r'^#P\(Tag\|' + pattern_quoted + r'\)\s+' +
            pattern_float + r'\s+' + pattern_float + r'\s+' +
            pattern_float + r'\s+' + pattern_float)
        catpriors_re = re.compile(r'^#PTag\(' + pattern_quoted + r'\)\s+' +
                                  pattern_float)
        

        for line in comments_io._read_text_file(REFERENCE_PROBS):
            m = ppl_re.match(line)
            if m:
                self.perplexities[m.group(1)] = (float(m.group(2)),
                                                 float(m.group(3)),
                                                 int(m.group(4)))
            m = condprobs_re.match(line)
            if m:
                self.condprobs[m.group(1)] = (float(m.group(2)),
                                             float(m.group(3)),
                                             float(m.group(4)),
                                             float(m.group(5)))

            m = catpriors_re.match(line)
            if m:
                catpriors_tmp[m.group(1)] = float(m.group(2))
        self.catpriors = catmap.CatProbs(*(catpriors_tmp[x] for x in
                                           catmap.CatProbs._fields))

    def test_perplexities(self):
        for morph in self.perplexities:
            reference = self.perplexities[morph]
            if morph not in self.model.contexts:
                raise KeyError('%s not in observed morphs' % (morph,))
            observed = self.model.contexts[morph]
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
            if morph not in self.model.condprobs:
                raise KeyError('%s not in observed morphs' % (morph,))
            observed = self.model.condprobs[morph]
            msg = 'P(%s | "%s"), %s not almost equal to %s'

            for (i, category) in enumerate(catmap.CatProbs._fields):
                self.assertAlmostEqual(observed[i], reference[i], places=9,
                    msg=msg % (category, morph, observed[i], reference[i]))

    def test_catpriors(self):
        for (i, category) in enumerate(catmap.CatProbs._fields):
            reference = self.catpriors
            observed = self.model.catpriors
            msg = 'P(%s), %s not almost equal to %s'
            self.assertAlmostEqual(observed[i], reference[i], places=9,
                msg=msg % (category, observed[i], reference[i]))

    #def test_posteriors(self):
    #    pass

if __name__ == '__main__':
    unittest.main()
