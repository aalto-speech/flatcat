#!/usr/bin/env python
"""
Tests for Morfessor 2.0 Categories-MAP variant.
"""

import collections
import logging
import math
import re
import unittest

import morfessor
import catmap


# Directory for reference input and output files
REFERENCE_DIR = 'reference_data/'
# A baseline segmentation, e.g. produced by morphessor 0.9.2
REFERENCE_BASELINE_SEGMENTATION = REFERENCE_DIR + 'baselineseg.final.gz'
# Probabilities estimated from the above baseline segmentation
REFERENCE_BASELINE_PROBS = REFERENCE_DIR + 'baseline.probs.gz'
# Initial viterbi tagging
REFERENCE_BASELINE_TAGGED = REFERENCE_DIR + 'baseline.i.tagged.gz'
# Probabilities re-estimated based on above viterbi tagging
REFERENCE_REESTIMATE_PROBS = REFERENCE_DIR + 'baseline.i.probs.gz'

logging.basicConfig()


def _load_baseline():
        baseline = morfessor.BaselineModel()
        io = morfessor.MorfessorIO(encoding='latin-1')

        baseline.load_segmentations(io.read_segmentation_file(
            REFERENCE_BASELINE_SEGMENTATION))
        return baseline


def _load_catmap(baseline_seg, no_emissions=False):
        m_usage = catmap.MorphUsageProperties(ppl_treshold=10, ppl_slope=1,
                                              length_treshold=3,
                                              length_slope=2,
                                              use_word_tokens=False)
        model = catmap.CatmapModel(m_usage)
        model.load_baseline(baseline_seg, no_emissions)
        return model


class TestProbabilityEstimation(unittest.TestCase):
    def _config(self):
        """Overridden later to test re-estimation"""
        self.reference_file = REFERENCE_BASELINE_PROBS
        self.baseline = _load_baseline()
        self.model = _load_catmap(self.baseline.get_segmentations(),
                                  no_emissions=True)

    def setUp(self):
        self.perplexities = dict()
        self.condprobs = dict()
        self.posteriors = dict()
        self.transitions = dict()
        catpriors_tmp = dict()

        self._config()

        self.comments_io = morfessor.MorfessorIO(encoding='latin-1',
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

        for line in self.comments_io._read_text_file(self.reference_file):
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
                self.posteriors[m.group(1)] = catmap.ByCategory(
                    float(m.group(2)), float(m.group(3)),
                    float(m.group(4)), float(m.group(5)))
                continue

            m = transitions_re.match(line)
            if m:
                def _tr_wb(x):
                    if x == '#':
                        return catmap.CatmapModel.word_boundary
                    return x

                cats = tuple([_tr_wb(x) for x in (m.group(1), m.group(2))])
                self.transitions[cats] = (float(m.group(3)), int(m.group(4)))

        self.catpriors = catmap.ByCategory(*(catpriors_tmp[x] for x in
                                           self.model.get_categories()))

    def test_perplexities(self):
        for morph in self.perplexities:
            reference = self.perplexities[morph]
            if morph not in self.model._morph_usage.seen_morphs():
                raise KeyError('%s not in observed morphs' % (morph,))
            observed = self.model._morph_usage._contexts[morph]
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
            if morph not in self.model._morph_usage.seen_morphs():
                raise KeyError('%s not in observed morphs' % (morph,))
            observed = self.model._morph_usage.condprobs(morph)
            msg = 'P(%s | "%s"), %s not almost equal to %s'

            for (i, category) in enumerate(self.model.get_categories()):
                self.assertAlmostEqual(observed[i], reference[i], places=9,
                    msg=msg % (category, morph, observed[i], reference[i]))

    def test_catpriors(self):
        # Need to set category totals manually
        marginalizer = catmap.Marginalizer()
        for morph in self.model._morph_usage.seen_morphs():
            # Un-normalized marginalzation
            # (scale by frequency and accumulate elementwise)
            marginalizer.add(self.model._morph_usage.count(morph),
                             self.model._morph_usage.condprobs(morph))
        observed = marginalizer.normalized()

        for (i, category) in enumerate(self.model.get_categories()):
            reference = self.catpriors
            msg = 'P(%s), %s not almost equal to %s'
            self.assertAlmostEqual(observed[i], reference[i], places=9,
                msg=msg % (category, observed[i], reference[i]))

    def test_posterior_emission_probs(self):
        sumA = 0.0
        sumB = 0.0
        for morph in self.posteriors:
            reference = self.posteriors[morph]
            msg = 'P(%s | "%s"), %s not almost equal to %s'

            for (i, category) in enumerate(self.model.get_categories()):
                try:
                    observed = _zexp(
                        self.model._catmap_coding.log_emissionprob(category,
                                                                   morph))
                except KeyError:
                    raise KeyError('%s not in observed morphs' % (morph,))
                self.assertAlmostEqual(observed, reference[i], places=9,
                    msg=msg % (morph, category, observed, reference[i]))
                sumA += observed
                sumB += reference[i]
        #print('sums {} vs {}'.format(sumA, sumB))

    def test_transitions(self):
        categories = self.model.get_categories(True)
        msg = 'P(%s -> %s), %s not almost equal to %s'
        reference = self.transitions
        for cat1 in categories:
            for cat2 in categories:
                pair = (cat1, cat2)
                obsval = _zexp(self.model._catmap_coding.log_transitionprob(
                                                                    *pair))
                self.assertAlmostEqual(obsval, reference[pair][0], places=9,
                                       msg=msg % (cat1, cat2,
                                                  obsval, reference[pair][0]))


class TestProbabilityReEstimation(TestProbabilityEstimation):
    def _config(self):
        self.reference_file = REFERENCE_REESTIMATE_PROBS
        self.baseline = _load_baseline()
        self.model = _load_catmap(self.baseline.get_segmentations())
        self.retagged = []

        io = morfessor.MorfessorIO(encoding='latin-1')
        segmentations = io.read_segmentation_file(
            REFERENCE_BASELINE_SEGMENTATION)


class TestBaselineSegmentation(unittest.TestCase):
    def setUp(self):
        self.baseline = _load_baseline()
        self.model = _load_catmap(self.baseline.get_segmentations(),
                                  no_emissions=True)

        io = morfessor.MorfessorIO(encoding='latin-1')
        line_re = re.compile(r'^[0-9]* (.*)')
        separator_re = re.compile(r' \+ ')
        tag_re = re.compile(r'([^/]*)/(.*)')

        self.detagged = []
        self.references = []
        for line in io._read_text_file(REFERENCE_BASELINE_TAGGED):
            m = line_re.match(line)
            if not m:
                continue
            segments = separator_re.split(m.group(1))
            detagged_tmp = []
            ref_tmp = []
            for segment in segments:
                m = tag_re.match(segment)
                assert m, 'Could not parse "%s" in "%s"' % (segment, line)
                ref_tmp.append(catmap.CategorizedMorph(m.group(1),
                                                       m.group(2)))
                detagged_tmp.append(m.group(1))
            self.references.append(ref_tmp)
            self.detagged.append(detagged_tmp)

    def test_viterbitag(self):
        for (reference, tagless) in zip(self.references, self.detagged):
            observed = self.model.viterbi_tag(tagless)
            # causes UnicodeEncodeError, so sadly you don't get the details
            # u'"%s" does not match "%s"' % (observed, reference)
            msg = 'FIXME'
            for (r, o) in zip(reference, observed):
                self.assertEqual(r, o, msg=msg)


class TestModelConsistency(unittest.TestCase):
    dummy_segmentation = (
        (1, ('AA', 'BBBBB')),)

    simple_segmentation = (
        (1, ('AA', 'BBBBB')),
        (1, ('AA', 'CCCC')),
        (1, ('BBBBB', 'EE')),
        (2, ('DDDDD',)),
        (1, ('AADDDDDEE',)),
        (1, ('ZZBBBBB',)),
        (1, ('BBBBBZZ',)))

    def setUp(self):
        self.model = _load_catmap(TestModelConsistency.dummy_segmentation)

    def test_initial_state(self):
        """Tests that the initial state produced by loading a baseline
        segmentation is consistent."""
        self.initial_state_asserts()

    def test_presplit_state(self):
        """Tests that the initial parameter estimation from the baseline
        segmentation is consistent."""
        self.presplit()
        self.initial_state_asserts()

    def test_remove_readd(self):
        self.presplit()

        morph = 'AA'

        old_cost = self.model.get_cost()
        old_emissions = self.model._remove(morph)

        mid_cost = self.model.get_cost()

        self.model._readd(morph, old_emissions)
        new_cost = self.model.get_cost()

        msg = (u'_remove followed by _readd did not return to same cost ' +
               u'old: {}, new: {}'.format(old_cost, new_cost))
        self.assertAlmostEqual(old_cost, new_cost, places=4, msg=msg)

        # The model should have returned to initial state
        self.initial_state_asserts()

        # sanity check: costs should never be negative
        self.assertTrue(old_cost >= 0)
        self.assertTrue(mid_cost >= 0)
        self.assertTrue(new_cost >= 0)
        # sanity check: removing a morph but not replacing it with anything
        # else should always lower the cost.
        self.assertTrue(mid_cost < old_cost)

    def test_modify_coding_costs(self):
        self._modify_coding_costs('AA', 'BBBBB', 'PRE', 'STM', 2)
        self._modify_coding_costs('A', 'B', 'PRE', 'STM', 2)
        self._modify_coding_costs('A', 'A', 'PRE', 'STM', 2)

    def _modify_coding_costs(self, prefix, suffix,
                             prefix_cat, suffix_cat, count):
        self.presplit()

        old_cost = self.model.get_cost()
        self.model._modify_coding_costs((prefix, suffix),
                                        (prefix_cat, suffix_cat),
                                        count)

        mid_cost = self.model.get_cost()

        self.model._modify_coding_costs((prefix, suffix),
                                        (prefix_cat, suffix_cat),
                                        -count)
        new_cost = self.model.get_cost()

        msg = (u'_modify_coding_costs with opposite sign ' +
               u'did not return to same cost ' +
               u'old: {}, new: {} '.format(old_cost, new_cost))
        self.assertAlmostEqual(old_cost, new_cost, places=4, msg=msg)

        # The model should have returned to initial state
        self.initial_state_asserts()

        # sanity check: costs should never be negative
        self.assertTrue(old_cost >= 0)
        self.assertTrue(mid_cost >= 0)
        self.assertTrue(new_cost >= 0)
        # sanity check: adding submorphs without removing the parent
        # first should always increase the cost
        self.assertTrue(mid_cost > old_cost)

    def test_estimate_remove_temporaries(self):
        self.presplit()

        prefix = 'BB'
        suffix = 'BBB'

        tmp = (self.model._morph_usage.estimate_contexts(prefix + suffix,
                                                         (prefix, suffix)))
        self.model._morph_usage.remove_temporaries(tmp)

        self.initial_state_asserts()

    def presplit(self):
        self.model.viterbi_tag_corpus()
        self.model._reestimate_probabilities()

    def initial_state_asserts(self):
        marginalizer = catmap.Marginalizer()
        for morph in self.model._morph_usage.seen_morphs():
            # Un-normalied marginalzation
            # (scale by frequency and accumulate elementwise)
            marginalizer.add(self.model._morph_usage.count(morph),
                             self.model._morph_usage.condprobs(morph))
        category_totals = marginalizer.category_token_count

        self.assertAlmostEqual(sum(category_totals), 2.0, places=9)

        # morph usage
        self.assertEqual(self.model._morph_usage.seen_morphs(),
                         ['AA', 'BBBBB'])
        self.assertEqual(self.model._morph_usage._contexts,
                         {'AA': catmap.MorphContext(1, 1.0, 1.0),
                          'BBBBB': catmap.MorphContext(1, 1.0, 1.0)})

        # lexicon coding
        self.assertEqual(self.model._lexicon_coding.tokens,
                         len('AA') + len('BBBBB'))
        self.assertEqual(self.model._lexicon_coding.boundaries, 2)
        self.assertAlmostEqual(self.model._lexicon_coding.logfeaturesum,
            4.0 * catmap.universalprior(1.0), places=9)
        self.assertAlmostEqual(self.model._lexicon_coding.logtokensum,
            (2.0 * math.log(2.0)) + (5.0 * math.log(5.0)), places=9)

        # catmap coding
        self.general_consistency_asserts()

    def general_consistency_asserts(self):
        """ These values should be internally consistent at all times."""
        self.assertAlmostEqual(
            sum(self.model._catmap_coding._transition_counts.values()),
            sum(self.model._catmap_coding._cat_tagcount.values()),
            places=4)

        sum_transitions_from = collections.Counter()
        sum_transitions_to = collections.Counter()
        categories = self.model.get_categories(wb=True)
        forbidden = catmap.MorphUsageProperties.zero_transitions
        for prev_cat in categories:
            for next_cat in categories:
                count = self.model._catmap_coding._transition_counts[
                    (prev_cat, next_cat)]
                if count == 0:
                    continue
                if (prev_cat, next_cat) in forbidden:
                    # FIXME, make it an assert once the cause is fixed
                    print('Nonzero count for forbidden transition ' +
                        u'{} -> {}'.format(prev_cat, next_cat))
                sum_transitions_from[prev_cat] += count
                sum_transitions_to[next_cat] += count
        for cat in categories:
            # These hold, because for each incoming transition there is
            # exactly one outgoing transition (except for word boundary,
            # of which there are one of each in every word)
            msg = ('Transition counts were not symmetrical. ' +
                   u'category {}: {}, {}, {}'.format(cat,
                    sum_transitions_from[cat], sum_transitions_to[cat],
                    self.model._catmap_coding._cat_tagcount[cat]))

            self.assertEqual(sum_transitions_from[cat],
                             sum_transitions_to[cat],
                             msg)
            self.assertEqual(sum_transitions_to[cat],
                             self.model._catmap_coding._cat_tagcount[cat],
                             msg)


def _zexp(x):
    if x >= catmap.LOGPROB_ZERO:
        return 0.0
    return math.exp(-x)


def _exp_catprobs(probs):
    """Convenience function to convert a ByCategory object containing log
    probabilities into one with actual probabilities"""
    return catmap.ByCategory(*[_zexp(x) for x in probs])


if __name__ == '__main__':
    unittest.main()
