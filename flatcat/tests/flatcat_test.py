#!/usr/bin/env python
from __future__ import unicode_literals
"""
Tests for Morfessor 2.0 FlatCat variant.
"""

import collections
import logging
import math
import re
import unittest

import morfessor
from flatcat import flatcat
from flatcat import categorizationscheme as scheme
from flatcat.categorizationscheme import CategorizedMorph
from flatcat.utils import LOGPROB_ZERO


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


def _load_flatcat(baseline_seg, init='full'):
        """
        Arguments:
            baseline_seg -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
            init -- Controls how far the initialization with regard to
                    retagging/reestimation
                    will be performed.
                    Transition probabilities will remain as unigram
                    estimates, emission counts will be zero and
                    category total counts will be zero.
                    This means that the model is not completely
                    initialized: you will need to set the
                    emission counts and category totals.
                    (Default: full init)
        """

        m_usage = flatcat.MorphUsageProperties(ppl_threshold=10, ppl_slope=1,
                                              length_threshold=3,
                                              length_slope=2,
                                              type_perplexity=True)
        model = flatcat.FlatcatModel(m_usage)
        model.add_corpus_data(baseline_seg)
        if init == 'no_emissions':
            model._calculate_usage_features()
            model._unigram_transition_probs()
        elif init == 'first':
            model._calculate_usage_features()
            model._unigram_transition_probs()
            model.viterbi_tag_corpus()
            model._calculate_transition_counts()
            model._calculate_emission_counts()
        else:
            model.initialize_baseline()
            model.reestimate_probabilities()
        return model


class TestProbabilityEstimation(unittest.TestCase):
    def _config(self):
        """Overridden later to test re-estimation"""
        self.reference_file = REFERENCE_BASELINE_PROBS
        self.baseline = _load_baseline()
        self.model = _load_flatcat(self.baseline.get_segmentations(),
                                   init='no_emissions')

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
                self.posteriors[m.group(1)] = flatcat.ByCategory(
                    float(m.group(2)), float(m.group(3)),
                    float(m.group(4)), float(m.group(5)))
                continue

            m = transitions_re.match(line)
            if m:
                def _tr_wb(x):
                    if x == '#':
                        return flatcat.FlatcatModel.word_boundary
                    return x

                cats = tuple([_tr_wb(x) for x in (m.group(1), m.group(2))])
                self.transitions[cats] = (float(m.group(3)), int(m.group(4)))

        self.catpriors = flatcat.ByCategory(*(catpriors_tmp[x] for x in
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
        observed = self.model._morph_usage.marginal_class_probs

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
                        self.model._corpus_coding.log_emissionprob(category,
                                                                   morph))
                except KeyError:
                    raise KeyError('%s not in observed morphs' % (morph,))
                self.assertAlmostEqual(observed, reference[i], places=9,
                    msg=msg % (morph, category, observed, reference[i]))
                sumA += observed
                sumB += reference[i]

    def test_transitions(self):
        categories = self.model.get_categories(True)
        msg = 'P(%s -> %s), %s not almost equal to %s'
        reference = self.transitions
        for cat1 in categories:
            for cat2 in categories:
                pair = (cat1, cat2)
                obsval = _zexp(self.model._corpus_coding.log_transitionprob(
                                                                    *pair))
                self.assertAlmostEqual(obsval, reference[pair][0], places=9,
                                       msg=msg % (cat1, cat2,
                                                  obsval, reference[pair][0]))


class TestProbabilityReEstimation(TestProbabilityEstimation):
    def _config(self):
        self.reference_file = REFERENCE_REESTIMATE_PROBS
        self.baseline = _load_baseline()
        self.model = _load_flatcat(self.baseline.get_segmentations(),
                                   init='first')
        self.retagged = []

        io = morfessor.MorfessorIO(encoding='latin-1')
        segmentations = io.read_segmentation_file(
            REFERENCE_BASELINE_SEGMENTATION)


class TestBaselineSegmentation(unittest.TestCase):
    def setUp(self):
        self.baseline = _load_baseline()
        self.model = _load_flatcat(self.baseline.get_segmentations(),
                                   init='no_emissions')

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
                ref_tmp.append(flatcat.CategorizedMorph(m.group(1),
                                                       m.group(2)))
                detagged_tmp.append(m.group(1))
            self.references.append(ref_tmp)
            self.detagged.append(detagged_tmp)

    def test_viterbitag(self):
        for (reference, tagless) in zip(self.references, self.detagged):
            observed = self.model.viterbi_tag(tagless)
            # causes UnicodeEncodeError, so sadly you don't get the details
            # '"%s" does not match "%s"' % (observed, reference)
            msg = 'FIXME'
            for (r, o) in zip(reference, observed):
                self.assertEqual(r, o, msg=msg)


class TestOnline(unittest.TestCase):
    def setUp(self):
        self.baseline = _load_baseline()
        self.model = _load_flatcat(self.baseline.get_segmentations(),
                                   init='no_emissions')

    def test_focus(self):
        assert self.model.training_focus is None
        self.assertEqual(len(self.model.segmentations),
                         len(list(self.model._training_focus_filter())))

        self.model.training_focus = [5, 10, 2]
        self.assertEqual(len(list(self.model._training_focus_filter())), 3)
        self.assertEqual(next(self.model._training_focus_filter()),
                         self.model.segmentations[2])

        self.model.training_focus = [0, len(self.model.segmentations) - 1]
        self.assertEqual(len(list(self.model._training_focus_filter())), 2)
        self.assertEqual(next(self.model._training_focus_filter()),
                         self.model.segmentations[0])


class TestModelConsistency(unittest.TestCase):
    dummy_segmentation = (
        (1, ('AA', 'BBBBB')),)

    one_split_segmentation = (
        (500, ('AA', 'BBBBB')),
        (500, ('BBBBB', 'EE')),
        (500, ('AA', 'CCCC')),
        (500, ('CCCC', 'EE')),
        (500, ('AA', 'DDDD')),
        (500, ('DDDD', 'EE')),
        (500, ('AA', 'FFFF')),
        (500, ('FFFF', 'EE')),
        (500, ('AA', 'GGGG')),
        (500, ('GGGG', 'EE')),
        (2000, ('BBBBB',)),
        (2000, ('CCCC',)),
        (2000, ('DDDD',)),
        (2000, ('FFFF',)),
        (2000, ('GGGG',)),
        (2000, ('AAXXXXX',)),
        (500, ('SSSSS',)))

    dummy_annotation = {
        'ABCDEFG': ((CategorizedMorph('ABC', None),
                     CategorizedMorph('DEFG', None)),),
        'HIJKLMN': ((CategorizedMorph('H', None),
                     CategorizedMorph('IJKLMN', None)),
                    (CategorizedMorph('H', None),
                     CategorizedMorph('IJ', None),
                     CategorizedMorph('KLMN', None)))}

    one_split_annotation = {
        'AASSSSS': ((CategorizedMorph('AA', None),
                     CategorizedMorph('SSSSS', None)),)}

    def setUp(self):
        self.model = _load_flatcat(TestModelConsistency.dummy_segmentation)

    def test_initial_state(self):
        """Tests that the initial state produced by loading a baseline
        segmentation is consistent."""
        self._initial_state_asserts()

    def test_presplit_state(self):
        """Tests that the initial parameter estimation from the baseline
        segmentation is consistent."""
        self._presplit()
        self._initial_state_asserts()
        self._destructive_backlink_check()

    # Warning: this test is broken in the same way as
    # test_transforms_with_annotations, but it works for now
    def test_transforms(self):
        self.model.add_corpus_data(
            TestModelConsistency.one_split_segmentation)
        self._presplit()

        tmp = ((('AA', 'BBBBB'), ('AABBBBB',)),           # join
               (('AAXXXXX',), ('AA', 'XXXXX')),           # split
               (('BBBBB',), ('BB', 'B', 'BB')))           # silly

        def simple_transformation(old_analysis, new_analysis):
            return flatcat.Transformation(
                flatcat.TransformationRule(
                    [flatcat.CategorizedMorph(morph, None)
                        for morph in old_analysis]),
                [flatcat.CategorizedMorph(morph, None)
                    for morph in new_analysis])

        def apply_transformation(transformation):
            matched_targets, num_matches = self.model._find_in_corpus(
                transformation.rule, None)

            for morph in self.model.detag_word(transformation.rule):
                # Remove the old representation, but only from
                # morph counts (emissions and transitions updated later)
                self.model._modify_morph_count(morph, -num_matches)
            for morph in self.model.detag_word(transformation.result):
                # Add the new representation to morph counts
                self.model._modify_morph_count(morph, num_matches)

            for i in matched_targets:
                new_analysis = transformation.apply(
                    self.model.segmentations[i],
                    self.model, corpus_index=i)
                self.model.segmentations[i] = new_analysis
            self.model._update_counts(transformation.change_counts, 1)
            self.model._morph_usage.remove_zeros()

        for a, b in tmp:
            forward = simple_transformation(a, b)
            backward = simple_transformation(b, a)

            self._apply_revert(
                lambda: apply_transformation(forward),
                lambda: apply_transformation(backward),
                None)
        self._destructive_backlink_check()

    def test_update_counts(self):
        self._presplit()
        # manual change to join the one occurence of AA BBBBB
        cc = flatcat.ChangeCounts(
            emissions={flatcat.CategorizedMorph('AA', 'ZZZ'): -1,
                       flatcat.CategorizedMorph('BBBBB', 'STM'): -1,
                       flatcat.CategorizedMorph('AABBBBB', 'STM'): 1},
            transitions={(flatcat.WORD_BOUNDARY, 'ZZZ'): -1,
                         ('ZZZ', 'STM'): -1,
                         (flatcat.WORD_BOUNDARY, 'STM'): 1})

        def apply_update_counts():
            self.model._update_counts(cc, 1)

        def revert_update_counts():
            self.model._update_counts(cc, -1)

        self._apply_revert(apply_update_counts, revert_update_counts, None)

    def test_add_annotations(self):
        self.model.add_annotations(TestModelConsistency.dummy_annotation)
        self._presplit()
        self.model._update_annotation_choices()
        self.model._update_annotation_choices()
        self.model._update_annotation_choices()
        self._apply_revert(self.model._update_annotation_choices,
                           self.model._update_annotation_choices,
                           None)
        self._destructive_backlink_check()

# This test is inherently broken: it assumes that the tagging for a particular
# segmentation is unique, which is not true.
#     def test_transforms_with_annotations(self):
#         self.model.add_corpus_data(
#             TestModelConsistency.one_split_segmentation)
#         self.model.add_annotations(TestModelConsistency.one_split_annotation)
#         self._presplit()
#         self.model._update_annotation_choices()
#
#         tmp = ((('AA', 'BBBBB'), ('AABBBBB',)),           # join
#                (('AAXXXXX',), ('AA', 'XXXXX')),           # split
#                (('BBBBB',), ('BB', 'B', 'BB')))           # silly
#
#         def simple_transformation(old_analysis, new_analysis):
#             return flatcat.Transformation(
#                 flatcat.TransformationRule(
#                     [flatcat.CategorizedMorph(morph, None)
#                         for morph in old_analysis]),
#                 [flatcat.CategorizedMorph(morph, None)
#                     for morph in new_analysis])
#
#         def apply_transformation(transformation):
#             self.model.viterbi_tag_corpus()
#             self.model.reestimate_probabilities()
#             self.model.viterbi_tag_corpus()
#             self.model.reestimate_probabilities()
#
#             matched_targets, num_matches = self.model._find_in_corpus(
#                 transformation.rule, None)
#
#             for morph in self.model.detag_word(transformation.rule):
#                 # Remove the old representation, but only from
#                 # morph counts (emissions and transitions updated later)
#                 self.model._modify_morph_count(morph, -num_matches)
#             for morph in self.model.detag_word(transformation.result):
#                 # Add the new representation to morph counts
#                 self.model._modify_morph_count(morph, num_matches)
#
#             for i in matched_targets:
#                 new_analysis = transformation.apply(
#                     self.model.segmentations[i],
#                     self.model, corpus_index=i)
#                 self.model.segmentations[i] = new_analysis
#             self.model._update_counts(transformation.change_counts, 1)
#             if self.model._supervised:
#                 self.model._annot_coding.reset_contributions()
#             self.model._morph_usage.remove_zeros()
#
#             self.model.viterbi_tag_corpus()
#             self.model.reestimate_probabilities()
#             self.model.viterbi_tag_corpus()
#             self.model.reestimate_probabilities()
#
#         for a, b in tmp:
#             forward = simple_transformation(a, b)
#             backward = simple_transformation(b, a)
#
#             self._apply_revert(
#                 lambda: apply_transformation(forward),
#                 lambda: apply_transformation(backward),
#                 None)
#         self._destructive_backlink_check()

    def _apply_revert(self, apply_func, revert_func, is_remove):
        state_exact, state_approx = self._store_state()
        old_cost = self.model.get_cost()

        apply_func()

        mid_cost = self.model.get_cost()

        revert_func()

        new_cost = self.model.get_cost()

        msg = ('_apply_revert with {} and {} did not return to same cost. ' +
               'old: {}, new: {}')
        msg = msg.format(apply_func.__name__, revert_func.__name__,
                         old_cost, new_cost)
        self.assertAlmostEqual(old_cost, new_cost, places=4, msg=msg)

        # The model should have returned to initial state
        self._compare_to_stored_state(state_exact, state_approx)

        # sanity check: costs should never be negative
        self.assertTrue(old_cost >= 0)
        self.assertTrue(mid_cost >= 0)
        self.assertTrue(new_cost >= 0)
        # sanity check: adding submorphs without removing the parent
        # first should always increase the cost, while removing without
        # replacing should always lower the cost
        if is_remove is not None:
            if is_remove:
                self.assertTrue(mid_cost < old_cost)
            else:
                self.assertTrue(mid_cost > old_cost)

    def test_estimate_remove_temporaries(self):
        self._presplit()

        prefix = 'BB'
        suffix = 'BBB'

        tmp = (self.model._morph_usage.estimate_contexts(prefix + suffix,
                                                         (prefix, suffix)))
        self.model._morph_usage.remove_temporaries(tmp)

        self._initial_state_asserts()

    def _presplit(self):
        self.model.viterbi_tag_corpus()
        self.model.reestimate_probabilities()
        self.model.viterbi_tag_corpus()
        self.model.reestimate_probabilities()
        self.model.viterbi_tag_corpus()
        self.model.reestimate_probabilities()
        self.model.viterbi_tag_corpus()
        self.model.reestimate_probabilities()

    def _initial_state_asserts(self):
        category_totals = self.model._morph_usage.category_token_count

        self.assertAlmostEqual(sum(category_totals), 2.0, places=9)

        # morph usage
        self.assertEqual(sorted(self.model._morph_usage.seen_morphs()),
                         ['AA', 'BBBBB'])
        self.assertEqual(self.model._morph_usage._contexts,
                         {'AA': scheme.MorphContext(1, 1.0, 1.0),
                          'BBBBB': scheme.MorphContext(1, 1.0, 1.0)})

        # lexicon coding
        self.assertEqual(self.model._lexicon_coding.tokens,
                         len('AA') + len('BBBBB'))
        self.assertEqual(self.model._lexicon_coding.boundaries, 2)
        self.assertAlmostEqual(self.model._lexicon_coding.logfeaturesum,
            4.0 * scheme.universalprior(1.0), places=9)
        self.assertAlmostEqual(self.model._lexicon_coding.logtokensum,
            (2.0 * math.log(2.0)) + (5.0 * math.log(5.0)), places=9)

        # flatcat coding
        self._general_consistency_asserts()

    def _store_state(self):
        state_exact = {
            'seen_morphs': sorted(self.model._morph_usage.seen_morphs()),
            'contexts': dict(self.model._morph_usage._contexts),
            'lexicon_tokens': int(self.model._lexicon_coding.tokens),
            'segmentations': tuple(self.model.segmentations),
            'emission_counts': _remove_zeros(
                self.model._corpus_coding._emission_counts),
            'transition_counts': _remove_zeros(
                self.model._corpus_coding._transition_counts),
            'cat_tagcount': dict(self.model._corpus_coding._cat_tagcount)}
        state_approx = {
            'corpus_logtokensum': float(
                self.model._corpus_coding.logtokensum),
            'corpus_logcondprobsum': float(
                self.model._corpus_coding.logcondprobsum),
            'lexicon_logtokensum': float(
                self.model._lexicon_coding.logtokensum),
            'logfeaturesum': float(self.model._lexicon_coding.logfeaturesum)}
        for (i, tmp) in enumerate(
                self.model._morph_usage.category_token_count):
            state_approx['category_token_count_{}'.format(i)] = float(tmp)
        if self.model._annot_coding is not None:
            state_approx['annot_logemissionsum'] = float(
                self.model._annot_coding.logemissionsum)
            state_exact['annot_transition_cost'] = float(
                self.model._annot_coding.transition_cost())
        return (state_exact, state_approx)

    def _compare_to_stored_state(self, state_exact, state_approx):
        current_exact, current_approx = self._store_state()
        for key in state_exact:
            self.assertEqual(state_exact[key], current_exact[key],
                msg='Reverting did not return to same state: {} '.format(key) +
                    '({} vs {})'.format(state_exact[key], current_exact[key]))
        for key in state_approx:
            self.assertAlmostEqual(state_approx[key], current_approx[key],
                places=3,
                msg='Reverting did not return to same state: {} '.format(key) +
                    '({} vs {})'.format(state_approx[key], current_approx[key]))

    def _general_consistency_asserts(self):
        """ These values should be internally consistent at all times."""
        self.assertAlmostEqual(
            sum(self.model._corpus_coding._transition_counts.values()),
            sum(self.model._corpus_coding._cat_tagcount.values()),
            places=4)

        sum_transitions_from = collections.Counter()
        sum_transitions_to = collections.Counter()
        categories = self.model.get_categories(wb=True)
        forbidden = scheme.MorphUsageProperties.zero_transitions
        for prev_cat in categories:
            for next_cat in categories:
                count = self.model._corpus_coding._transition_counts[
                    (prev_cat, next_cat)]
                if count == 0:
                    continue
                if (prev_cat, next_cat) in forbidden:
                    # FIXME, make it an assert once the cause is fixed
                    print('Nonzero count for forbidden transition ' +
                        '{} -> {}'.format(prev_cat, next_cat))
                sum_transitions_from[prev_cat] += count
                sum_transitions_to[next_cat] += count
        for cat in categories:
            # These hold, because for each incoming transition there is
            # exactly one outgoing transition (except for word boundary,
            # of which there are one of each in every word)
            msg = ('Transition counts were not symmetrical. ' +
                   'category {}: {}, {}, {}'.format(cat,
                    sum_transitions_from[cat], sum_transitions_to[cat],
                    self.model._corpus_coding._cat_tagcount[cat]))

            self.assertEqual(sum_transitions_from[cat],
                             sum_transitions_to[cat],
                             msg)
            self.assertEqual(sum_transitions_to[cat],
                             self.model._corpus_coding._cat_tagcount[cat],
                             msg)

    def _destructive_backlink_check(self):
        """Destructively checks that morph backlinks cover the whole corpus.
        """

        for morph in self.model.morph_backlinks:
            for i in self.model.morph_backlinks[morph]:
                seg = self.model.segmentations[i]
                self.model.segmentations[i] = flatcat.WordAnalysis(
                    seg.count, [x for x in seg.analysis
                                if x.morph != morph])
        for seg in self.model.segmentations:
            self.assertEqual(len(seg.analysis), 0,
                             msg='missing backlinks: {}'.format(seg))


def _zexp(x):
    if x >= LOGPROB_ZERO:
        return 0.0
    return math.exp(-x)


def _exp_catprobs(probs):
    """Convenience function to convert a ByCategory object containing log
    probabilities into one with actual probabilities"""
    return scheme.ByCategory(*[_zexp(x) for x in probs])


def _remove_zeros(d):
    out = dict()
    for key in d:
        if d[key] == 0:
            continue
        try:
            if sum(abs(x) for x in d[key]) == 0:
                continue
        except TypeError:
            pass
        out[key] = d[key]
    return out

if __name__ == '__main__':
    unittest.main()
