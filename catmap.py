#!/usr/bin/env python
"""
Morfessor 2.0 Categories-MAP variant.
"""

__all__ = ['CatmapIO', 'CatmapModel']

__version__ = '2.0.0prealpha1'
__author__ = 'Stig-Arne Gronroos'
__author_email__ = "morfessor@cis.hut.fi"

import collections
import datetime
import logging
import math

import morfessor

_logger = logging.getLogger(__name__)
_logger.level = logging.DEBUG   # FIXME development convenience

LOGPROB_ZERO = 1000000


class WordBoundary(object):
    def __repr__(self):
        return '#'

    def __len__(self):
        return 0

WORD_BOUNDARY = WordBoundary()

# Grid node for viterbi algorithm
ViterbiNode = collections.namedtuple('ViterbiNode', ['cost', 'backpointer'])

WordAnalysis = collections.namedtuple('WordAnalysis', ['count', 'analysis'])

##################################
### Categorization-dependent code:
### to change the categories, only code in this section needs to be changed.


# A data structure with one value for each category.
# This also defines the set of possible categories
ByCategory = collections.namedtuple('ByCategory',
                                    ['PRE', 'STM', 'SUF', 'ZZZ'])


# The morph usage/context features used to calculate the probability of a
# morph belonging to a category.
MorphContext = collections.namedtuple('MorphContext',
                                      ['count', 'left_perplexity',
                                       'right_perplexity'])


class MorphContextBuilder(object):
    """Temporary structure used when calculating the MorphContexts."""
    def __init__(self):
        self.count = 0
        self.left = collections.Counter()
        self.right = collections.Counter()

    @property
    def left_perplexity(self):
        return MorphContextBuilder._perplexity(self.left)

    @property
    def right_perplexity(self):
        return MorphContextBuilder._perplexity(self.right)

    @staticmethod
    def _perplexity(contexts):
        entropy = 0
        total_tokens = float(sum(contexts.values()))
        for c in contexts:
            p = float(contexts[c]) / total_tokens
            entropy -= p * math.log(p)
        return math.exp(entropy)


class MorphUsageProperties(object):
    """This class describes how the prior probabilities are calculated
    from the usage of morphs.
    """

    # These transitions are impossible
    zero_transitions = ((WORD_BOUNDARY, WORD_BOUNDARY),
                        ('PRE', WORD_BOUNDARY),
                        ('PRE', 'SUF'),
                        (WORD_BOUNDARY, 'SUF'))
    # These transitions are additionally not considered for splitting a morph
    invalid_split_transitions = (('SUF', 'PRE'),
                                 ('SUF', 'STM'),
                                 ('STM', 'PRE'))

    # Cache for memoized valid transitions
    _valid_split_transitions = None
    _valid_transitions = None

    def __init__(self, ppl_treshold=100, ppl_slope=None, length_treshold=3,
                 length_slope=2, use_word_tokens=True,
                 min_perplexity_length=4):
        """Initialize the model parameters describing morph usage.

        Arguments:
            ppl_treshold -- Treshold value for sigmoid used to calculate
                            probabilities from left and right perplexities.
            ppl_slope -- Slope value for sigmoid used to calculate
                         probabilities from left and right perplexities.
            length_treshold -- Treshold value for sigmoid used to calculate
                               probabilities from length of morph.
            length_slope -- Slope value for sigmoid used to calculate
                            probabilities from length of morph.
            use_word_tokens -- If true, perplexity is based on word tokens.
                               If false, perplexity is based on word types.
            min_perplexity_length -- Morphs shorter than this length are
                                     ignored when calculating perplexity.
        """

        self._ppl_treshold = float(ppl_treshold)
        self._length_treshold = float(length_treshold)
        self._length_slope = float(length_slope)
        self.use_word_tokens = bool(use_word_tokens)
        self._min_perplexity_length = int(min_perplexity_length)
        if ppl_slope is not None:
            self._ppl_slope = float(ppl_slope)
        else:
            self._ppl_slope = 10.0 / self._ppl_treshold

        # Counts of different contexts in which a morph occurs
        self._contexts = Sparse(default=MorphContext(0, 1.0, 1.0))
        self._context_builders = collections.defaultdict(MorphContextBuilder)

        # Cache for memoized feature-based conditional class probabilities
        self._condprob_cache = collections.defaultdict(int)
        self._marginalizer = None

    def clear(self):
        """Resets the context variables.
        Use before fully reprocessing a segmented corpus."""
        self._contexts.clear()
        self._context_builders.clear()
        self._condprob_cache.clear()
        self._marginalizer = None

    def add_to_context(self, morph, pcount, rcount, i, segments):
        """Collect information about the contexts in which the morph occurs"""
        # Previous morph.
        if i == 0:
            # Word boundaries are counted as separate contexts
            neighbour = WORD_BOUNDARY
        else:
            neighbour = segments[i - 1]
            # Contexts shorter than treshold don't affect perplexity
            if len(neighbour) < self._min_perplexity_length:
                neighbour = None
        if neighbour is not None:
            self._context_builders[morph].left[neighbour] += pcount

        # Next morph.
        if i == len(segments) - 1:
            neighbour = WORD_BOUNDARY
        else:
            neighbour = segments[i + 1]
            if len(neighbour) < self._min_perplexity_length:
                neighbour = None
        if neighbour is not None:
            self._context_builders[morph].right[neighbour] += pcount

        self._context_builders[morph].count += rcount

    def compress_contexts(self):
        """Calculate compact features from the context data collected into
        _context_builders. This is done to save memory."""
        for morph in self._context_builders:
            tmp = self._context_builders[morph]
            self._contexts[morph] = MorphContext(tmp.count,
                                                 tmp.left_perplexity,
                                                 tmp.right_perplexity)
        self._context_builders.clear()

    def condprobs(self, morph):
        """Calculate feature-based conditional probabilities P(Category|Morph)
        from the contexts in which the morphs occur.

        Arguments:
            morph -- A string representation of the morph type.
        """
        if morph not in self._condprob_cache:
            context = self._contexts[morph]

            prelike = sigmoid(context.right_perplexity, self._ppl_treshold,
                            self._ppl_slope)
            suflike = sigmoid(context.left_perplexity, self._ppl_treshold,
                            self._ppl_slope)
            stmlike = sigmoid(len(morph), self._length_treshold,
                            self._length_slope)

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

            self._condprob_cache[morph] = ByCategory(p_pre, p_stm, p_suf,
                                                     p_nonmorpheme)
        return self._condprob_cache[morph]

    @property
    def marginal_class_probs(self):
        """True distribution of class probabilities,
        calculated by marginalizing over the feature based conditional
        probabilities over all observed morphs.
        This will not give the same result as the observed count based
        calculation.
        """
        return self._get_marginalizer().normalized()

    @property
    def category_token_count(self):
        return self._get_marginalizer().category_token_count

    def _get_marginalizer(self):
        if self._marginalizer is None:
            self._marginalizer = Marginalizer()
            for morph in self.seen_morphs():
                self._marginalizer.add(self.count(morph),
                                       self.condprobs(morph))
        return self._marginalizer

    def feature_cost(self, morph):
        """The cost of encoding the necessary features along with a morph.

        The length in characters of the morph is also a feature, but it does
        not need to be encoded as it is available from the surface form.
        """
        context = self._contexts[morph]
        return (universalprior(context.right_perplexity) +
                universalprior(context.left_perplexity))

    def estimate_contexts(self, old_morphs, new_morphs):
        """Estimates context features for new unseen morphs.

        Arguments:
            old_morphs -- A sequence of morphs being replaced. The existing
                          context of these morphs can be used in the
                          estimation.
            new_morphs -- A sequence of morphs that replaces the old ones.
                          Any previously unseen morphs in this sequence
                          will get context features estimated from their
                          surface form and/or from the contexts of the
                          old morphs they replace.
        Returns:
            A list of temporary morph contexts that have been estimated.
            These should be removed by the caller if no longer necessary.
            The removal is done using MorphContext.remove_temporaries.
        """
        temporaries = []
        for (i, morph) in enumerate(new_morphs):
            if morph in self:
                # The morph already has real context: no need to estimate
                continue
            if i == 0:
                # Prefix inherits left perplexity of leftmost parent
                l_ppl = self._contexts[old_morphs[0]].left_perplexity
            else:
                # Otherwise assume that the morph doesn't appear in any
                # other contexts, which gives perplexity 1.0
                l_ppl = 1.0
            if i == len(new_morphs) - 1:
                r_ppl = self._contexts[old_morphs[-1]].right_perplexity
            else:
                r_ppl = 1.0
            count = 0   # estimating does not add instances of the morph
            self._contexts[morph] = MorphContext(count, l_ppl, r_ppl)
            temporaries.append(morph)
        return temporaries

    def remove_temporaries(self, temporaries):
        """Remove estimated temporary morph contexts when no longer needed."""
        for morph in temporaries:
            if morph not in self:
                continue
            msg = u'{}: {}'.format(morph, self._contexts[morph].count)
            assert self._contexts[morph].count == 0, msg
            del self._contexts[morph]

    # The methods in this class below this line are helpers that will
    # probably not need to be modified if the categorization scheme changes

    def seen_morphs(self):
        """All morphs that have defined contexts."""
        return self._contexts.keys()

    def __contains__(self, morph):
        return morph in self._contexts

    def get(self, morph):
        """Returns the context features of a seen morph."""
        return self._contexts[morph]

    def count(self, morph):
        """The counts in the corpus of morphs with contexts."""
        if morph not in self._contexts:
            return 0
        return self._contexts[morph].count

    def set_count(self, morph, new_count):
        """Set the number of observed occurences of a morph."""
        self._contexts[morph] = self._contexts[morph]._replace(count=new_count)

    @classmethod
    def valid_split_transitions(cls):
        """Returns all category pairs that are considered valid ways
        to split a morph."""
        if cls._valid_split_transitions is None:
            cls._valid_split_transitions = []
            categories = list(ByCategory._fields)
            skiplist = list(MorphUsageProperties.zero_transitions)
            skiplist.extend(MorphUsageProperties.invalid_split_transitions)
            for prev_cat in categories:
                for next_cat in categories:
                    if (prev_cat, next_cat) in skiplist:
                        continue
                    cls._valid_split_transitions.append((prev_cat, next_cat))
            cls._valid_split_transitions = tuple(cls._valid_split_transitions)
        return cls._valid_split_transitions

    @classmethod
    def valid_transitions(cls):
        if cls._valid_transitions is None:
            cls._valid_transitions = []
            categories = CatmapModel.get_categories(wb=True)
            for cat1 in categories:
                for cat2 in categories:
                    if (cat1, cat2) in cls.zero_transitions:
                        continue
                    cls._valid_transitions.append((cat1, cat2))
            cls._valid_transitions = tuple(cls._valid_transitions)
        return cls._valid_transitions

### End of categorization-dependent code
########################################


class CatmapIO(morfessor.MorfessorIO):
    """Extends data file formats to include category tags."""

    def __init__(self, encoding=None, construction_separator=' + ',
                 comment_start='#', compound_separator='\s+',
                 category_separator='/'):
        morfessor.MorfessorIO.__init__(
            self, encoding=encoding,
            construction_separator=construction_separator,
            comment_start=comment_start, compound_separator=compound_separator,
            atom_separator=None)
        self.category_separator = category_separator

    def write_segmentation_file(self, file_name, segmentations, **kwargs):
        """Write segmentation file.

        File format (single line, wrapped only for pep8):
        <count> <construction1><cat_sep><category1><cons_sep>...
                <constructionN><cat_sep><categoryN>
        """

        _logger.info("Saving segmentations to '%s'..." % file_name)
        with self._open_text_file_write(file_name) as file_obj:
            d = datetime.datetime.now().replace(microsecond=0)
            file_obj.write('# Output from Morfessor Cat-MAP {}, {!s}\n'.format(
                __version__, d))
            for count, morphs in segmentations:
                s = self.construction_separator.join(
                    [u'{}{}{}'.format(m.morph, self.category_separator,
                                      m.category)
                     for m in morphs])
                file_obj.write(u'{} {}\n'.format(count, s))
        _logger.info("Done.")


class CatmapModel(object):
    """Morfessor Categories-MAP model class."""

    word_boundary = WORD_BOUNDARY

    def __init__(self, morph_usage, transition_cutoff=0.00000000001):
        """Initialize a new model instance.

        Arguments:
            morph_usage -- A MorphUsageProperties object describing how
                           the usage of a morph affects the category.
            transition_cutoff -- FIXME
        """

        self._morph_usage = morph_usage
        self._transition_cutoff = float(transition_cutoff)

        # The analyzed (segmented and tagged) corpus
        self.segmentations = []

        # Morph occurence backlinks
        # A dict of sets. Keys are morphs, set contents are indices to
        # self.segmentations for words in which the morph occurs
        self.morph_backlinks = collections.defaultdict(set)

        # Cost variables
        self._lexicon_coding = CatmapLexiconEncoding(morph_usage)
        # Catmap encoding also stores the HMM parameters
        self._catmap_coding = CatmapEncoding(morph_usage, self._lexicon_coding)

        # Log probabilities of single letters, for alphabetization cost
        # FIXME: no longer needed? assume identical distribution?
        # FIXME: lexicon logtokens? viterbi analysis of new words?
        self._log_letterprobs = dict()

    def load_baseline(self, segmentations, no_emissions=False):
        """Initialize the model using the segmentation produced by a morfessor
        baseline model.

        Arguments:
            segmentations -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
            no_emissions -- If True, no retagging/reestimation
                            will be performed.
                            Transition probabilities will remain as unigram
                            estimates, emission counts will be zero and
                            category total counts will be zero.
                            This means that the model is not completely
                            initialized: you will need to set the
                            emission counts and category totals.
                            (Default: False)
        """

        self.add_corpus_data(segmentations)
        self._calculate_usage_features()
        self._unigram_transition_probs()
        if no_emissions:
            return
        self.viterbi_tag_corpus()
        self._calculate_transition_counts()
        self._calculate_emission_counts()

    def add_corpus_data(self, segmentations):
        """Adds the given segmentations (with counts) to the corpus data"""
        i = len(self.segmentations)
        for segmentation in segmentations:
            segmentation = WordAnalysis(*segmentation)
            self.segmentations.append(segmentation)
            for morph in segmentation.analysis:
                self.morph_backlinks[morph].add(i)
            i += 1

    def train(self):
        """Perform Cat-MAP training on the model.
        The model must have been initialized by loading a baseline.
        """
        self.convergence_of_analysis(
            self._calculate_transition_counts,
            self.viterbi_tag_corpus,
            max_iterations=1)    # FIXME debug max
        self._calculate_emission_counts()
        self.convergence_of_cost(
            self._split_generator,
            max_iterations=2)    # FIXME debug max

    def _reestimate_probabilities(self):
        """Re-estimates model parameters from a segmented, tagged corpus.

        theta(t) = arg min { L( theta, Y(t), D ) }
        """
        self._calculate_usage_features()
        self._calculate_transition_counts()
        self._calculate_emission_counts()

    def _calculate_usage_features(self):
        """Recalculates the morph usage features (perplexities).
        """

        num_letter_tokens = collections.Counter()
        self._catmap_coding.boundaries = 0
        self._lexicon_coding.clear()
        self._morph_usage.clear()

        for rcount, segments in self.segmentations:
            self._catmap_coding.boundaries += rcount
            # Category tags are not needed for these calculations
            segments = CatmapModel.detag_word(segments)

            if self._morph_usage.use_word_tokens:
                pcount = rcount
            else:
                # pcount used for perplexity, rcount is real count
                pcount = 1
            num_letter_tokens[WORD_BOUNDARY] += pcount

            for (i, morph) in enumerate(segments):
                # Collect information about the contexts in which
                # the morphs occur.
                self._morph_usage.add_to_context(morph, pcount, rcount,
                                                 i, segments)

                for letter in morph:
                    num_letter_tokens[letter] += pcount
        self._morph_usage.compress_contexts()

        for morph in self._morph_usage.seen_morphs():
            self._lexicon_coding.add(morph)

        # Calculate letter log probabilities
        total_letter_tokens = sum(num_letter_tokens.values())
        log_tlt = math.log(total_letter_tokens)
        self._log_letterprobs = dict()
        for letter in num_letter_tokens:
            self._log_letterprobs[letter] = (log_tlt -
                math.log(num_letter_tokens[letter]))

    def _unigram_transition_probs(self):
        """Initial transition probabilities based on unigram distribution.

        Each tag is presumed to be succeeded by the expectation over all data
        of the number of prefixes, suffixes, stems, non-morphemes and word
        boundaries.
        """

        category_totals = self._morph_usage.category_token_count

        transitions = collections.Counter()
        nclass = {WORD_BOUNDARY: self.word_tokens}
        for (i, category) in enumerate(CatmapModel.get_categories()):
            nclass[category] = float(category_totals[i])

        num_tokens_tagged = 0.0
        valid_transitions = MorphUsageProperties.valid_transitions()

        for (cat1, cat2) in valid_transitions:
            # count all possible valid transitions
            num_tokens_tagged += nclass[cat2]
            transitions[(cat1, cat2)] = nclass[cat2]

        for pair in MorphUsageProperties.zero_transitions:
            transitions[pair] = 0.0

        normalization = (sum(nclass.values()) / num_tokens_tagged)
        for (prev_cat, next_cat) in transitions:
            self._catmap_coding.update_transition_count(
                prev_cat, next_cat,
                transitions[(prev_cat, next_cat)] * normalization)

    def _calculate_transition_counts(self):
        """Count the number of transitions of each type.
        Can be used to estimate transition probabilities from
        a category-tagged segmented corpus.
        """

        self._catmap_coding.clear_transition_counts()
        for rcount, segments in self.segmentations:
            # Only the categories matter, not the morphs themselves
            categories = [x.category for x in segments]
            # Include word boundaries
            categories.insert(0, WORD_BOUNDARY)
            categories.append(WORD_BOUNDARY)
            for (prev_cat, next_cat) in ngrams(categories, 2):
                pair = (prev_cat, next_cat)
                if pair in MorphUsageProperties.zero_transitions:
                    _logger.warning(u'Impossible transition ' +
                                    u'{!r} -> {!r}'.format(*pair))
                self._catmap_coding.update_transition_count(prev_cat,
                                                            next_cat,
                                                            rcount)

    def _calculate_emission_counts(self):
        """Recalculates the emission counts from a retagged segmentation."""
        self._catmap_coding.clear_emission_counts()
        for (count, analysis) in self.segmentations:
            for morph in analysis:
                self._catmap_coding.update_emission_count(morph.category,
                                                          morph.morph,
                                                          count)

    def _transformation_epoch(self, transformation_generator):
        for experiment in transformation_generator:
            (transform_group, targets, temporaries) = experiment
            # Cost of doing nothing
            best = (self.get_cost(), None, set())

            matched_targets = set()
            for transform in transform_group:
                for target in targets:
                    old_analysis = self.segmentations[target]
                    new_analysis = transform.apply(old_analysis, self)

                    if old_analysis == new_analysis:
                        continue
                    matched_targets.add(target)
                # All transforms in group must match the same words
                targets = matched_targets

                # Apply change to encoding
                self._update_counts(transform.change_counts, 1)
                cost = self.get_cost()
                if cost < best[0]:
                    best = (cost, transform, matched_targets)
                # Revert change to encoding
                self._update_counts(transform.change_counts, -1)

            if best[1] is not None:
                self._update_counts(best[1].change_counts, 1)
                for target in best[2]:
                    new_analysis = best[1].apply(self.segmentations[target],
                                                 self, corpus_index=target)
                    self.segmentations[target] = new_analysis
                    # any morph used in the best segmentation
                    # is no longer temporary
                    temporaries.difference_update(
                        self.detag_word(new_analysis.analysis))
            self._morph_usage.remove_temporaries(temporaries)

    def _split_generator(self):
        # FIXME random shuffle or sort by length/frequency?
        epoch_morphs = tuple(self._morph_usage.seen_morphs())
        for morph in epoch_morphs:
            if len(morph) == 1:
                continue
            if self._morph_usage.count(morph) == 0:
                continue

            # Match the parent morph with any category
            rule = TransformationRule((CategorizedMorph(morph, None),))
            transforms = []
            # Apply to all words in which the morph occurs
            targets = self.morph_backlinks[morph]
            # Temporary estimated contexts
            temporaries = set()
            for splitloc in range(1, len(morph)):
                prefix = morph[:splitloc]
                suffix = morph[splitloc:]
                # Make sure that there are context features available
                # (real or estimated) for the submorphs
                tmp = (self._morph_usage.estimate_contexts(morph,
                                                           (prefix, suffix)))
                temporaries.update(tmp)
                transforms.append(
                    Transformation(rule,
                                   (CategorizedMorph(prefix, None),
                                    CategorizedMorph(suffix, None))))
            yield (transforms, targets, temporaries)

    def _split_morph(self, morph):
        if len(morph) == 1:
            return
        total_count = self._morph_usage.count(morph)
        if total_count == 0:
            return

        # Cost of leaving the morph un-split
        best = (self.get_cost(), 0, None)

        # Remove all instances of the morph
        old_emissions = self._remove(morph)
        msg = (u'count {} for "{}" did not match sum of emissions {}'.format(
            total_count, morph, sum(old_emissions)))
        assert total_count == sum(old_emissions), msg

        # Temporary estimated contexts
        temporaries = set()

        # Cost of each possible split into two parts
        tmp_valid = MorphUsageProperties.valid_split_transitions()
        for splitloc in range(1, len(morph)):
            prefix = morph[:splitloc]
            suffix = morph[splitloc:]
            # Make sure that there are context features available
            # (real or estimated) for the submorphs
            tmp = (self._morph_usage.estimate_contexts(morph,
                                                       (prefix, suffix)))
            temporaries.update(tmp)
            # Consider tagging the newly formed morphs in all valid ways
            # Simplifying assumption: all instances are tagged the same way
            for (prefix_cat, suffix_cat) in tmp_valid:
                self._modify_coding_costs((prefix, suffix),
                                          (prefix_cat, suffix_cat),
                                          total_count)

                cost = self.get_cost()

                if cost < best[0]:
                    best = (cost, splitloc, (prefix_cat, suffix_cat))

                # Undo the changes
                self._modify_coding_costs((prefix, suffix),
                                          (prefix_cat, suffix_cat),
                                          -total_count)

        if best[1] == 0:
            # Best option was to do nothing: revert changes
            self._readd(morph, old_emissions)
            self._morph_usage.remove_temporaries(temporaries)
        else:
            splitloc = best[1]
            prefix_cat, suffix_cat = best[2]
            prefix = morph[:splitloc]
            suffix = morph[splitloc:]
            _logger.debug(u'Found a good split {}/{} + {}/{}'.format(
                prefix, prefix_cat, suffix, suffix_cat))
            # Re-apply the best split
            self._modify_coding_costs((prefix, suffix),
                                        (prefix_cat, suffix_cat),
                                        total_count)
            # New morphs used in split should no longer be removed
            if prefix in temporaries:
                temporaries.remove(prefix)
            if suffix in temporaries:
                temporaries.remove(suffix)
            self._morph_usage.remove_temporaries(temporaries)

            # FIXME recursion disabled
            #self._recursive_split(prefix)
            #if prefix != suffix:
            #    self._recursive_split(suffix)

    def _join_epoch(self):
        # FIXME random shuffle or sort by bigram frequency?
        # FIXME this is broken
        for (count, segments) in self.segmentations:
            segments = [WORD_BOUNDARY] + segments + [WORD_BOUNDARY]
            for quad in ngrams(segments, n=4):
                prev_morph, prefix, suffix, next_morph = quad
                self._join_bimorph(prev_morph, prefix, suffix, next_morph,
                                   count)

    def _join_bimorph(self, prev_morph, prefix, suffix, next_morph, count):
        # Cost of leaving the morphs un-joined
        best = (self.get_cost(), None)
        old_cost = best[0]

        old_categories = []
        if prev_morph == WORD_BOUNDARY:
            old_categories.append(WORD_BOUNDARY)
        else:
            old_categories.append(prev_morph.category)
        old_categories.extend((prefix.category, suffix.category))
        if next_morph == WORD_BOUNDARY:
            old_categories.append(WORD_BOUNDARY)
        else:
            old_categories.append(next_morph.category)
        morph = ''.join((prefix.morph, suffix.morph))
        old_morphs = (None, prefix.morph, suffix.morph, None)
        new_morphs = (None, morph, None)

        # Remove *these* instances of the child morphs
        # This is different from the old Cat-MAP
        self._modify_coding_costs(old_morphs, old_categories, -count)

        # Temporary estimated contexts
        temporaries = set()

        # Make sure that there are context features available
        # (real or estimated) for the new morph
        self._morph_usage.estimate_contexts((prefix, suffix), morph)

        # Cost of each tagging of the joined morph
        for new_cat in CatmapModel.get_categories():
            new_categories = (old_categories[0], new_cat, old_categories[-1])
            self._modify_coding_costs(new_morphs, new_categories, count)

            cost = self.get_cost()

            if cost < best[0]:
                best = (cost, new_cat)

            # Undo the changes
            self._modify_coding_costs(new_morphs, new_categories, -count)

        if best[1] is None:
            # Best option was to do nothing: revert changes
            self._modify_coding_costs(old_morphs, old_categories, count)
        else:
            # Re-apply with the best tag
            new_categories = (old_categories[0], best[1], old_categories[-1])
            self._modify_coding_costs(new_morphs, new_categories, count)
            _logger.debug(u'Found a good join {} + {} -> {}/{}'.format(
                prefix, suffix, morph, best[1]))

    def _modify_coding_costs(self, morphs, categories, diff_count):
        for (morph, category) in zip(morphs, categories):
            if morph is not None:
                self._catmap_coding.update_emission_count(category,
                                                          morph,
                                                          diff_count)
                self._modify_morph_count(morph, diff_count)
        for (prev_cat, next_cat) in ngrams(categories, 2):
            self._catmap_coding.update_transition_count(prev_cat,
                                                        next_cat,
                                                        diff_count)

    def _modify_morph_count(self, morph, diff_count):
        """Modifies the count of a morph in the lexicon.
        Does not affect transitions or emissions."""
        old_count = self._morph_usage.count(morph)
        new_count = old_count + diff_count
        self._morph_usage.set_count(morph, new_count)
        if old_count == 0 and new_count > 0:
            self._lexicon_coding.add(morph)
        elif old_count > 0 and new_count == 0:
            self._lexicon_coding.remove(morph)

    def _remove(self, morph):
        """Removes a morph completely from the lexicon.
        Transitions and emissions are also updated."""
        self._modify_morph_count(morph, -self._morph_usage.count(morph))
        old_emissions = self._catmap_coding.get_emission_counts(morph)
        self._catmap_coding.set_emission_counts(morph, _nt_zeros(ByCategory))
        return old_emissions

    def _readd(self, morph, emissions):
        """Readds a morph previously removed from the lexicon.
        Transitions and emissions are also restored."""
        count = sum(emissions)
        self._catmap_coding.set_emission_counts(morph, emissions)
        self._modify_morph_count(morph, count)

    def viterbi_tag(self, segments):
        """Tag a pre-segmented word using the learned model.

        Arguments:
            segments -- A list of morphs to tag.
                        Raises KeyError if morph is not present in the
                        training data.
                        For segmenting and tagging new words,
                        use viterbi_segment(compound).
        """

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = CatmapModel.get_categories(wb=True)
        wb = categories.index(WORD_BOUNDARY)

        # Grid consisting of
        # the lowest accumulated cost ending in each possible state.
        # and back pointers that indicate the best path.
        # Initialized to pseudo-zero for all states
        grid = [[ViterbiNode(LOGPROB_ZERO, None)] * len(categories)]
        # Except probability one that first state is a word boundary
        grid[0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = []
        best = []

        # Throw away old category information, if any
        segments = self.detag_word(segments)
        for (i, morph) in enumerate(segments):
            for next_cat in range(len(categories)):
                if next_cat == wb:
                    # Impossible to visit boundary in the middle of the
                    # sequence
                    best.append(ViterbiNode(LOGPROB_ZERO, None))
                    continue
                for prev_cat in range(len(categories)):
                    # Cost of selecting prev_cat as previous state
                    # if now at next_cat
                    cost.append(grid[i][prev_cat].cost +
                                self._catmap_coding.transit_emit_cost(
                                    categories[prev_cat],
                                    categories[next_cat], morph))
                best.append(ViterbiNode(*_minargmin(cost)))
                cost = []
            # Update grid to prepare for next iteration
            grid.append(best)
            best = []

        # Last transition must be to word boundary
        for prev_cat in range(len(categories)):
            pair = (categories[prev_cat], WORD_BOUNDARY)
            cost = (grid[-1][prev_cat].cost +
                    self._catmap_coding.log_transitionprob(*pair))
            best.append(cost)
        backtrace = ViterbiNode(*_minargmin(best))

        # Backtrace for the best category sequence
        result = [CategorizedMorph(segments[-1],
                  categories[backtrace.backpointer])]
        for i in range(len(segments) - 1, 0, -1):
            backtrace = grid[i + 1][backtrace.backpointer]
            result.insert(0, CategorizedMorph(segments[i - 1],
                categories[backtrace.backpointer]))
        return result

    def viterbi_tag_corpus(self):
        """(Re)tags the corpus segmentations using viterbi_tag"""
        num_changed_words = 0
        for (i, word) in enumerate(self.segmentations):
            self.segmentations[i] = WordAnalysis(word.count,
                self.viterbi_tag(word.analysis))
            if word != self.segmentations[i]:
                num_changed_words += 1
        return num_changed_words

    def viterbi_segment(self, segments):
        """Simultaneously segment and tag a word using the learned model.
        Can be used to segment unseen words

        Arguments:
            segments -- A list of morphs to tag.
        """

        if isinstance(segments, basestring):
            word = segments
        else:
            # Throw away old category information, if any
            segments = self.detag_word(segments)
            # Merge potential segments
            word = ''.join(segments)

        # This function uses internally indices of categories,
        # instead of names and the word boundary object,
        # to remove the need to look them up constantly.
        categories = CatmapModel.get_categories(wb=True)
        categories_nowb = [i for (i, c) in enumerate(categories)
                           if c != WORD_BOUNDARY]
        wb = categories.index(WORD_BOUNDARY)

        # Grid consisting of
        # the lowest accumulated cost ending in each possible state.
        # and back pointers that indicate the best path.
        # The grid is 3-dimensional:
        # grid [POSITION_IN_WORD]
        #      [MORPHLEN_OF_MORPH_ENDING_AT_POSITION - 1]
        #      [TAGINDEX_OF_MORPH_ENDING_AT_POSITION]
        # Initialized to pseudo-zero for all states
        zeros = [ViterbiNode(LOGPROB_ZERO, None)] * len(categories)
        grid = [[zeros]]
        # Except probability one that first state is a word boundary
        grid[0][0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = None
        best = ViterbiNode(LOGPROB_ZERO, None)

        for pos in range(1, len(word) + 1):
            grid.append([])
            for next_len in range(1, pos + 1):
                grid[pos].append(list(zeros))
                prev_pos = pos - next_len
                morph = word[prev_pos:pos]

                if morph not in self._morph_usage:
                    # The morph corresponding to this substring has not
                    # been encountered: zero probability for this solution
                    # FIXME: alphabetize instead?
                    grid[pos][next_len - 1] = zeros
                    continue

                for next_cat in categories_nowb:
                    best = ViterbiNode(LOGPROB_ZERO, None)
                    if prev_pos == 0:
                        # First morph in word
                        cost = self._catmap_coding.transit_emit_cost(
                            WORD_BOUNDARY, categories[next_cat], morph)
                        if cost <= best.cost:
                            best = ViterbiNode(cost, ((0, wb),
                                CategorizedMorph(morph, categories[next_cat])))
                    # implicit else: for-loop will be empty if prev_pos == 0
                    for prev_len in range(1, prev_pos + 1):
                        for prev_cat in categories_nowb:
                            cost = (
                                grid[prev_pos][prev_len - 1][prev_cat].cost +
                                self._catmap_coding.transit_emit_cost(
                                    categories[prev_cat],
                                    categories[next_cat],
                                    morph))
                            if cost <= best.cost:
                                prev_start = prev_pos - prev_len
                                prev_morph = word[prev_start:prev_pos]
                                best = ViterbiNode(cost, ((prev_len, prev_cat),
                                    CategorizedMorph(morph,
                                                     categories[next_cat])))
                    grid[pos][next_len - 1][next_cat] = best

        # Last transition must be to word boundary
        best = ViterbiNode(LOGPROB_ZERO, None)
        for prev_len in range(1, len(word) + 1):
            for prev_cat in categories_nowb:
                cost = (grid[-1][prev_len - 1][prev_cat].cost +
                        self._catmap_coding.log_transitionprob(
                            categories[prev_cat],
                            WORD_BOUNDARY))
                if cost <= best.cost:
                    prev_start = len(word) - prev_len
                    prev_morph = word[prev_start:]
                    best = ViterbiNode(cost, ((prev_len, prev_cat),
                        CategorizedMorph(WORD_BOUNDARY, WORD_BOUNDARY)))

        if best.cost == LOGPROB_ZERO:
            _logger.warning(
                u'No possible segmentation for word {}'.format(word))
            return word

        # Backtrace for the best morph-category sequence
        result = []
        backtrace = best
        pos = len(word)
        bt_len = backtrace.backpointer[0][0]
        bt_cat = backtrace.backpointer[0][1]
        while pos > 0:
            backtrace = grid[pos][bt_len - 1][bt_cat]
            bt_len = backtrace.backpointer[0][0]
            bt_cat = backtrace.backpointer[0][1]
            result.insert(0, backtrace.backpointer[1])
            pos -= len(backtrace.backpointer[1])
        return result

    def viterbi_resegment_corpus(self, corpus):
        """Convenience wrapper around viterbi_segment for a
        list of word strings or segmentations with attached counts.
        Segmented input can be with or without tags.
        """
        for (count, word) in corpus:
            if isinstance(word, basestring):
                word = (word,)
            yield (count, self.viterbi_segment(word))

    def convergence_of_cost(self, transform_generator, max_cost_difference=0,
                            max_iterations=15):
        """Iterates the specified training function until the model cost
        no longer improves enough or until maximum number of iterations
        is reached.

        On each iteration train_func is called without arguments.
        The data used to train the model must therefore be completely
        contained in the model itself. This can e.g. mean iterating over
        the morphs already stored in the lexicon.

        Arguments:
            transform_generator -- A function that returns a
                                   transform generator,
                                   which can be given as argument to
                                   _transformation_epoch
            max_cost_difference -- Stop iterating if cost reduction between
                                   iterations is below this limit.
            max_iterations -- Maximum number of iterations. Default 15.
        """

        previous_cost = self.get_cost()
        for iteration in range(max_iterations):
            _logger.info(u'Iteration {!r} number {}/{}.'.format(
                transform_generator.__name__, iteration + 1, max_iterations))

            # perform the optimization
            self._transformation_epoch(transform_generator())

            cost = self.get_cost()
            cost_diff = cost - previous_cost
            if -cost_diff <= max_cost_difference:
                _logger.info(u'Converged, with cost difference ' +
                    u'{} in final iteration.'.format(cost_diff))
                break
            else:
                _logger.info(u'Cost difference: {}'.format(cost_diff))
            previous_cost = cost

    def convergence_of_analysis(self, train_func, resegment_func,
                                max_differences=0,
                                max_cost_difference=-10000,
                                max_iterations=15):
        """Iterates the specified training function until the segmentations
        produced by the model no longer changes more than
        the specified treshold, until the model cost no longer improves
        enough or until maximum number of iterations is reached.

        On each iteration the current optimal analysis for the corpus is
        produced by calling resegment_func.
        This corresponds to:

        Y(t) = arg min { L( theta(t-1), Y, D ) }

        Then the train_func function is called, which corresponds to:

        theta(t) = arg min { L( theta, Y(t), D ) }

        Neither train_func nor resegment_func may require any arguments.

        Arguments:
            train_func -- A method of CatmapModel which causes some aspect
                          of the model to be trained.
            resegment_func -- A method of CatmapModel that resegments or
                              retags the segmentations, to produce the
                              results to compare. Should return the number
                              of changed words.
            max_differences -- Maximum number of words with changed
                               segmentation or category tags in the final
                               iteration. Default 0.
            max_cost_difference -- Stop iterating if cost reduction between
                                   iterations is below this limit.
            max_iterations -- Maximum number of iterations. Default 15.
        """

        previous_cost = self.get_cost()
        for iteration in range(max_iterations):
            _logger.info(u'Iteration {!r} number {}/{}.'.format(
                train_func.__name__, iteration + 1, max_iterations))

            # perform the optimization
            train_func()

            cost = self.get_cost()
            cost_diff = cost - previous_cost
            if -cost_diff <= max_cost_difference:
                _logger.info(u'Converged, with cost difference ' +
                    u'{} in final iteration.'.format(cost_diff))
                break

            # perform the reanalysis
            differences = resegment_func()

            if differences <= max_differences:
                _logger.info(u'Converged, with ' +
                    u'{} differences in final iteration.'.format(differences))
                break
            _logger.info(u'{} differences. Cost difference: {}'.format(
                differences, cost_diff))
            previous_cost = cost

    def get_cost(self):
        """Return current model encoding cost."""
        # FIXME: annotation coding cost for supervised
        return self._catmap_coding.get_cost() + self._lexicon_coding.get_cost()

    def _update_counts(self, change_counts, multiplier):
        for cmorph in change_counts.emissions:
            self._catmap_coding.update_emission_count(
                cmorph.category,
                cmorph.morph,
                change_counts.emissions[cmorph] * multiplier)

        for (prev_cat, next_cat) in change_counts.transitions:
            self._catmap_coding.update_transition_count(
                prev_cat, next_cat,
                change_counts.transitions[(prev_cat, next_cat)] * multiplier)

        if multiplier > 0:
            bl_rm = change_counts.backlinks_remove
            bl_add = change_counts.backlinks_add
        else:
            bl_rm = change_counts.backlinks_add
            bl_add = change_counts.backlinks_remove

        for morph in bl_rm:
            self.morph_backlinks[morph].difference_update(
                change_counts.backlinks_remove[morph])
        for morph in bl_add:
            self.morph_backlinks[morph].update(
                change_counts.backlinks_add[morph])

    @staticmethod
    def get_categories(wb=False):
        """The category tags supported by this model.
        Argumments:
            wb -- If True, the word boundary will be included. Default: False.
        """
        categories = list(ByCategory._fields)
        if wb:
            categories.append(WORD_BOUNDARY)
        return categories

    @staticmethod
    def _detag_morph(morph):
        if isinstance(morph, CategorizedMorph):
            return morph.morph
        return morph

    @staticmethod
    def detag_word(segments):
        return [CatmapModel._detag_morph(x) for x in segments]

    @staticmethod
    def detag_corpus(segmentations):
        """Removes category tags from a segmented corpus."""
        for rcount, segments in segmentations:
            yield ((rcount, [CatmapModel._detag_morph(x) for x in segments]))

    @property
    def _log_unknownletterprob(self):
        """The probability of an unknown letter is defined to be the squared
        probability of the rarest known letter"""
        return 2 * max(self._log_letterprobs.values())

    @property
    def word_tokens(self):
        return self._catmap_coding.boundaries


class CategorizedMorph(object):
    """Represents a morph with attached category information.
    These objects should be treated as immutable, even though
    it is not enforced by the code.
    """
    no_category = object()

    __slots__ = ['morph', 'category']

    def __init__(self, morph, category=None):
        self.morph = morph
        if category is not None:
            self.category = category
        else:
            self.category = CategorizedMorph.no_category

    def __repr__(self):
        if self.category == CategorizedMorph.no_category:
            return unicode(self.morph)
        return u'{}/{}'.format(self.morph, self.category)

    def __eq__(self, other):
        if not isinstance(other, CategorizedMorph):
            return False
        return (self.morph == other.morph and
                self.category == other.category)

    def __hash__(self):
        return hash(self.__repr__())

    def __len__(self):
        return len(self.morph)


class ChangeCounts(object):
    __slots__ = ['emissions', 'transitions',
                 'backlinks_remove', 'backlinks_add']

    def __init__(self, emissions=None, transitions=None):
        if emissions is None:
            self.emissions = collections.Counter()
        else:
            self.emissions = emissions
        if transitions is None:
            self.transitions = collections.Counter()
        else:
            self.transitions = transitions
        self.backlinks_remove = collections.defaultdict(set)
        self.backlinks_add = collections.defaultdict(set)

    def update(self, analysis, count, corpus_index=None):
        for cmorph in analysis:
            self.emissions[cmorph] += count
            if corpus_index is not None:
                if count < 0:
                    self.backlinks_remove[cmorph.morph].add(corpus_index)
                elif count > 0:
                    self.backlinks_add[cmorph.morph].add(corpus_index)
        wb_extended = list(analysis)
        wb_extended.insert(0, CategorizedMorph(WORD_BOUNDARY, WORD_BOUNDARY))
        wb_extended.append(CategorizedMorph(WORD_BOUNDARY, WORD_BOUNDARY))
        for (prefix, suffix) in ngrams(wb_extended, n=2):
            self.transitions[(prefix.category, suffix.category)] += count
        # Make sure that backlinks_remove and backlinks_add are disjoint
        # Removal followed by readding is the same as just adding
        for morph in self.backlinks_add:
            self.backlinks_remove[morph].difference_update(
                self.backlinks_add[morph])


class Transformation(object):
    __slots__ = ['cost', 'rule', 'result', 'change_counts']

    def __init__(self, rule, result):
        self.cost = 0
        self.rule = rule
        self.result = result
        self.change_counts = ChangeCounts()

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.rule, self.result)

    def apply(self, word, model, corpus_index=None):
        i = 0
        out = []
        matches = 0
        while i + len(self.rule) <= len(word.analysis):
            if self.rule.match(word.analysis, i):
                out.extend(self.result)
                i += len(self.rule)
                matches += 1
            else:
                out.append(word.analysis[i])
                i += 1
        while i < len(word.analysis):
            out.append(word.analysis[i])
            i += 1

        if matches > 0:
            # Morph counts need to be updated before tagging,
            # to get sensible emission probabilities for new morphs
            for morph in model.detag_word(word.analysis):
                model._modify_morph_count(morph, -word.count)
            for morph in model.detag_word(out):
                model._modify_morph_count(morph, word.count)
            # Only retag if the rule matched something
            out = model.viterbi_tag(out)

            if corpus_index is None:
                # Only revert morph count if not reapplying
                for morph in model.detag_word(word.analysis):
                    model._modify_morph_count(morph, word.count)
                for morph in model.detag_word(out):
                    model._modify_morph_count(morph, -word.count)

            self.change_counts.update(word.analysis, -word.count,
                                      corpus_index)
            self.change_counts.update(out, word.count, corpus_index)

        return WordAnalysis(word.count, out)


class TransformationRule(object):
    """A simple transformation rule that requires a pattern of category
    and/or morph matches. Don't care -values are marked by None.
    This simple rule does not account for context outside the part to
    be replaced.
    """

    def __init__(self, categorized_morphs):
        self._rule = categorized_morphs

    def __len__(self):
        return len(self._rule)

    def __repr__(self):
        return u'{}({})'.format(self.__class__.__name__, self._rule)

    def match(self, analysis, i):
        for (j, cmorph) in enumerate(analysis[i:(i + len(self))]):
            if self._rule[j].category != CategorizedMorph.no_category:
                # Rule requires category at this point to match
                if self._rule[j].category != cmorph.category:
                    return False
            if self._rule[j].morph is not None:
                # Rule requires morph at this point to match
                if self._rule[j].morph != cmorph.morph:
                    return False
        # No comparison failed
        return True


class Marginalizer(object):
    """An accumulator for marginalizing the class probabilities
    P(Category) from all the individual conditional probabilities
    P(Category|Morph) and observed morph probabilities P(Morph).

    First the unnormalized distribution is obtained by summing over
    #(Morph) * P(Category|Morph) over each morph, separately for each
    category. P(Category) is then obtained by normalizing the
    distribution.
    """

    def __init__(self):
        self._counts = [0.0] * len(ByCategory._fields)

    def add(self, rcount, condprobs):
        """Add the products #(Morph) * P(Category|Morph)
        for one observed morph."""
        for i, x in enumerate(condprobs):
            self._counts[i] += float(rcount) * float(x)

    def normalized(self):
        """Returns the marginal probabilities for all categories."""
        total = self.total_token_count
        return ByCategory(*[x / total for x in self._counts])

    @property
    def total_token_count(self):
        """Total number of tokens seen."""
        return sum(self._counts)

    @property
    def category_token_count(self):
        """Tokens seen per category."""
        return ByCategory(*self._counts)


class CatmapLexiconEncoding(morfessor.LexiconEncoding):
    """Extends LexiconEncoding to include the coding costs of the
    encoding cost of morph usage (context) features.
    """
    # FIXME qq, should this be renamed CatmapParameterEncoding?
    # i.e., is it P(lexicon) or P(theta) = P(lexicon) + P(grammar)?

    def __init__(self, morph_usage):
        super(CatmapLexiconEncoding, self).__init__()
        self._morph_usage = morph_usage
        self.logfeaturesum = 0.0

    def clear(self):
        """Resets the cost variables.
        Use before fully reprocessing a segmented corpus."""
        self.logtokensum = 0.0
        self.logfeaturesum = 0.0
        self.tokens = 0
        self.boundaries = 0
        self.atoms.clear()

    def add(self, morph):
        super(CatmapLexiconEncoding, self).add(morph)
        self.logfeaturesum += self._morph_usage.feature_cost(morph)

    def remove(self, morph):
        super(CatmapLexiconEncoding, self).remove(morph)
        self.logfeaturesum -= self._morph_usage.feature_cost(morph)

    def get_cost(self):
        assert self.boundaries >= 0
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return  ((n * math.log(n)
                  - self.boundaries * math.log(self.boundaries)
                  - self.logtokensum
                  + self.permutations_cost()
                  + self.logfeaturesum   # FIXME should it be weighted?
                 ) * self.weight
                 + self.frequency_distribution_cost())

    def get_codelength(self, morph):
        cost = super(CatmapLexiconEncoding, self).get_codelength(morph)
        cost += self._morph_usage.feature_cost(morph)
        return cost


class CatmapEncoding(morfessor.CorpusEncoding):
    """Class for calculating the encoding costs of the grammar and the
    corpus. Also stores the HMM parameters.

    tokens: the number of emissions observed.
    boundaries: the number of word tokens observed.
    """
    # can inherit without change: frequency_distribution_cost,

    def __init__(self, morph_usage, lexicon_encoding, weight=1.0):
        self._morph_usage = morph_usage
        super(CatmapEncoding, self).__init__(lexicon_encoding, weight)

        # Counts of emissions observed in the tagged corpus.
        # A dict of ByCategory objects indexed by morph. Counts occurences.
        self._emission_counts = Sparse(default=_nt_zeros(ByCategory))

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

        # Counts of observed category tags.
        # Single Counter object (ByCategory is unsuitable, need break also).
        self._cat_tagcount = collections.Counter()

        # Caches for transition and emission logprobs,
        # to avoid wasting effort recalculating.
        self._log_transitionprob_cache = dict()
        self._log_emissionprob_cache = dict()

    # Transition count methods

    def get_transition_count(self, prev_cat, next_cat):
        return self._transition_counts[(prev_cat, next_cat)]

    def log_transitionprob(self, prev_cat, next_cat):
        pair = (prev_cat, next_cat)
        if pair not in self._log_transitionprob_cache:
            if self._cat_tagcount[prev_cat] == 0:
                self._log_transitionprob_cache[pair] = LOGPROB_ZERO
            else:
                self._log_transitionprob_cache[pair] = (
                    _zlog(self._transition_counts[(prev_cat, next_cat)]) -
                    _zlog(self._cat_tagcount[prev_cat]))
        return self._log_transitionprob_cache[pair]

    def update_transition_count(self, prev_cat, next_cat, diff_count):
        """Updates the number of observed transitions between
        categories.

        Arguments:
            prev_cat -- The name (not index) of the category
                        transitioned from.
            next_cat -- The name (not index) of the category
                        transitioned to.
            diff_count -- The change in the number of transitions.
        """

        diff_count = float(diff_count)
        msg = 'update_transition_count needs category names, not indices'
        assert not isinstance(prev_cat, int), msg
        assert not isinstance(next_cat, int), msg
        pair = (prev_cat, next_cat)

        self._transition_counts[pair] += diff_count
        self._cat_tagcount[prev_cat] += diff_count

        msg = 'subzero transition count for {}'.format(pair)
        assert self._transition_counts[pair] >= 0, msg
        assert self._cat_tagcount[prev_cat] >= 0

        # invalidate cache
        self._log_transitionprob_cache.clear()

    def clear_transition_counts(self):
        """Resets transition counts, costs and cache.
        Use before fully reprocessing a tagged segmented corpus."""
        self._transition_counts.clear()
        self._cat_tagcount.clear()
        self._log_transitionprob_cache.clear()

    # Emission count methods

    def get_emission_counts(self, morph):
        return self._emission_counts[morph]

    def log_emissionprob(self, category, morph):
        """-Log of posterior emission probability P(morph|category)"""
        pair = (category, morph)
        if pair not in self._log_emissionprob_cache:
            cat_index = CatmapModel.get_categories().index(category)
            # Not equal to what you get by:
            # _zlog(self._emission_counts[morph][cat_index]) +
            if self._cat_tagcount[category] == 0:
                self._log_emissionprob_cache[pair] = LOGPROB_ZERO
            else:
                self._log_emissionprob_cache[pair] = (
                    _zlog(self._morph_usage.count(morph)) +
                    _zlog(self._morph_usage.condprobs(morph)[cat_index]) -
                    _zlog(self._morph_usage.category_token_count[cat_index]))
        return self._log_emissionprob_cache[pair]

    def update_emission_count(self, category, morph, diff_count):
        """Updates the number of observed emissions of a single morph from a
        single category, and the logtokensum (which is category independent).

        Arguments:
            category -- name of category from which emission occurs.
            morph -- string representation of the morph.
            diff_count -- the change in the number of occurences.
        """
        cat_index = CatmapModel.get_categories().index(category)
        new_counts = self._emission_counts[morph]._replace(
            **{category: (self._emission_counts[morph][cat_index] +
                          diff_count)})
        self.set_emission_counts(morph, new_counts)

    def set_emission_counts(self, morph, new_counts):
        """Set the number of emissions of a morph from all categories
        simultaneously.

        Arguments:
            morph -- string representation of the morph.
            new_counts -- ByCategory object with new counts.
        """

        old_total = sum(self._emission_counts[morph])
        self._emission_counts[morph] = new_counts
        new_total = sum(new_counts)

        if old_total > 1:
            self.logtokensum -= old_total * math.log(old_total)
        if new_total > 1:
            self.logtokensum += new_total * math.log(new_total)

        # invalidate cache
        self._log_emissionprob_cache.clear()

    def clear_emission_counts(self):
        """Resets emission counts and costs.
        Use before fully reprocessing a tagged segmented corpus."""
        self.tokens = 0
        self.logtokensum = 0.0
        self._emission_counts.clear()

    # General methods

    def transit_emit_cost(self, prev_cat, next_cat, morph):
        """Cost of transitioning from prev_cat to next_cat and emitting
        the morph."""
        if (prev_cat, next_cat) in MorphUsageProperties.zero_transitions:
            return LOGPROB_ZERO
        return (self.log_transitionprob(prev_cat, next_cat) +
                self.log_emissionprob(next_cat, morph))

    def update_count(self, construction, old_count, new_count):
        raise Exception('Inherited method not appropriate for CatmapEncoding')

    def get_cost(self):
        """Override for the Encoding get_cost function.

        This is P( D_W | theta, Y ) + ???
        """     # FIXME qq does catmap cost replace or add to baseline cost?
        if self.boundaries == 0:
            return 0.0

        # This can't be accumulated in log sum form, due to different total
        # number of transitions
        categories = CatmapModel.get_categories(wb=True)
        logtransitionsum = 0.0
        logcategorysum = 0.0
        total = 0.0
        # FIXME: this can be optimized when getting rid of the assertions
        sum_transitions_from = collections.Counter()
        sum_transitions_to = collections.Counter()
        forbidden = MorphUsageProperties.zero_transitions
        for prev_cat in categories:
            for next_cat in categories:
                if (prev_cat, next_cat) in forbidden:
                    continue
                count = self._transition_counts[(prev_cat, next_cat)]
                if count == 0:
                    continue
                sum_transitions_from[prev_cat] += count
                sum_transitions_to[next_cat] += count
                total += count
                if count > 0:
                    logtransitionsum += math.log(count)
        for cat in categories:
            # These hold, because for each incoming transition there is
            # exactly one outgoing transition (except for word boundary,
            # of which there are one of each in every word)
            assert sum_transitions_from[cat] == sum_transitions_to[cat]
            assert sum_transitions_to[cat] == self._cat_tagcount[cat]

            if self._cat_tagcount[cat] > 0:
                logtransitionsum -= ((len(categories) - 1) *
                                    math.log(self._cat_tagcount[cat]))
        logcategorysum -= len(categories) * math.log(total)

        n = self.tokens + self.boundaries
        return  ((n * math.log(n)
                  - self.boundaries * math.log(self.boundaries)
                  - self.logtokensum
                  + logtransitionsum
                  + logcategorysum
                 ) * self.weight
                 #+ self.frequency_distribution_cost())
                )


class Sparse(dict):
    """A defaultdict-like data structure, which tries to remain as sparse
    as possible. If a value becomes equal to the default value, it (and the
    key associated with it) are transparently removed.

    Only supports immutable values, e.g. namedtuples.
    """

    def __init__(self, *args, **kwargs):
        """Create a new Sparse datastructure.
        Keyword arguments:
            default -- Default value. Unlike defaultdict this should be a
                       prototype immutable, not a factory.
        """

        self._default = kwargs.pop('default')
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            return self._default
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        if value == self._default:
            if key in self:
                del self[key]
        else:
            dict.__setitem__(self, key, value)


def sigmoid(value, treshold, slope):
    return 1.0 / (1.0 + math.exp(-slope * (value - treshold)))


_LOG_C = math.log(2.865)
def universalprior(positive_number):
    """Compute the number of nats that are necessary for coding
    a positive integer according to Rissanen's universal prior.
    """

    return _LOG_C + math.log(positive_number)


def ngrams(sequence, n=2):
    """Returns all ngram tokens in an input sequence, for a specified n.
    E.g. ngrams(['A', 'B', 'A', 'B', 'D'], n=2) yields
    ('A', 'B'), ('B', 'A'), ('A', 'B'), ('B', 'D')
    """

    window = []
    for item in sequence:
        window.append(item)
        if len(window) > n:
            # trim back to size
            window = window[-n:]
        if len(window) == n:
            yield(tuple(window))


def _nt_zeros(constructor, zero=0.0):
    """Convenience function to return a namedtuple initialized to zeros,
    without needing to know the number of fields."""
    zeros = [zero] * len(constructor._fields)
    return constructor(*zeros)


def _minargmin(sequence):
    """Returns the minimum value and the first index at which it can be
    found in the input sequence."""
    best = (None, None)
    for (i, value) in enumerate(sequence):
        if best[0] is None or value < best[0]:
            best = (value, i)
    return best


def _zlog(x):
    """Logarithm which uses constant value for log(0) instead of -inf"""
    if x == 0:
        return LOGPROB_ZERO
    return -math.log(x)


def _log_catprobs(probs):
    """Convenience function to convert a ByCategory object containing actual
    probabilities into one with log probabilities"""

    return ByCategory(*[_zlog(x) for x in probs])
