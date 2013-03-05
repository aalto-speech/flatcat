#!/usr/bin/env python
"""
Morfessor 2.0 Categories-MAP variant.
"""

__all__ = ['CatmapIO', 'CatmapModel']

import collections
import logging
import math

import morfessor

_logger = logging.getLogger(__name__)
_logger.level = logging.DEBUG   # FIXME development convenience

LOGPROB_ZERO = 1000000


class WordBoundary:
    def __repr__(self):
        return '#'

WORD_BOUNDARY = WordBoundary()

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


class MorphContextBuilder:
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


class MorphUsageProperties:
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
        self._contexts = Sparse(MorphContext(0, 1.0, 1.0))
        self._context_builders = collections.defaultdict(MorphContextBuilder)

    def clear(self):
        self._contexts.clear()
        self._context_builders.clear()

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

    def condprob(self, morph):
        """Calculate conditional probabilities P(Category|Morph) from the
        contexts in which the morphs occur.

        Arguments:
            morph -- A string representation of the morph type.
        """
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

        return ByCategory(p_pre, p_stm, p_suf, p_nonmorpheme)

    def feature_cost(self, morph):
        """The cost of encoding the necessary features along with a morph.

        The length in characters of the morph is also a feature, but it does
        not need to be encoded as it is available from the surface form.
        """
        context = self._contexts[morph]
        return (universalprior(context.right_perplexity) +
                universalprior(context.left_perplexity))

    def estimate_contexts(self, old_morphs, new_morphs):
        temporaries = []
        count = self.count(old_morphs[0])
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
            self._contexts[morph] = MorphContext(count, l_ppl, r_ppl)
            temporaries.append(morph)
        return temporaries

    def remove_temporaries(self, temporaries):
        for morph in temporaries:
            if morph not in self:
                continue
            del self._contexts[morph]

    # The methods in this class below this line are helpers that will
    # probably not need to be modified if the categorization scheme changes

    def seen_morphs(self):
        """All morphs that have defined contexts."""
        return self._contexts.keys()

    def __contains__(self, morph):
        return morph in self._contexts

    def get(self, morph):
        return self._contexts[morph]

    def count(self, morph):
        """The counts in the corpus of morphs with contexts."""
        if morph not in self._contexts:
            return 0
        return self._contexts[morph].count

    @staticmethod
    def valid_split_transitions():
        """Yields all category pairs that are considered valid ways
        to split a morph."""
        categories = list(ByCategory._fields)
        skiplist = list(MorphUsageProperties.zero_transitions)
        skiplist.extend(MorphUsageProperties.invalid_split_transitions)
        for prev_cat in categories:
            for next_cat in categories:
                if (prev_cat, next_cat) in skiplist:
                    continue
                yield (prev_cat, next_cat)

### End of categorization-dependent code
########################################


class CatmapIO(morfessor.MorfessorIO):
    """Extends data file formats to include category tags."""
    # FIXME unimplemented

    def __init__(self, encoding=None, construction_separator=' + ',
                 comment_start='#', compound_separator='\s+',
                 atom_separator=None, category_separator='/'):
        morfessor.MorfessorIO.__init__(
            self, encoding=encoding,
            construction_separator=construction_separator,
            comment_start=comment_start, compound_separator=compound_separator,
            atom_separator=atom_separator)
        self.category_separator = category_separator


class CatmapModel:
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

        # Cost variables
        self._lexicon_coding = LexiconEncoding(morph_usage)
        # Catmap encoding also stores the HMM parameters
        self._catmap_coding = CatmapEncoding(self._lexicon_coding)

        # Priors for categories P(Category)
        # Single ByCategory object. Log-probabilities.
        self._log_catpriors = None

    def train(self, segmentations):
        """Perform Cat-MAP training on the model.

        Arguments:
            segmentations -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
        """
        # FIXME this is not good for big files. OTOH loading the baseline
        # will be done differently.
        segmentations = tuple(segmentations)
        self.load_baseline(segmentations)
        segmentations = self.until_convergence(
            self._calculate_transition_counts,
            self.viterbi_tag_segmentations,
            segmentations, max_iterations=1)    # FIXME max
        self._calculate_emission_counts(segmentations)
        segmentations = self.until_convergence(
            lambda x: self._split_epoch(self._recursive_split, x),
            self.viterbi_tag_segmentations,
            segmentations)

    def load_baseline(self, segmentations):
        """Initialize the model using the segmentation produced by a morfessor
        baseline model.

        Arguments:
            segmentations -- Segmentation of corpus using the baseline method.
                             Format: (count, (morph1, morph2, ...))
        """

        self._estimate_probabilities(segmentations)
        self._unigram_transition_probs(self._category_totals,
                                       self.word_tokens)

    def _estimate_probabilities(self, segmentations):
        """Estimates P(Category|Morph), P(Category) and P(Morph|Category).
        """

        num_letter_tokens = collections.Counter()
        self._catmap_coding.boundaries = 0
        self._lexicon_coding.clear()
        self._morph_usage.clear()

        # Conditional probabilities P(Category|Morph).
        # A dict of ByCategory objects indexed by morph. Actual probabilities.
        _condprobs = dict()

        for rcount, segments in segmentations:
            self._catmap_coding.boundaries += rcount
            # Category tags are not needed for these calculations
            segments = [CatmapModel._detag_morph(x) for x in segments]

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

        # Calculate conditional probabilities from the encountered contexts
        marginalizer = Marginalizer()
        for morph in self._morph_usage.seen_morphs():
            self._lexicon_coding.add(morph)
            _condprobs[morph] = self._morph_usage.condprob(morph)
            # Marginalize (scale by frequency and accumulate elementwise)
            marginalizer.add(self._morph_usage.count(morph),
                             _condprobs[morph])
        # Category priors from marginalization
        self._log_catpriors = _log_catprobs(marginalizer.normalized())

        # Calculate posterior emission probabilities
        self._category_totals = marginalizer.category_token_count
        for morph in self._morph_usage.seen_morphs():
            tmp = []
            for (i, total) in enumerate(self._category_totals):
                tmp.append(_condprobs[morph][i] *
                           self._morph_usage.count(morph) /
                           total)
            lep = _log_catprobs(ByCategory(*tmp))
            self._catmap_coding.set_log_emissionprobs(morph, lep)

        # Calculate letter log probabilities
        self._total_letter_tokens = sum(num_letter_tokens.values())
        log_tlt = math.log(self._total_letter_tokens)
        self._log_letterprobs = dict()
        for letter in num_letter_tokens:
            self._log_letterprobs[letter] = (log_tlt -
                math.log(num_letter_tokens[letter]))

    def _unigram_transition_probs(self, category_token_count,
                                  num_word_tokens):
        """Initial transition probabilities based on unigram distribution.

        Each tag is presumed to be succeeded by the expectation over all data
        of the number of prefixes, suffixes, stems, non-morphemes and word
        boundaries.

        Arguments:
            category_token_count -- A ByCategory with unnormalized
                                    morph token counts.
            num_word_tokens -- Total number of word tokens, for word boundary
                               probability.
        """

        transitions = collections.Counter()
        nclass = {WORD_BOUNDARY: num_word_tokens}
        for (i, category) in enumerate(CatmapModel.get_categories()):
            nclass[category] = float(category_token_count[i])

        num_tokens_tagged = collections.Counter()
        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in MorphUsageProperties.zero_transitions:
                    continue
                # count all possible valid transitions
                num_tokens_tagged[cat1] += nclass[cat2]

        for cat1 in nclass:
            for cat2 in nclass:
                if (cat1, cat2) in MorphUsageProperties.zero_transitions:
                    continue
                transitions[(cat1, cat2)] = nclass[cat2]

        for pair in MorphUsageProperties.zero_transitions:
            transitions[pair] = 0

        self._catmap_coding.set_transition_counts(transitions,
                                                  num_tokens_tagged)

    def _calculate_transition_counts(self, segmentations):
        """Count the number of transitions of each type.
        Can be used to estimate transition probabilities from
        a category-tagged segmented corpus.

        Arguments:
            segmentations -- Category-tagged segmented words.
                List of format:
                (count, (CategorizedMorph1, CategorizedMorph2, ...)), ...
        """
        self._catmap_coding.clear_transitions()
        for rcount, segments in segmentations:
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
                self._catmap_coding.update_transitions(prev_cat, next_cat,
                                                       rcount)

    def _split_epoch(self, func, segmentations):
        # FIXME random shuffle or sort by length?
        for morph in self._morph_usage.seen_morphs():
            func(morph)
        # FIXME should these be recalculated each iteration or not?
        self._calculate_transition_counts(segmentations)
        self._calculate_emission_counts(segmentations)

    def _recursive_split(self, morph):
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
        temporaries = []

        # Cost of each possible split into two parts
        for splitloc in range(1, len(morph)):
            prefix = morph[:splitloc]
            suffix = morph[splitloc:]
            # Make sure that there are context features available
            # (real or estimated) for the submorphs
            temporaries.extend(self._morph_usage.estimate_contexts(
                morph, (prefix, suffix)))
            # Consider tagging the newly formed morphs in all valid ways
            # Simplifying assumption: all instances are tagged the same way
            tmp_valid = MorphUsageProperties.valid_split_transitions()
            for (prefix_cat, suffix_cat) in tmp_valid:
                self._catmap_coding.update_emission(prefix_cat, prefix,
                                                   total_count)
                self._modify_morph_count(prefix, total_count)
                self._catmap_coding.update_emission(suffix_cat, suffix,
                                                   total_count)
                self._modify_morph_count(suffix, total_count)
                self._catmap_coding.update_transitions(prefix_cat,
                                                       suffix_cat,
                                                       total_count)

                cost = self.get_cost()
                if cost < best[0]:
                    best = (cost, splitloc, (prefix_cat, suffix_cat))

                # Undo the changes
                self._catmap_coding.update_emission(prefix_cat, prefix,
                                                   -total_count)
                self._modify_morph_count(prefix, -total_count)
                self._catmap_coding.update_emission(suffix_cat, suffix,
                                                   -total_count)
                self._modify_morph_count(suffix, -total_count)
                self._catmap_coding.update_transitions(prefix_cat,
                                                       suffix_cat,
                                                       -total_count)

        if best[1] == 0:
            self._readd(morph, old_emissions)
        else:
            splitloc = best[1]
            prefix_cat, suffix_cat = best[2]
            prefix = morph[:splitloc]
            suffix = morph[splitloc:]
            print(u'FIXME, found a good split {}/{} + {}/{}'.format(
                prefix, prefix_cat, suffix, suffix_cat))
            # FIXME actually perform the split, and then recurse
        self._morph_usage.remove_temporaries(temporaries)

    def _modify_morph_count(self, morph, dcount):
        """Modifies the count of a morph in the lexicon.
        Does not affect transitions or emissions."""
        old_count = self._morph_usage.count(morph)
        new_count = old_count + dcount
        if old_count == 0 and new_count > 0:
            self._lexicon_coding.add(morph)
        elif old_count > 0 and new_count == 0:
            self._lexicon_coding.remove(morph)

    def _remove(self, morph):
        """Removes a morph completely from the lexicon.
        Transitions and emissions are also updated."""
        self._modify_morph_count(morph, -self._morph_usage.count(morph))
        return self._catmap_coding.remove_emissions(morph)

    def _readd(self, morph, emissions):
        """Readds a morph previously removed from the lexicon.
        Transitions and emissions are also restored."""
        count = sum(emissions)
        self._catmap_coding.set_emissions(morph, emissions)
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
        ViterbiNode = collections.namedtuple('ViterbiNode',
                                             ['cost', 'backpointer'])
        grid = [[ViterbiNode(LOGPROB_ZERO, None)] * len(categories)]
        # Except probability one that first state is a word boundary
        grid[0][wb] = ViterbiNode(0, None)

        # Temporaries
        # Cumulative costs for each category at current time step
        cost = []
        best = []

        for (i, morph) in enumerate(segments):
            for next_cat in range(len(categories)):
                if next_cat == wb:
                    # Impossible to visit boundary in the middle of the
                    # sequence
                    best.append(ViterbiNode(LOGPROB_ZERO, None))
                    continue
                for prev_cat in range(len(categories)):
                    pair = (categories[prev_cat], categories[next_cat])
                    # Cost of selecting prev_cat as previous state
                    # if now at next_cat
                    cost.append(grid[i][prev_cat].cost +
                                self._catmap_coding.transit_emit_cost(
                                pair[0], pair[1], morph))
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

    def viterbi_tag_segmentations(self, segmentations):
        """Convenience wrapper around viterbi_tag for a list of segmentations
        with attached counts."""
        for (count, segmentation) in segmentations:
            yield (count, self.viterbi_tag(segmentation))

    def _calculate_emission_counts(self, segmentations):
        """Recalculates the emission counts from a retagged segmentation."""
        self._catmap_coding.clear_emissions()
        for (count, segmentation) in segmentations:
            for morph in segmentation:
                self._catmap_coding.update_emission(morph.category,
                                                    morph.morph,
                                                    count)

    def until_convergence(self, train_func, resegment_func, segmentations,
                          max_differences=0, max_cost_difference=-10000,
                          max_iterations=15):
        """Iterates the specified training function until the segmentations
        produced by the model for the given input no longer change more than
        the specified treshold, or until maximum number of iterations is
        reached.

        Arguments:
            train_func -- A method of CatmapModel that takes one argument:
                          segmentations, and which causes some aspect
                          of the model to be trained.
            resegment_func -- A method of CatmapModed that resegments or
                              retags the segmentations, to produce the
                              results to compare.  Takes one argument:
                              the (detagged) segmentations.
            segmentations -- list of (count, segmentation) pairs. Can be
                             either tagged or untagged.
            max_differences -- Maximum number of words with changed
                               segmentation or category tags in the final
                               iteration. Default 0.
            max_cost_difference -- Stop iterating if cost reduction between
                                   iterations is below this limit.
            max_iterations -- Maximum number of iterations. Default 15.
        Returns:
            Tagged segmentation produced by last iteration
        """

        detagged = tuple(CatmapModel._detag_segmentations(segmentations))
        current_segmentation = tuple(
            resegment_func(detagged))
        previous_cost = self.get_cost()
        for iteration in range(max_iterations):
            previous_segmentation = current_segmentation
            _logger.info(u'Iteration {!r} number {}/{}.'.format(
                train_func.__name__, iteration + 1, max_iterations))
            # perform the optimization
            train_func(previous_segmentation)

            cost = self.get_cost()
            cost_diff = cost - previous_cost
            if -cost_diff <= max_cost_difference:
                _logger.info(u'Converged, with cost difference ' +
                    u'{} in final iteration.'.format(cost_diff))
                break
            current_segmentation = tuple(resegment_func(detagged))
            differences = 0
            for (r, o) in zip(previous_segmentation, current_segmentation):
                if r != o:
                    differences += 1
            if differences <= max_differences:
                _logger.info(u'Converged, with ' +
                    u'{} differences in final iteration.'.format(differences))
                break
            _logger.info(u'{} differences. Cost difference: {}'.format(
                differences, cost_diff))
            previous_cost = cost
        return current_segmentation

    def log_emissionprobs(self, morph):
        return self._catmap_coding.log_emissionprobs(morph)

    def get_cost(self):
        """Return current model encoding cost."""
        # FIXME: annotation coding cost for supervised
        return self._catmap_coding.get_cost() + self._lexicon_coding.get_cost()

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
    def _detag_segmentations(segmentations):
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


class CategorizedMorph:
    """Represents a morph with attached category information."""
    no_category = object()

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
        return (self.morph == other.morph and
                self.category == other.category)


class Marginalizer():
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


class LexiconEncoding(morfessor.LexiconEncoding):
    def __init__(self, morph_usage):
        super(LexiconEncoding, self).__init__()
        self._morph_usage = morph_usage

    def clear(self):
        self.logtokensum = 0.0
        self.tokens = 0
        self.boundaries = 0

    def add(self, morph):
        super(LexiconEncoding, self).add(morph)
        self.logtokensum += self._morph_usage.feature_cost(morph)

    def remove(self, morph):
        super(LexiconEncoding, self).remove(morph)
        self.logtokensum -= self._morph_usage.feature_cost(morph)

    def get_codelength(self, construction):
        cost = super(LexiconEncoding, self).get_codelength(construction)
        cost += self._morph_usage.feature_cost(morph)
        return cost


class CatmapEncoding(morfessor.CorpusEncoding):
    """Class for calculating the encoding costs of the grammar and the
    corpus. Also stores the HMM parameters.

    tokens: the number of emissions observed.
    boundaries: the number of word tokens observed.
    """
    # can inherit without change: frequency_distribution_cost,

    def __init__(self, lexicon_encoding, weight=1.0):
        super(CatmapEncoding, self).__init__(lexicon_encoding, weight)

        # Posterior emission probabilities P(Morph|Category).
        # A dict of ByCategory objects indexed by morph. Log-probabilities.
        # No smoothing: default probability is zero
        self._log_emissionprobs = collections.defaultdict(_lp_zero_factory)

        # Counts of emissions observed in the tagged corpus.
        # Not equivalent to _log_emissionprobs (which is the MAP estimate,
        # while these would give the ML estimate)
        # A dict of ByCategory objects indexed by morph. Counts occurences.
        self._emission_counts = Sparse(_nt_zeros(ByCategory))

        # Counts of transitions between categories.
        # P(Category -> Category) can be calculated from these.
        # A dict of integers indexed by a tuple of categories.
        # Counts occurences.
        self._transition_counts = collections.Counter()

        # Counts of observed category tags.
        # Single Counter object (ByCategory is unsuitable, need break also).
        self._cat_tagcount = collections.Counter()

        # Cache for transition logprobs, to avoid wasting effort recalculating.
        self._log_transitionprob_cache = dict()

    def set_log_emissionprobs(self, morph, lep):
        self._log_emissionprobs[morph] = lep

    def set_transition_counts(self, transitions, num_tokens_tagged):
        self._transition_counts = transitions
        self._cat_tagcount = num_tokens_tagged

    def get_transition_count(self, prev_cat, next_cat):
        return self._transition_counts[(prev_cat, next_cat)]

    def clear_transitions(self):
        self._transition_counts.clear()
        self._cat_tagcount.clear()
        self._log_transitionprob_cache.clear()

    def update_transitions(self, prev_cat, next_cat, rcount):
        rcount = float(rcount)
        self._transition_counts[(prev_cat, next_cat)] += rcount
        self._cat_tagcount[prev_cat] += rcount
        # invalidate cache
        self._log_transitionprob_cache.clear()

    def log_transitionprob(self, prev_cat, next_cat):
        pair = (prev_cat, next_cat)
        if pair not in self._log_transitionprob_cache:
            self._log_transitionprob_cache[pair] = (
                _zlog(self._transition_counts[(prev_cat, next_cat)]) -
                _zlog(self._cat_tagcount[prev_cat]))
        return self._log_transitionprob_cache[pair]

    def log_emissionprobs(self, morph):
        return self._log_emissionprobs[morph]

    def transit_emit_cost(self, prev_cat, next_cat, morph):
        """Cost of transitioning from prev_cat to next_cat and emitting
        the morph."""
        next_i = CatmapModel.get_categories().index(next_cat)
        return (self.log_transitionprob(prev_cat, next_cat) +
                self._log_emissionprobs[morph][next_i])

    def clear_emissions(self):
        self.tokens = 0
        self.logtokensum = 0.0
        self._emission_counts.clear()

    def update_emission(self, category, morph, diff_count):
        """Updates the number of observed emissions of a single morph from a
        single category, and the cumulative cost of the corpus.
        """
        if not isinstance(category, int):
            category = CatmapModel.get_categories().index(category)
        old_count = self._emission_counts[morph][category]
        new_count = old_count + diff_count
        self._update_emission_cost(category, morph, diff_count)
        self._emission_counts[morph] = _set_nt_at_index(
            self._emission_counts[morph], category, new_count)

    def remove_emissions(self, morph):
        """Removes all emissions of a morph from all categories"""
        old_emissions = self._emission_counts[morph]
        for category in range(len(CatmapModel.get_categories())):
            diff_count = -self._emission_counts[morph][category]
            self._update_emission_cost(category, morph, diff_count)
        del self._emission_counts[morph]
        return old_emissions

    def set_emissions(self, morph, new_emissions):
        if morph in self._emission_counts:
            self.remove_emissions(morph)
        for category in range(len(CatmapModel.get_categories())):
            self._update_emission_cost(category, morph,
                                       new_emissions[category])
        self._emission_counts[morph] = new_emissions

    def _update_emission_cost(self, category, morph, diff_count):
        self.tokens += diff_count
        self.logtokensum += (diff_count *
                             self._log_emissionprobs[morph][category])

    def update_count(self, construction, old_count, new_count):
        raise Exception('Inherited method not appropriate for CatmapEncoding')

    def get_cost(self):
        """Override for the Encoding get_cost function."""
        if self.boundaries == 0:
            return 0.0

        # FIXME: could be accumulated instead of recalculated
        logtransitionsum = 0
        categories = CatmapModel.get_categories(wb=True)
        for next_cat in categories:
            for prev_cat in categories:
                pair = (prev_cat, next_cat)
                logtransitionsum += (self._transition_counts[pair] *
                                     self.log_transitionprob(*pair))

        n = self.tokens + self.boundaries
        return  ((n * math.log(n)
                  - self.boundaries * math.log(self.boundaries)
                  - self.logtokensum + logtransitionsum
                 ) * self.weight
                 + self.frequency_distribution_cost())


class Sparse(dict):
    """A defaultdict-like data structure, which tries to remain as sparse
    as possible. If a value becomes equal to the default value, it (and the
    key associated with it) are transparently removed.

    Only supports namedtuple values.
    """

    def __init__(self, default=None, *args, **kwargs):
        """Create a new Sparse datastructure.
        Arguments:
            default -- Default value. Unlike defaultdict this should be a
                       prototype namedtuple, not a factory.
        """

        self._default = default
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

    def set_at_index(self, key, index, value):
        if key not in self:
            nt = self._default
            self[key] = record
        else:
            nt = self[key]
        nt = _set_nt_at_index(nt, index, value)
        if nt == self._default:
            if key in self:
                del self[key]
        else:
            self[key] = nt


def sigmoid(value, treshold, slope):
    return 1.0 / (1.0 + math.exp(-slope * (value - treshold)))


_LOG_C = math.log(2.865)
def universalprior(positive_number):
    """Compute the number of nats that are necessary for coding
    a positive integer according to Rissanen's universal prior.
    """

    return _LOG_C + math.log(positive_number)


def ngrams(sequence, n=2):
    window = []
    for item in sequence:
        window.append(item)
        if len(window) > n:
            # trim back to size
            window = window[-n:]
        if len(window) == n:
            yield(tuple(window))


def _nt_zeros(constructor):
    """Convenience function to return a namedtuple initialized to zeros,
    without needing to know the number of fields."""
    zeros = [0.0] * len(constructor._fields)
    return constructor(*zeros)


def _set_nt_at_index(previous_namedtuple, index, new_value):
    """Convenience function to return a copy of a namedtuple with
    just one value altered.
    Arguments:
        previous_namedtuple -- namedtuple to use for all unchanged values.
        index -- index of value to change
        new_value -- new value to set at index
    """

    mutable = list(previous_namedtuple)
    mutable[index] = new_value
    return previous_namedtuple.__class__(*mutable)


def _minargmin(sequence):
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


def _lp_zero_factory():
    zeros = [LOGPROB_ZERO] * len(ByCategory._fields)
    return ByCategory(*zeros)
