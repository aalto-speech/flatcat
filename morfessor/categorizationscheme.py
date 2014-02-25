"""A scheme for assigning categories to morphs.
To change the number or meaning of categories,
only this file should need to be modified.
"""
from __future__ import unicode_literals
import collections
import logging
import math

from . import utils

_logger = logging.getLogger(__name__)


class WordBoundary(object):
    """A special symbol for marking word boundaries.
    Using an object of this type allows arbitrary characters in the corpus,
    while using a string e.g. '#' instead causes that char to be reserved.
    """
    def __repr__(self):
        return '#'

    def __len__(self):
        return 0

    def __eq__(self, other):
        # Word boundaries from different runs should be equal
        if isinstance(other, WordBoundary):
            return True
        return False

    def __hash__(self):
        # This is called a lot. Using constant for minor optimization.
        #return hash(self.__class__.__name__)
        return 8364886854198508766


# Using a string is slightly faster.
# Change to WordBoundary if you want to e.g. support '#':s in the corpus
WORD_BOUNDARY = '#'  # WordBoundary()

##################################
### Categorization-dependent code:
### to change the categories, only code in this module
### should need to be changed.

# A data structure with one value for each category.
# This also defines the set of possible categories
ByCategory = collections.namedtuple('ByCategory',
                                    ['PRE', 'STM', 'SUF', 'ZZZ'])

DEFAULT_CATEGORY = 'STM'


# The morph usage/context features used to calculate the probability of a
# morph belonging to a category.
MorphContext = collections.namedtuple('MorphContext',
                                      ['count', 'left_perplexity',
                                       'right_perplexity'])

AnalysisAlternative = collections.namedtuple('AnalysisAlternative',
                                             ['analysis', 'penalty'])

# Context type flags, from which the context type is formed.
# (Binary flags in integer format)
CONTEXT_FLAG_INITIAL = 1
CONTEXT_FLAG_FINAL = 2
# The context type values
CONTEXT_TYPE_INTERNAL = 0
CONTEXT_TYPE_INITIAL = CONTEXT_TYPE_INTERNAL + CONTEXT_FLAG_INITIAL
CONTEXT_TYPE_FINAL = CONTEXT_TYPE_INTERNAL + CONTEXT_FLAG_FINAL
CONTEXT_TYPE_BOTH = (CONTEXT_TYPE_INTERNAL + CONTEXT_FLAG_INITIAL +
                     CONTEXT_FLAG_FINAL)

# Penalty for each non-morpheme, in heuristic postprocessing
# Must be smaller than LOGPROB_ZERO, to prevent impossible taggings from
# being generated.
NON_MORPHEME_PENALTY = 50


class HeuristicPostprocessor(object):
    def __init__(self, operations=None, max_join_stem_len=4):
        self.temporaries = set()
        self.max_join_stem_len = max_join_stem_len
        self.operations = []

    def remove_nonmorfemes(self, analysis, model):
        """Remove nonmorfemes from the analysis by joining or retagging
        morphs, using heuristics."""

        # Nothing to do if there are no nonmorphemes
        if all([m.category != 'ZZZ' for m in analysis]):
            return analysis

        if len(analysis) == 1:
            return (CategorizedMorph(analysis[0].morph, 'STM'),)

        # Sequencs of ZZZs should be joined
        analysis = self._join_sequences(analysis, model.forcesplit)

        # Resulting long ZZZs are retagged as stems
        self._long_to_stem(analysis, 4)

        # Might be done at this point
        if all(m.category != 'ZZZ' for m in analysis):
            return analysis

        # Retag parts of a multiple-suffix tail as SUF
        self._tail_suffixes(analysis)

        # If not: stronger measures are needed
        # Force join remaining
        analysis = self._force_join(analysis, model.forcesplit)

        # Retag with non-morphemes forbidden
        analysis = model.viterbi_tag(analysis, forbid_zzz=True)
        return analysis

    def _join_sequences(self, analysis, forcesplit):
        prev = None
        out = []
        for m in analysis:
            if (prev is None or
                    (m.category != 'ZZZ' or m.morph in forcesplit) or
                    (prev.morph in forcesplit) or
                    (prev.category != 'ZZZ')):
                if prev is not None:
                    out.append(prev)
                prev = m
                continue
            # prev is also a non-morpheme, and eligible for joining
            prev = CategorizedMorph(prev.morph + m.morph, 'ZZZ')
        if prev is not None:
            out.append(prev)
        return out

    def _long_to_stem(self, analysis, min_len):
        #In-place
        for m in analysis:
            if m.category == 'ZZZ' and len(m.morph) >= min_len:
                m.category = 'STM'

    def _tail_suffixes(self, analysis):
        #In-place
        for (i, m) in enumerate(analysis):
            if i == 0:
                continue
            if m.category == 'ZZZ' and analysis[i - 1].category == 'SUF':
                if all(tail.category in ('SUF', 'ZZZ')
                       for tail in analysis[(i + 1):]):
                    m.category = 'SUF'

    def _force_join(self, analysis, forcesplit):
        prev = None
        out = []
        if len(analysis) < 2:
            return analysis
        if (analysis[0].category == 'ZZZ' and
                analysis[0].morph not in forcesplit and
                analysis[1].morph not in forcesplit):
            analysis = self._join_at(analysis, 0)
        for m in analysis:
            if prev is None:
                prev = m
                continue
            if ((m.category != 'ZZZ' or m.morph in forcesplit) or
                    (prev.morph in forcesplit)):
                if prev is not None:
                    out.append(prev)
                prev = m
                continue
            # prev is eligible for joining
            prev = CategorizedMorph(prev.morph + m.morph, 'ZZZ')
        if prev is not None:
            out.append(prev)
        return out

    def _join_at(self, analysis, i):
        tag = analysis[i].category
        if analysis[i + 1].category != 'ZZZ':
            tag = analysis[i + 1].category
        if tag == 'ZZZ':
            tag = 'STM'
        morph = analysis[i].morph + analysis[i + 1].morph
        cmorph = CategorizedMorph(morph, tag)
        self.temporaries.add(cmorph)
        out = list(analysis[:i]) + [cmorph]
        if len(analysis) > (i + 2):
            out.extend(analysis[(i + 2):])
        return out


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

    # Adding these transitions removes the use of non-morphemes
    forbid_zzz = ((WORD_BOUNDARY, 'ZZZ'),
                  ('PRE', 'ZZZ'),
                  ('STM', 'ZZZ'),
                  ('SUF', 'ZZZ'))

    # Cache for memoized valid transitions
    _valid_transitions = None

    def __init__(self, ppl_threshold=100, ppl_slope=None, length_threshold=3,
                 length_slope=2, use_word_tokens=True,
                 min_perplexity_length=4):
        """Initialize the model parameters describing morph usage.

        Arguments:
            ppl_threshold -- threshold value for sigmoid used to calculate
                            probabilities from left and right perplexities.
            ppl_slope -- Slope value for sigmoid used to calculate
                         probabilities from left and right perplexities.
            length_threshold -- threshold value for sigmoid used to calculate
                               probabilities from length of morph.
            length_slope -- Slope value for sigmoid used to calculate
                            probabilities from length of morph.
            use_word_tokens -- If true, perplexity is based on word tokens.
                               If false, perplexity is based on word types.
            min_perplexity_length -- Morphs shorter than this length are
                                     ignored when calculating perplexity.
        """

        self._ppl_threshold = float(ppl_threshold)
        self._length_threshold = float(length_threshold)
        self._length_slope = float(length_slope)
        self.use_word_tokens = bool(use_word_tokens)
        self._min_perplexity_length = int(min_perplexity_length)
        if ppl_slope is not None:
            self._ppl_slope = float(ppl_slope)
        else:
            self._ppl_slope = 10.0 / self._ppl_threshold

        # Counts of different contexts in which a morph occurs
        self._contexts = utils.Sparse(default=MorphContext(0, 1.0, 1.0))
        self._context_builders = collections.defaultdict(MorphContextBuilder)

        self._contexts_per_iter = 50000  # FIXME customizable

        # Cache for memoized feature-based conditional class probabilities
        self._condprob_cache = collections.defaultdict(float)
        self._marginalizer = None

    def calculate_usage_features(self, seg_func):
        """Calculate the usage features of morphs in the corpus."""
        self._clear()
        while True:
            # If risk of running out of memory, perform calculations in
            # multiple loops over the data
            conserving_memory = False
            for rcount, segments in seg_func():

                if self.use_word_tokens:
                    pcount = rcount
                else:
                    # pcount used for perplexity, rcount is real count
                    pcount = 1

                for (i, morph) in enumerate(segments):
                    # Collect information about the contexts in which
                    # the morphs occur.
                    if self._add_to_context(morph, pcount, rcount,
                                            i, segments):
                        conserving_memory = True

            self._compress_contexts()
            utils.memlog('after compress')

            if not conserving_memory:
                break

    def _clear(self):
        """Resets the context variables.
        Use before fully reprocessing a segmented corpus."""
        self._contexts.clear()
        self._context_builders.clear()
        self._condprob_cache.clear()
        self._marginalizer = None

    def _add_to_context(self, morph, pcount, rcount, i, segments):
        """Collect information about the contexts in which the morph occurs"""
        if morph in self._contexts:
            return False
        if (len(self._context_builders) > self._contexts_per_iter and
                morph not in self._context_builders):
            return True

        # Previous morph.
        if i == 0:
            # Word boundaries are counted as separate contexts
            neighbour = WORD_BOUNDARY
        else:
            neighbour = segments[i - 1]
            # Contexts shorter than threshold don't affect perplexity
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
        return False

    def _compress_contexts(self):
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

            prelike = sigmoid(context.right_perplexity, self._ppl_threshold,
                            self._ppl_slope)
            suflike = sigmoid(context.left_perplexity, self._ppl_threshold,
                            self._ppl_slope)
            stmlike = sigmoid(len(morph), self._length_threshold,
                            self._length_slope)

            p_nonmorpheme = (1. - prelike) * (1. - suflike) * (1. - stmlike)
            # assert 0 <= p_nonmorpheme <= 1

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
        """Un-normalized distribution of class probabilities,
        the sum of which is the number of observed morphs.
        See marginal_class_probs for the normalized version.
        """
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

    @staticmethod
    def context_type(prev_morph, next_morph, prev_cat, next_cat):
        """Cluster certain types of context, to allow making context-dependant
        joining decisions."""
        # This categorization scheme ignores prev_morph, next_morph,
        # and only uses the categories
        ctype = CONTEXT_TYPE_INTERNAL
        if prev_cat == WORD_BOUNDARY or prev_cat == 'PRE':
            ctype += CONTEXT_FLAG_INITIAL
        if next_cat == WORD_BOUNDARY or next_cat == 'SUF':
            ctype += CONTEXT_FLAG_FINAL
        return ctype

    ### End of categorization-dependent code
    ########################################
    # But not the end of the class:
    # The methods in this class below this line are helpers that will
    # probably not need to be modified if the categorization scheme changes
    #
    def remove_temporaries(self, temporaries):
        """Remove estimated temporary morph contexts when no longer needed."""
        for morph in temporaries:
            if morph not in self:
                continue
            msg = '{}: {}'.format(morph, self._contexts[morph].count)
            assert self._contexts[morph].count == 0, msg
            del self._contexts[morph]
            if morph in self._condprob_cache:
                del self._condprob_cache[morph]

    def remove_zeros(self):
        """Remove context information for all morphs contexts with zero
        count. This can save a bit more memory than just removing estimated
        temporary contexts. Estimated context will be used for the removed
        morphs for the rest of the iteration."""
        remove_list = []
        for morph in self._contexts.keys():
            if self._contexts[morph].count == 0:
                remove_list.append(morph)
        for morph in remove_list:
            del self._contexts[morph]
            if morph in self._condprob_cache:
                del self._condprob_cache[morph]

    def seen_morphs(self):
        """All morphs that have defined contexts."""
        return [morph for morph in self._contexts.keys()
                if self._contexts[morph].count > 0]

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
        """Set the number of observed occurences of a morph.
        Also updates the true category distribution.
        """

        if self._marginalizer is not None and self.count(morph) > 0:
            self._marginalizer.add(-self.count(morph),
                                   self.condprobs(morph))
        self._contexts[morph] = self._contexts[morph]._replace(count=new_count)
        assert self.count(morph) >= 0
        if self._marginalizer is not None and self.count(morph) > 0:
            self._marginalizer.add(self.count(morph),
                                   self.condprobs(morph))

    @classmethod
    def valid_transitions(cls):
        """Returns (and caches) all valid transitions as pairs
        (from_category, to_category). Any transitions not included
        in the list are forbidden, and must have count 0 and probability 0.
        """
        if cls._valid_transitions is None:
            cls._valid_transitions = []
            categories = get_categories(wb=True)
            for cat1 in categories:
                for cat2 in categories:
                    if (cat1, cat2) in cls.zero_transitions:
                        continue
                    cls._valid_transitions.append((cat1, cat2))
            cls._valid_transitions = tuple(cls._valid_transitions)
        return cls._valid_transitions


class CategorizedMorph(object):
    """Represents a morph with attached category information.
    These objects should be treated as immutable, even though
    it is not enforced by the code.
    """

    __slots__ = ['morph', 'category']

    def __init__(self, morph, category=None):
        self.morph = morph
        self.category = category

    def __repr__(self):
        if self.category is None:
            return unicode(self.morph)
        return self.morph + '/' + self.category

    def __eq__(self, other):
        if not isinstance(other, CategorizedMorph):
            return False
        return (self.morph == other.morph and
                self.category == other.category)

    def __hash__(self):
        return hash((self.morph, self.category))

    def __len__(self):
        return len(self.morph)


def get_categories(wb=False):
    """The category tags supported by this model.
    Argumments:
        wb -- If True, the word boundary will be included. Default: False.
    """
    categories = list(ByCategory._fields)
    if wb:
        categories.append(WORD_BOUNDARY)
    return categories


def sigmoid(value, threshold, slope):
    return 1.0 / (1.0 + math.exp(-slope * (value - threshold)))


_LOG_C = math.log(2.865)


def universalprior(positive_number):
    """Compute the number of nats that are necessary for coding
    a positive integer according to Rissanen's universal prior.
    """

    return _LOG_C + math.log(positive_number)


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
