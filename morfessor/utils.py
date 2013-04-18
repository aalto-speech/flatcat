import math
import sys


LOGPROB_ZERO = 1000000

# Progress bar for generators (length unknown):
# Print a dot for every GENERATOR_DOT_FREQ:th dot.
# Set to <= 0 to disable progress bar.
GENERATOR_DOT_FREQ = 500


class Sparse(dict):
    """A defaultdict-like data structure, which tries to remain as sparse
    as possible. If a value becomes equal to the default value, it (and the
    key associated with it) are transparently removed.

    Only supports immutable values, e.g. namedtuples.
    """

    def __init__(self, *pargs, **kwargs):
        """Create a new Sparse datastructure.
        Keyword arguments:
            default -- Default value. Unlike defaultdict this should be a
                       prototype immutable, not a factory.
        """

        self._default = kwargs.pop('default')
        dict.__init__(self, *pargs, **kwargs)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self._default

    def __setitem__(self, key, value):
        # attribute check is necessary for unpickling
        if '_default' in self and value == self._default:
            if key in self:
                del self[key]
        else:
            dict.__setitem__(self, key, value)


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


def minargmin(sequence):
    """Returns the minimum value and the first index at which it can be
    found in the input sequence."""
    best = (None, None)
    for (i, value) in enumerate(sequence):
        if best[0] is None or value < best[0]:
            best = (value, i)
    return best


def zlog(x):
    """Logarithm which uses constant value for log(0) instead of -inf"""
    assert x >= 0.0
    if x == 0:
        return LOGPROB_ZERO
    return -math.log(x)


def _generator_progress(generator):
    """Prints a progress bar for visualizing flow through a generator.
    The length of a generator is not known in advance, so the bar has
    no fixed length. GENERATOR_DOT_FREQ controls the frequency of dots.

    This function wraps the argument generator, returning a new generator.
    """

    if GENERATOR_DOT_FREQ <= 0:
        return generator

    def _progress_wrapper(generator):
        for (i, x) in enumerate(generator):
            if i % GENERATOR_DOT_FREQ == 0:
                sys.stderr.write('.')
                sys.stderr.flush()
            yield x
        sys.stderr.write('\n')

    return _progress_wrapper(generator)
