import math

"""Optimization algorithms designed for very heavy objective functions.
A hard limit can be placed on the number of evaluations of
the objective function.
Direction cues can be used to control the direction of the line search.

The algorithms have been reimplemented here instead of using a library
like scipy, due to the abovementioned limitations caused by the
unusually heavy objective functions.
"""

class MajorityVote(object):
    """Wraps an objective function in a majority voting scheme.
    Arguments:
        func -- The objective function (A function taking
                the point as first argument, and the sequence number
                of the repetition as second argument.
                Should return the function value and the direction
                cues (a list with -1, 0 or 1 for each dimension).
        num_sets -- How many times to call the objective function.
    """
    def __init__(self, func, num_sets):
        self.func = func
        self.num_sets = num_sets
        self.limit = int(math.ceil(float(num_sets) / 2))

    def evaluate(self, point, prev_best_f):
        """Evaluates the objective function several times,
        merging the results using the median.
        If it becomes clear that the result will be worse
        than the previous best, the evaluation is aborted
        returning None.
        Arguments:
            point -- The point at which to evaluate,
            prev_best_f -- Best result encountered previously,
                           to enable early aborting.
        Returns:
            (median_f, direction_cues)
            or (None, None) if aborted early
        """
        fs = []
        cues = [[] for _ in range(len(point))]
        decisions = []
        for j in range(self.num_sets):
            if self._early_abort(decisions):
                return (None, None)
            (f, cue) = self.func(point, j)
            fs.append(f)
            if prev_best_f is None:
                decisions.append(True)
            else:
                decisions.append(f > prev_best_f)
            for (i, d) in enumerate(cue):
                cues[i].append(d)
        # Direction cues are -1, 0, 1 or None
        # allowing majority vote by median
        direction_cues = [median(d) for d in cues]
        return (median(fs), direction_cues)

    def _early_abort(self, decisions):
        """Returns True if it is no longer possible to
        beat the previous best result."""
        successes = sum(decisions)
        chances = self.num_sets - len(decisions)
        return (successes + chances) < self.limit


class Intervals(object):
    """A pair of intervals sharing the middle point.
    Used for finding the next point to evaluate in the line search.

    Arguments:
        initial -- Point to start the line search from.
        vector -- Vector describing the direction of the line search.
        bidir -- If True, the reverse direction is not considered
                 to be explored. If False, the point at -1 * vector
                 is assumed to be known, so the back interval is
                 restricted to start at -0.5 * vector.
    """
    N_INF = object()    # Negative infinity
    P_INF = object()    # Positive infinity

    def __init__(self, initial, vector, bidir):
        self.a = Intervals.P_INF
        self.x = 0.
        if bidir:
            self.b = Intervals.N_INF
        else:
            self.b = -0.5
        self.sign = 1
        self.initial = initial
        self.vector = vector
        self.rejected_prev = False

    def step(self):
        """Calculate the next point to evaluate
        Returns: (scale, point) where
            scale -- The multipler for the search vector.
            point -- The next point.
        """
        if self.a == Intervals.P_INF:
            scale = self.x + 1.0
        elif self.a == Intervals.N_INF:
            scale = self.x - 1.0
        else:
            scale = (self.x + self.a) / 2.

        point = []
        for (i, iv) in enumerate(self.initial):
            point.append(iv + (self.vector[i] * scale))
        return (scale, point)

    def accept(self, x):
        """Accept the step to x.
        Arguments:
            x -- A multipler for the search vector (not a point).
        """
        self.b = self.x     # old inactive interval discarded
        self.x = x          # accepted point as midpoint
        self.rejected_prev = False

    def reject(self, x):
        """Reject the step to x.
        Arguments:
            x -- A multipler for the search vector (not a point).
        """
        self.a = x          # active interval shortened
        self.rejected_prev = True

    def use_direction_cue(self, cues):
        """Sets the direction (active part of the interval pair),
        using the direction cues."""
        # FIXME: if eval limit is increased, consider having a
        # treshold on the ratio of interval sizes for using the cues,
        # To avoid wasting evals on fine combing one side of the best
        # point while leaving the other side very rough.
        cumulative = 0.
        for (v, c) in zip(self.vector, cues):
            if c is not None:
                cumulative += v * c * self.sign
        if cumulative < 0:
            # Direction cues, weighted by the contribution of their dimension
            # to the search direction, sum up to prefer the reverse side
            self._flip()
        elif cumulative == 0 and self.rejected_prev:
            # No dimensions returned any direction cues.
            # Falling back to alternating the search direction when a step
            # is rejected
            self._flip()
    
    def _flip(self):
        (self.a, self.b) = (self.b, self.a)
        self.sign *= -1


class LineSearchBisection(object):
    """Performs a line search.
    First an attempt to bracket the optimum by a fixed length step is made.
    If a bracket is found, the search continues through bisection,
    where direction cues are used to choose the next interval.
    If the optimum cannot be bracketed, the search continues outward
    in the direction of growth.

    Arguments:
        func -- The objective function (A function taking as first argument
                the point to be evaluated, and as second argument the
                best value found this far).
        cb_eval -- Called before each function evaluation.
        cb_acc -- Called when a step is accepted.
        cb_rej -- Called when a step is rejected.
    """
    def __init__(self, func, cb_eval=None, cb_acc=None, cb_rej=None):
        self.func = func
        self.cb_eval = cb_eval
        self.cb_acc = cb_acc
        self.cb_rej = cb_rej
        
    def search(self, initial, vector, initial_cues, prev_best_f, evals,
               bidir=False):
        """
        Performs a line search.
        Arguments:
            initial -- Point to start the line search from.
            vector -- Vector describing the direction of the line search.
            initial_cues -- Direction cues at the initial point.
            prev_best_f -- Best value encountered this far.
            evals -- How many function evaluations to perform.
            bidir -- If True, the reverse direction is not considered
                    to be explored. If False, the point at -1 * vector
                    is assumed to be known, so the back interval is
                    restricted to start at -0.5 * vector.
        Returns: (point, best_f, best_cues, step, num_rejections)
            point -- The search result point.
            best_f -- Function value at the result point.
            best_cues -- Direction cues at the result point.
            sq_step -- Squared length of the distance from initial
                       to result point.
            num_rejections -- How many steps were rejected during the search.

        """
        intervals = Intervals(initial, vector, bidir)
        intervals.use_direction_cue(initial_cues)
        num_rejections = 0
        best = initial
        best_f = prev_best_f
        best_cues = initial_cues
        for i in range(evals):
            (cursor_x, cursor) = intervals.step()
            if self.cb_eval is not None:
                self.cb_eval(i, evals, cursor)
            (f, cursor_cues) = self.func(cursor, best_f)
            if f > best_f:
                intervals.accept(cursor_x)
                best_f = f
                best = cursor
                best_cues = cursor_cues
                if self.cb_acc is not None:
                    self.cb_acc(i, evals, best, best_f, best_cues)
            else:
                intervals.reject(cursor_x)
                num_rejections += 1
                if self.cb_rej is not None:
                    self.cb_rej(i, evals, cursor, f, best, best_f, best_cues)
            # Possibly reverse the direction of the search
            intervals.use_direction_cue(best_cues)

        sq_step = sum(((i - b) ** 2) for (i, b) in zip(initial, best))
        return (best, best_f, best_cues, sq_step, num_rejections)


def modified_powells(func, initial, max_iters, evals_per_vector, scale=1.0,
                     cb_vec=None, cb_eval=None, cb_acc=None, cb_rej=None):
    """A variant of Powell's algorithm.
    Powell's algorithm finds the maximum of a multivariate function,
    without need for derivatives.

    This variant is designed for very heavy objective functions.
    The typically used Brent's method for line search has been replaced
    by a search that tries to achieve an optimal result in a fixed number
    of function evaluations. This change allows a hard limit to be placed
    on the number of evaluations of the objective function.
    Direction cues can be used to control the direction of the line search.
    A further change is that a final search along the vector given by the
    final displacement vector is performed, to allow for meaningful results
    when only performing one iteration. In the standard Powell's method
    two iterations would be the minimum number for any deviation from
    a taxicab search.

    If the function is one dimensional, falls back to a single line search.

    Arguments:
        func -- The objective function (A function taking as first argument
                the point to be evaluated, and as second argument the
                best value found this far).
        initial -- Point to start the optimization from.
        max_iters -- The maximum number of iterations to perform.
        evals_per_vector -- Objective function evaluations per line search.
                            The total number of evaluations is given by
                            1 +
                                    # for the initial point
                            (max_iters * dimensions * evals_per_vector) +
                                    # for the normal iterations
                            evals_per_vector
                                    # for the final line search
        scale -- A scaling factor for the initial steps (Default 1.0)
        cb_vec -- Called before performing a line search.
        cb_eval -- Called before each function evaluation.
        cb_acc -- Called when a step is accepted.
        cb_rej -- Called when a step is rejected.
    """
    # Fall back to single line search for 1D case
    if len(initial) == 1:
        return single_dimension(func,
                                initial,
                                max_iters * evals_per_vector,
                                scale,
                                cb_vec, cb_eval, cb_acc, cb_rej)

    # Initial vectors are aligned to the axes
    vectors = []
    for (i, iv) in enumerate(initial):
        vectors.append([0] * len(initial))
        vectors[i][i] = float(scale)
    bidir = [True] * len(vectors)

    point = initial
    if cb_eval is not None:
        cb_eval('initial', None, point)
    # Evaluating the function at the initial point
    (best_f, cues) = func(initial, None)
    best_vector = 0
    line = LineSearchBisection(func, cb_eval, cb_acc, cb_rej)
    num_rejections = 0
    num_evals = 0
    for iteration in range(max_iters):
        best_vector_sq_step = 0.
        for (vec_num, vector) in enumerate(vectors):
            if cb_vec is not None:
                cb_vec(iteration, max_iters, vec_num, len(vectors),
                       vector, point, best_f, cues)
            (point, f, cues, sq_step, rej) = line.search(point,
                                                         vector,
                                                         cues,
                                                         best_f,
                                                         evals_per_vector,
                                                         bidir[vec_num])
            # Only first search along combo vector is unidirectional,
            bidir[vec_num] = True
            assert f >= best_f
            if sq_step > best_vector_sq_step:
                best_vector = vec_num
                best_vector_sq_step = sq_step
            best_f = f
            num_rejections += rej
            num_evals += evals_per_vector
        if best_vector_sq_step == 0:
            print('No improvement in this iteration')
            return point

        # Remove vector that contributed the most during this iteration
        # FIXME: should be based on length, not f!
        vectors.pop(best_vector)
        bidir.pop(best_vector)
        # Replace with the vector formed by the displacement
        # from the initial point
        scale = float(num_evals - num_rejections) / num_evals
        vectors.insert(0, [(x - y) * scale for (x, y) in zip(point, initial)])
        # First search along combo vector is unidirectional, because at
        # point -1 along that line lies the initial point.
        bidir.insert(0, False)

        if point == initial:
            print('No improvement from initial point')
            return initial

    # Finally search along the last combination vector
    if cb_vec is not None:
        cb_vec('final', max_iters, 0, 1, vectors[0], point, best_f, cues)
    (point, f, cues, rej) = line.search(
        point, vectors[0], cues, best_f, evals_per_vector)
        
    return point


def single_dimension(func, initial, evals, scale=1.0,
                     cb_vec=None, cb_eval=None, cb_acc=None, cb_rej=None):
    """Optimize a single dimensional function using line search.

    Arguments:
        func -- The objective function (A function taking as first argument
                the point to be evaluated, and as second argument the
                best value found this far).
        initial -- Point to start the optimization from.
        evals -- Objective function evaluations.
        scale -- A scaling factor for the initial step (Default 1.0)
        cb_vec -- Called after initial point, before line search.
        cb_eval -- Called before each function evaluation.
        cb_acc -- Called when a step is accepted.
        cb_rej -- Called when a step is rejected.
    """
    vector = [float(scale)]

    point = initial
    if cb_eval is not None:
        cb_eval('initial', None, point)
    # Evaluating the function at the initial point
    (best_f, cues) = func(initial, None)
    if cb_acc is not None:
        cb_acc('initial', None, point, best_f, cues)
    line = LineSearchBisection(func, cb_eval, cb_acc, cb_rej)

    if cb_vec is not None:
        cb_vec(0, 1, 0, 1, vector, point, best_f, cues)
    (point, f, cues, sq_step, rej) = line.search(point,
                                                 vector,
                                                 cues,
                                                 best_f,
                                                 evals,
                                                 True)
    return point


def median(data):
    data = sorted([x for x in data if x is not None])
    length = len(data)
    if length == 0:
        return None
    if not length % 2:
        return (data[length / 2] + data[(length / 2) - 1]) / 2.
    return data[length / 2]
