import math

class MajorityVote(object):
    def __init__(self, func, num_sets):
        self.func = func
        self.num_sets = num_sets
        self.limit = int(math.ceil(float(num_sets) / 2))

    def evaluate(self, point, prev_best_f):
        fs = []
        cues = [[]] * len(point)
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
        direction_cues = [median(d) for d in cues]
        return (median(fs), direction_cues)

    def _early_abort(decisions):
        successes = sum(decisions)
        chances = self.num_sets - len(decisions)
        return (successes + chances) < self.limit


class Intervals(object):
    N_INF = object()
    P_INF = object()

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
        self.b = self.x     # old inactive interval discarded
        self.x = x          # accepted point as midpoint
        self.rejected_prev = False

    def reject(self, x):
        self.a = x          # active interval shortened
        self.rejected_prev = True

    def use_direction_cue(self, cues):
        """Sets the direction (active part of the interval pair),
        using the direction cues."""
        # FIXME: if eval limit is increased, consider having a
        # treshold on the ratio of interval sizes for using the cues
        cumulative = 0.
        for (v, c) in zip(self.vector, cues):
            if c is not None:
                print(v, c, self.sign)
                cumulative += v * c * self.sign
        print('cumulative {}, rejected_prev {}, sign {}'.format(cumulative, self.rejected_prev, self.sign))
        if cumulative < 0:
            self._flip()
        elif cumulative == 0 and self.rejected_prev:
            self._flip()
    
    def _flip(self):
        (self.a, self.b) = (self.b, self.a)
        self.sign *= -1
        print('flipped. Now ({}, {}, {}) sign {}'.format(self.b, self.x, self.a, self.sign))


class LineSearchBisection(object):
    def __init__(self, func):
        self.func = func
        
    def search(self, initial, vector, initial_cues, prev_best_f, evals,
               bidir=False):
        intervals = Intervals(initial, vector, bidir)
        intervals.use_direction_cue(initial_cues)
        num_rejections = 0
        best = initial
        best_f = prev_best_f
        best_cues = initial_cues
        for _ in range(evals):
            (cursor_x, cursor) = intervals.step()
            (f, cursor_cues) = self.func(cursor, best_f)
            print('at {}. comparing {} > {}'.format(cursor, f, best_f))
            if f > best_f:
                intervals.accept(cursor_x)
                best_f = f
                best = cursor
                best_cues = cursor_cues
            else:
                intervals.reject(cursor_x)
                num_rejections += 1
            intervals.use_direction_cue(best_cues)

        return (best, best_f, best_cues, num_rejections)


def modified_powells(func, initial, max_iters, max_evals, scale):
    # Initial vectors are aligned to the axes
    vectors = []
    for (i, iv) in enumerate(initial):
        vectors.append([0] * len(initial))
        vectors[i][i] = float(scale)

    # + 1 is for the final search along the combination vector
    total_vector_loops = (max_iters * len(vectors)) + 1
    evals_per_vector = int(math.ceil(float(max_evals - 1) / total_vector_loops))
    print('loops: {}, evals per vec: {}'.format(total_vector_loops, evals_per_vector))
    evals_left = max_evals - evals_per_vector - 1
    point = initial
    (best_f, cues) = func(initial, None)
    best_vector = 0
    best_vector_increase = 0.
    line = LineSearchBisection(func)
    num_rejections = 0
    for iteration in range(max_iters):
        for (vec_num, vector) in enumerate(vectors):
            print('* doing vector {}: {}. evals_left {}'.format(vec_num, vector, evals_left))
            if evals_left <= 0:
                break
            evals = min(evals_left, evals_per_vector)
            (point, f, cues, rej) = line.search(
                point, vector, cues, best_f, evals, (iteration == 0))
            assert f >= best_f
            if f - best_f > best_vector_increase:
                best_vector = vec_num
                best_vector_increase = f - best_f
            best_f = f
            evals_left -= evals
            num_rejections += rej
        # remove best vector, replace with combo
        vectors.pop(best_vector)
        vectors.insert(0, [x - y for (x, y) in zip(point, initial)])
        print(vectors[0])
        if point == initial:
            print('No improvement from initial point')
            return initial
        if evals_left <= 0:
            break

    # finally search along the last combination vector
    print('* doing final vector: {}'.format(vectors[0]))
    (point, f, cues, rej) = line.search(
        point, vectors[0], cues, best_f, evals_per_vector)
        
    return point


def median(data):
    data = sorted([x for x in data if x is not None])
    length = len(data)
    if length == 0:
        return None
    if not length % 2:
        return (data[length / 2] + data[(length / 2) - 1]) / 2.
    return data[length / 2]
