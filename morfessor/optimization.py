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
            decisions.append(f > prev_best_f)
            for (i, d) in enumerate(cue):
                cues[i].append(d)
        direction_cues = [median(d) for d in cues]
        return (median(fs), direction_cues)

    def _early_abort(decisions):
        successes = sum(decisions)
        chances = self.num_sets - len(decisions)
        return (successes + chances) < self.limit

def line_search_bisection(func, initial, vector, prev_best_f, evals):
    (a, x, b) = (-1., 0., 1.)
    num_rejections = 0
    for _ in range(evals):

    return (point, f, num_rejections)


def modified_powells(func, initial, max_iters, max_evals):
    return point


def median(data):
    data = sorted([x for x in data if x is not None])
    length = len(data)
    if length == 0:
        return None
    if not length % 2:
        return (data[length / 2] + data[(length / 2) - 1]) / 2.
    return data[length / 2]
