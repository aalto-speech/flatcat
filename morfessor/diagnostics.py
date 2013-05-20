import collections
import logging
import time
import sys

_logger = logging.getLogger(__name__)

PY3 = sys.version_info.major == 3

NO_PLOTTING = True
if not PY3:     # my version of matplotlib doesn't support python 3
    try:
        from matplotlib import pyplot as plt
        NO_PLOTTING = False
    except ImportError:
        _logger.info('Unable to import matplotlib.pyplot: plotting disabled')

from . import baseline
from .exception import UnsupportedConfigurationError


class IterationStatistics(object):
    def __init__(self, title=None):
        self.iteration_numbers = []
        self.operation_numbers = []
        self.epoch_numbers = []

        self.costs = []
        self.corpuscosts = []
        self.lexiconcosts = []
        self.tag_counts = []
        self.morph_types = []
        self.morph_tokens = []
        self.durations = [0]
        self.morph_lengths = []
        self.changes = []

        self.gold_bpr = []
        self._reference = None

        self.max_morph_len = 0
        self.max_morph_len_count = 0
        self.t_prev = None
        self.word_tokens = 1.0
        self.categories = None

        if title is None:
            self.title = 'Iteration statistics {}'.format(
                time.strftime("%a, %d.%m.%Y %H:%M:%S"))
        else:
            self.title = title
        self.ops = None

    def set_names(self, model, training_operations):
        self.ops = training_operations
        self.categories = model.get_categories()

    def set_gold_standard(self, reference):
        self._reference = reference

    def callback(self, model, epoch_number=0):
        t_cur = time.time()

        self.iteration_numbers.append(model._iteration_number)
        self.operation_numbers.append(model._operation_number)
        self.epoch_numbers.append(epoch_number)

        self.costs.append(model.get_cost())
        self.corpuscosts.append(model._corpus_coding.get_cost() /
                                model._corpus_coding.weight)
        self.lexiconcosts.append(model._lexicon_coding.get_cost() /
                                 model._lexicon_coding.weight)
        tcounts = self._extract_tag_counts(model)
        self.tag_counts.append(tcounts)
        self.morph_types.append(len(model._morph_usage.seen_morphs()))
        self.morph_tokens.append(sum(tcounts))
        self.word_tokens = float(model.word_tokens)
        self.changes.append(len(model._changed_segmentations))

        if self._reference is not None:
            tmp = self._reference.items()
            wlist, annotations = zip(*tmp)
            segments = [model.viterbi_segment(w)[0] for w in wlist]
            self.gold_bpr.append(
                baseline.AnnotationsModelUpdate._bpr_evaluation(
                    [[x] for x in segments],
                    annotations))

        if self.t_prev is not None:
            self.durations.append(t_cur - self.t_prev)
        current_lengths = collections.Counter()
        for morph in model._morph_usage.seen_morphs():
            current_lengths[len(morph)] += 1
            if current_lengths[len(morph)] > self.max_morph_len_count:
                self.max_morph_len_count = current_lengths[len(morph)]
            if len(morph) > self.max_morph_len:
                self.max_morph_len = len(morph)
        self.morph_lengths.append(current_lengths)
        self.t_prev = t_cur

    def _extract_tag_counts(self, model):
        out = []
        counter = model._corpus_coding._cat_tagcount
        for cat in self.categories:
            out.append(counter[cat])
        return out


class IterationStatisticsPlotter(object):
    def __init__(self, stats):
        if NO_PLOTTING:
            raise UnsupportedConfigurationError(
                'Unable to import library matplotlib')
        self.stats = stats

    def all(self):
        plt.figure()
        self.costs()
        plt.figure()
        self.basecosts()
        plt.figure()
        self.tag_counts()
        plt.figure()
        self.avg_morphs()
        plt.figure()
        self.durations()
        plt.figure()
        self.types_and_tokens()
        plt.figure()
        self.morph_lengths()
        plt.figure()
        self.changes()
        if self.stats._reference is not None:
            plt.figure()
            self.gold_bpr()
        plt.show()

    def costs(self):
        plt.plot(self.stats.costs)
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Model cost')
        self._title()

    def basecosts(self):
        plt.plot(self.stats.corpuscosts, color='red')
        plt.plot(self.stats.lexiconcosts, color='green')
        plt.plot([sum(x) for x in zip(self.stats.corpuscosts,
                                      self.stats.lexiconcosts)],
                 color='black')
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Unweighted base cost (no penalties)')
        plt.legend(['Corpus', 'Lexicon (approx)', 'Sum'])
        self._title()

    def tag_counts(self):
        unzipped = zip(*self.stats.tag_counts)
        for (i, series) in enumerate(unzipped):
            plt.plot(series, color=plt.cm.jet(float(i) /
                float(len(self.stats.categories))))
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Category occurence count')
        self._title()
        if self.stats.categories is not None:
            plt.legend(self.stats.categories)

    def avg_morphs(self):
        normalized = [x / self.stats.word_tokens
                      for x in self.stats.morph_tokens]
        plt.plot(normalized)
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Avg number of morphs per word token')
        self._title()

    def types_and_tokens(self):
        plt.plot(self.stats.morph_tokens, color="red")
        plt.plot(self.stats.morph_types, color="blue")
        plt.legend(['Tokens', 'Types'])
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Count of morph tokens / types')
        self._title()

    def durations(self):
        by_iter = [0.0] * (max(self.stats.iteration_numbers) + 1)
        by_op = [0.0] * (max(self.stats.operation_numbers) + 1)
        by_epoch = [0.0] * (max(self.stats.epoch_numbers) + 1)

        for i in range(len(self.stats.iteration_numbers)):
            by_iter[self.stats.iteration_numbers[i]] += self.stats.durations[i]
            by_op[self.stats.operation_numbers[i]] += self.stats.durations[i]
            by_epoch[self.stats.epoch_numbers[i]] += self.stats.durations[i]

        plt.subplot(2, 2, 1)
        plt.plot(self.stats.durations)
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Epoch duration [s]')
        self._title()

        plt.subplot(2, 2, 2)
        plt.bar(range(len(by_iter)), by_iter)
        plt.ylabel('Total iteration duration [s]')
        xls = range(len(by_iter))
        xs = [x + 0.5 for x in xls]
        plt.xticks(xs, xls)

        plt.subplot(2, 2, 3)
        plt.bar(range(len(by_op)), by_op)
        plt.ylabel('Total operation duration [s]')
        xls = range(len(by_op))
        xs = [x + 0.5 for x in xls]
        if self.stats.ops is not None:
            xls = self.stats.ops
        plt.xticks(xs, xls)

        plt.subplot(2, 2, 4)
        plt.bar(range(len(by_epoch)), by_epoch)
        plt.ylabel('Total epoch duration [s]')
        xls = range(len(by_epoch))
        xs = [x + 0.5 for x in xls]
        plt.xticks(xs, xls)

    def morph_lengths(self):
        for (x, lens) in enumerate(self.stats.morph_lengths):
            for y in lens:
                normalized = lens[y] / float(self.stats.max_morph_len_count)
                c = [1.0 - normalized] * 3
                plt.plot(x, y, 's', color=c, markersize=(normalized * 20.))
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Morph type length distribution')
        self._title()

    def gold_bpr(self):
        plt.plot(self.stats.gold_bpr)
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Boundary precision recall score')
        plt.legend(['Precision', 'Recall', 'F-measure'])
        self._title()

    def changes(self):
        plt.plot(self.stats.changes)
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Changed segmentations (cumulative within iteration)')
        self._title()

    def _iteration_grid(self):
        for i in range(len(self.stats.iteration_numbers) - 1):
            if (self.stats.iteration_numbers[i] !=
                    self.stats.iteration_numbers[i + 1]):
                plt.axvline(x=(i + 0.5), color=[.6, .6, .6])
            if (self.stats.operation_numbers[i] <
                    self.stats.operation_numbers[i + 1]):
                plt.axvline(x=(i + 0.5), color=[.5, .5, .5], linestyle=':')

    def _title(self):
        plt.title(self.stats.title)
