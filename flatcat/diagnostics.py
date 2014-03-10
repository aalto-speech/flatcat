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

from morfessor import baseline
from .exception import UnsupportedConfigurationError, ArgumentException


class IterationStatistics(object):
    def __init__(self, title=None):
        self.epoch_numbers = []
        self.operation_numbers = []
        self.iteration_numbers = []

        self.costs = []
        self.cost_parts = []
        self.tag_counts = []
        self.morph_types = []
        self.morph_tokens = []
        self.durations = [0]
        self.morph_lengths = []
        self.changes = []
        self.changes_op = []
        self.violated_annots = []

        self.gold_bpr = []
        self._reference = None

        self.max_morph_len = 0
        self.max_morph_len_count = 0
        self.t_prev = None
        self.word_tokens = 1.0
        self.categories = None

        if title is None:
            self.title = 'epoch statistics {}'.format(
                time.strftime("%a, %d.%m.%Y %H:%M:%S"))
        else:
            self.title = title
        self.ops = None

    def set_names(self, model, training_operations):
        self.ops = training_operations
        self.categories = model.get_categories()
        model._changed_segmentations = set()
        model._changed_segmentations_op = set()

    def set_gold_standard(self, reference):
        self._reference = reference

    def callback(self, model, iteration_number=0):
        t_cur = time.time()

        self.epoch_numbers.append(model._epoch_number)
        self.operation_numbers.append(model._operation_number)
        self.iteration_numbers.append(iteration_number)

        self.costs.append(model.get_cost())
        ccc = model._corpus_coding.get_cost()
        lcc = model._lexicon_coding.get_cost()
        if model._supervised:
            acc_unscaled = model._annot_coding.get_cost()
            acc = acc_unscaled / model._annot_coding.weight
        else:
            acc_unscaled = 0
            acc = 0
        self.cost_parts.append([(ccc / model._corpus_coding.weight),
                                (lcc / model._lexicon_coding.weight),
                                acc,
                                ccc,
                                lcc,
                                acc_unscaled
                               ])
        tcounts = self._extract_tag_counts(model)
        self.tag_counts.append(tcounts)
        self.morph_types.append(len(model._morph_usage.seen_morphs()))
        self.morph_tokens.append(sum(tcounts))
        self.word_tokens = float(model.word_tokens)
        self.changes.append(len(model._changed_segmentations))
        self.changes_op.append(len(model._changed_segmentations_op))
        if model._supervised:
            # sum expression gives length of the generator
            self.violated_annots.append(
                sum(1 for _ in model.violated_annotations()))
        else:
            self.violated_annots.append(0)

        if self._reference is not None:
            tmp = self._reference.items()
            wlist, annotations = zip(*tmp)
            segments = [model.viterbi_analyze(w)[0] for w in wlist]
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

    def show(self, style):
        if style == 'stacked':
            self.stacked()
        else:
            self.all()

    def all(self):
        plt.figure()
        self.costs()
        self._title()
        plt.figure()
        self.basecosts()
        self._title()
        plt.figure()
        self.violated_annots()
        self._title()
        plt.figure()
        self.tag_counts()
        self._title()
        plt.figure()
        self.avg_morphs()
        self._title()
        plt.figure()
        self.durations()
        plt.figure()
        self.types_and_tokens()
        self._title()
        #plt.figure()
        #self.morph_lengths()
        plt.figure()
        self.changes()
        if self.stats._reference is not None:
            plt.figure()
            self.gold_bpr()
            self._title()
        plt.show()

    def stacked(self):
        plt.figure(figsize=(5.5 * 2, 5.5 * 2))
        plt.subplot(3, 2, 1)
        self.costs(xlabel=False, zoom=True)
        plt.subplot(3, 2, 2)
        self.types(xlabel=False)
        plt.subplot(3, 2, 3)
        self.violated_annots(xlabel=False)
        if self.stats._reference is not None:
            plt.subplot(3, 2, 4)
            self.gold_bpr(xlabel=False)
        plt.subplot(3, 2, 5)
        self.changes(both=False)
        plt.subplot(3, 2, 6)
        self.tag_counts()
        plt.subplots_adjust(left=0.123, bottom=0.06, right=0.98, top=0.97,
                            wspace=None, hspace=0)
        plt.show()

    def costs(self, xlabel=True, zoom=False):
        plt.plot(self.stats.costs, marker='+')
        self._epoch_grid(xlabel=xlabel)
        if zoom:
            plt.ylim(min(self.stats.costs[1:]), max(self.stats.costs))
        if xlabel:
            plt.xlabel('iteration number')
        plt.ylabel('Model cost')

    def violated_annots(self, xlabel=True):
        plt.plot(self.stats.violated_annots, marker='+')
        self._epoch_grid(xlabel=xlabel)
        if xlabel:
            plt.xlabel('iteration number')
        plt.ylabel('Violated annotations')

    def basecosts(self):
        if (len(self.stats.cost_parts) == 0 or
                len(self.stats.cost_parts[0]) != 6):
            _logger.info('Not plotting cost components: ' +
                         'wrong number of variables (old data?)')
            return
        plt.plot(self.stats.cost_parts, marker='+')
        self._epoch_grid()
        plt.xlabel('iteration number')
        plt.ylabel('Cost component')
        plt.legend(['U Corp', 'U Lexi', 'U Anno',
                    'W Corp', 'W Lexi', 'W Anno'], loc='best')

    def tag_counts(self, xlabel=True):
        plt.plot(self.stats.tag_counts, marker='+')
        plt.gca().yaxis.get_major_formatter().set_powerlimits((-3, 4))
        #unzipped = zip(*self.stats.tag_counts)
        #for (i, series) in enumerate(unzipped):
        #    plt.plot(series, color=plt.cm.jet(float(i) /
        #        float(len(self.stats.categories))), marker='+')
        self._epoch_grid(xlabel=xlabel)
        if xlabel:
            plt.xlabel('iteration number')
        plt.ylabel('Category occurence count')
        if self.stats.categories is not None:
            plt.legend(self.stats.categories, loc='best',
                       prop={'size': 11}, labelspacing=0.2)

    def avg_morphs(self):
        normalized = [x / self.stats.word_tokens
                      for x in self.stats.morph_tokens]
        plt.plot(normalized, marker='+')
        self._epoch_grid()
        plt.xlabel('iteration number')
        plt.ylabel('Avg number of morphs per word token')

    def types(self, xlabel=True):
        plt.plot(self.stats.morph_types, color="blue", marker='+')
        self._epoch_grid(xlabel=xlabel)
        if xlabel:
            plt.xlabel('iteration number')
        plt.ylabel('Count of morph types')

    def types_and_tokens(self):
        plt.plot(self.stats.morph_tokens, color="red", marker='+')
        plt.plot(self.stats.morph_types, color="blue", marker='+')
        plt.legend(['Tokens', 'Types'], loc='best')
        self._epoch_grid()
        plt.xlabel('iteration number')
        plt.ylabel('Count of morph tokens / types')

    def durations(self):
        by_epoch = [0.0] * (max(self.stats.epoch_numbers) + 1)
        by_op = [0.0] * (max(self.stats.operation_numbers) + 1)
        by_iteration = [0.0] * (max(self.stats.iteration_numbers) + 1)

        for i in range(len(self.stats.epoch_numbers)):
            by_epoch[self.stats.epoch_numbers[i]] += self.stats.durations[i]
            by_op[self.stats.operation_numbers[i]] += self.stats.durations[i]
            by_iteration[
                self.stats.iteration_numbers[i]] += self.stats.durations[i]

        plt.subplot(2, 2, 1)
        plt.plot(self.stats.durations, marker='+')
        self._epoch_grid()
        plt.xlabel('iteration number')
        plt.ylabel('iteration duration [s]')
        self._title()

        plt.subplot(2, 2, 2)
        plt.bar(range(len(by_epoch)), by_epoch)
        plt.ylabel('Total epoch duration [s]')
        xls = range(len(by_epoch))
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
        plt.bar(range(len(by_iteration)), by_iteration)
        plt.ylabel('Total iteration duration [s]')
        xls = range(len(by_iteration))
        xs = [x + 0.5 for x in xls]
        plt.xticks(xs, xls)

    def morph_lengths(self):
        for (x, lens) in enumerate(self.stats.morph_lengths):
            for y in lens:
                normalized = lens[y] / float(self.stats.max_morph_len_count)
                c = [1.0 - normalized] * 3
                plt.plot(x, y, 's', color=c, markersize=(normalized * 20.))
        self._epoch_grid()
        plt.xlabel('iteration number')
        plt.ylabel('Morph type length distribution')
        self._title()

    def gold_bpr(self, xlabel=True):
        plt.plot(self.stats.gold_bpr, marker='+')
        self._epoch_grid(xlabel=xlabel)
        if xlabel:
            plt.xlabel('iteration number')
        plt.ylabel('Boundary precision recall score')
        plt.legend(['Precision', 'Recall', 'F-measure'], loc='best',
                   prop={'size': 11}, labelspacing=0.2)

    def changes(self, xlabel=True, both=True):
        if both:
            plt.plot(self.stats.changes, color='blue', marker='+')
        plt.plot(self.stats.changes_op, color='red', marker='+')
        if both:
            plt.legend(['cumulative w/in epoch', 'in iteration'], loc='best')
        self._epoch_grid(xlabel=xlabel)
        if xlabel:
            plt.xlabel('iteration number')
        plt.ylabel('Changed segmentations')

    def _epoch_grid(self, xlabel=True):
        num_ticks = len(self.stats.epoch_numbers) - 1
        for i in range(num_ticks):
            if (self.stats.epoch_numbers[i] !=
                    self.stats.epoch_numbers[i + 1]):
                plt.axvline(x=(i + 0.5), color=[.6, .6, .6])
            if (self.stats.operation_numbers[i] <
                    self.stats.operation_numbers[i + 1]):
                plt.axvline(x=(i + 0.5), color=[.5, .5, .5], linestyle=':')
        if not xlabel:
            plt.xticks(range(num_ticks), [''] * num_ticks)

    def _title(self):
        plt.title(self.stats.title)
