import logging
import time

_logger = logging.getLogger(__name__)

NO_PLOTTING = True
try:
    from matplotlib import pyplot as plt
    NO_PLOTTING = False
except ImportError:
    _logger.info('Unable to import matplotlib.pyplot: plotting disabled')
    
from .exception import UnsupportedConfigurationError

class IterationStatistics(object):
    def __init__(self, title=None):
        self.iteration_numbers = []
        self.operation_numbers = []
        self.epoch_numbers = []

        self.costs = []
        self.tag_counts = []
        self.morph_tokens = []
        self.durations = [0]

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

    def callback(self, model, epoch_number=0):
        t_cur = time.time()

        self.iteration_numbers.append(model._iteration_number)
        self.operation_numbers.append(model._operation_number)
        self.epoch_numbers.append(epoch_number)

        self.costs.append(model.get_cost())
        tcounts = self._extract_tag_counts(model)
        self.tag_counts.append(tcounts)
        self.morph_tokens.append(sum(tcounts))
        self.word_tokens = float(model.word_tokens)
        if self.t_prev is not None:
            self.durations.append(t_cur - self.t_prev)

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
        self.tag_counts()
        plt.figure()
        self.avg_morphs()
        plt.figure()
        self.durations()
        plt.show()

    def costs(self):
        plt.plot(self.stats.costs)
        self._iteration_grid()
        plt.xlabel('Epoch number')
        plt.ylabel('Model cost')
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

    def _iteration_grid(self):
        for i in range(len(self.stats.iteration_numbers) - 1):
            if (self.stats.iteration_numbers[i] !=
                    self.stats.iteration_numbers[i + 1]):
                plt.axvline(x=(i + 0.5), color=[.6, .6, .6])
            if (self.stats.operation_numbers[i] <
                    self.stats.operation_numbers[i + 1]):
                plt.axvline(x=(i + 0.5), color=[.6, .6, .6], linestyle=':')

    def _title(self):
        plt.title(self.stats.title)
