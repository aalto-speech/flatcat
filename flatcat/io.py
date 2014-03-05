from __future__ import unicode_literals

import bz2
import collections
import codecs
import datetime
import gzip
import locale
import logging
import re
import sys

import morfessor

from . import get_version
from .categorizationscheme import get_categories, CategorizedMorph
from .exception import InvalidCategoryError
from .utils import _generator_progress

try:
    # In Python2 import cPickle for better performance
    import cPickle as pickle
except ImportError:
    import pickle

PY3 = sys.version_info.major == 3

_logger = logging.getLogger(__name__)


class FlatcatIO(morfessor.MorfessorIO):
    """Definition for all input and output files. Also handles all
    encoding issues.

    The only state this class has is the separators used in the data.
    Therefore, the same class instance can be used for initializing multiple
    files.

    Extends Morfessor Baseline data file formats to include category tags.
    """

    def __init__(self,
                 encoding=None,
                 construction_separator=' + ',
                 comment_start='#',
                 compound_separator='\s+',
                 analysis_separator=',',
                 category_separator='/'):
        super(FlatcatIO, self).__init__(
            encoding=encoding,
            construction_separator=construction_separator,
            comment_start=comment_start,
            compound_separator=compound_separator,
            atom_separator=None)
        self.analysis_separator = analysis_separator
        self.category_separator = category_separator

    def write_segmentation_file(self, file_name, segmentations):
        """Write segmentation file.

        File format (single line, wrapped only for pep8):
        <count> <construction1><cat_sep><category1><cons_sep>...
                <constructionN><cat_sep><categoryN>
        """

        _logger.info("Saving segmentations to '%s'..." % file_name)
        with self._open_text_file_write(file_name) as file_obj:
            d = datetime.datetime.now().replace(microsecond=0)
            file_obj.write('# Output from Morfessor FlatCat {}, {!s}\n'.format(
                get_version(), d))
            for count, morphs in segmentations:
                s = self.construction_separator.join(
                    ['{}{}{}'.format(m.morph, self.category_separator,
                                      m.category)
                     for m in morphs])
                file_obj.write('{} {}\n'.format(count, s))
        _logger.info("Done.")

    def read_segmentation_file(self, file_name):
        """Read segmentation file.
        see docstring for write_segmentation_file for file format.
        """
        _logger.info("Reading segmentations from '%s'..." % file_name)
        re_space = re.compile(r'\s+')
        for line in self._read_text_file(file_name):
            count, analysis = re_space.split(line, 1)
            try:
                count = int(count)
            except ValueError:
                # first column was compound instead of count
                count = 1
            cmorphs = []
            for morph_cat in analysis.split(self.construction_separator):
                cmorphs.append(self._morph_or_cmorph(morph_cat))
            yield(count, tuple(cmorphs))
        _logger.info("Done.")

    def read_annotations_file(self, file_name, construction_sep=' ',
                              analysis_sep=None):
        """Read an annotations file.

        Each line has the format:
        <compound> <constr1> <constr2>... <constrN>, <constr1>...<constrN>, ...

        Returns a defaultdict mapping a compound to a list of analyses.

        """
        analysis_sep = (analysis_sep if analysis_sep
                        else self.analysis_separator)
        annotations = collections.defaultdict(list)
        _logger.info("Reading annotations from '%s'..." % file_name)
        for line in self._read_text_file(file_name):
            compound, analyses_line = line.split(None, 1)
            analysis = self._read_annotation(analyses_line,
                                             construction_sep,
                                             analysis_sep)
            annotations[compound].extend(analysis)
        _logger.info("Done.")
        return annotations

    def read_combined_file(self, file_name, annotation_prefix='<',
                           construction_sep=' ',
                           analysis_sep=','):
        """Reads a file that combines unannotated word tokens
        and annotated data.
        The formats are the same as for files containing only one of the
        mentioned types of data, except that lines with annotations are
        additionally prefixed with a special symbol.
        """
        for line in self._read_text_file(file_name):
            if line.startswith(annotation_prefix):
                analysis = self._read_annotation(
                    line[len(annotation_prefix):],
                    construction_sep=construction_sep,
                    analysis_sep=analysis_sep)[0]
                compound = ''.join([x.morph for x in analysis])
                yield (True, 1, compound, analysis)
            else:
                for compound in self.compound_sep_re.split(line):
                    if len(compound) > 0:
                        yield (False, 1, compound, self._split_atoms(compound))

    def write_formatted_file(self,
                             file_name,
                             line_format,
                             data,
                             data_func,
                             newline_func=None,
                             output_tags=False,
                             construction_sep=None,
                             analysis_sep=None,
                             category_sep=None,
                             filter_tags=None,
                             filter_len=3):
        """Writes a file in the specified format.

        Formatting is flexible: even formats that cannot be read by
        FlatCat can be specified.
        """
        construction_sep = (construction_sep if construction_sep
                            else self.construction_separator)
        analysis_sep = (analysis_sep if analysis_sep    # FIXME
                        else self.analysis_separator)
        category_sep = (category_sep if category_sep
                        else self.category_separator)

        if output_tags:
            def _output_morph(cmorph):
                return '{}{}{}'.format(cmorph.morph,
                                        category_sep,
                                        cmorph.category)
        else:
            def _output_morph(cmorph):
                return cmorph.morph

        with self._open_text_file_write(file_name) as fobj:
            for item in _generator_progress(data):
                if newline_func is not None and newline_func(item):
                    fobj.write("\n")
                    continue
                (count, compound, alternatives, logp) = data_func(item)

                analysis = []
                for constructions in alternatives:
                    if filter_tags is not None:
                        constructions = [cmorph for cmorph in constructions
                                        if cmorph.category not in filter_tags
                                            or len(cmorph) > filter_len]
                    constructions = [_output_morph(cmorph)
                                     for cmorph in constructions]
                    analysis.append(construction_sep.join(constructions))
                analysis = analysis_sep.join(analysis)
                fobj.write(line_format.format(
                                analysis=analysis,
                                compound=compound,
                                count=count,
                                logprob=logp))

    def _read_annotation(self, line, construction_sep, analysis_sep):
        if analysis_sep is not None:
            analyses = line.split(analysis_sep)
        else:
            analyses = [line]

        out = []
        for analysis in analyses:
            analysis = analysis.strip()
            segments = analysis.split(construction_sep)
            out.append(tuple(self._morph_or_cmorph(x) for x in segments))
        return out

    def _morph_or_cmorph(self, morph_cat):
        """Parses a string describing a morph, either tagged
        or not tagged, returing a CategorizedMorph.
        """
        parts = morph_cat.rsplit(self.category_separator, 1)
        morph = parts[0].strip()
        if len(parts) == 1:
            category = None
        else:
            category = parts[1]
            if category not in get_categories():
                raise InvalidCategoryError(category)
        cmorph = CategorizedMorph(morph, category)
        return cmorph
