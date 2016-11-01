from __future__ import unicode_literals

import collections
import datetime
import logging
import re
import sys

import bz2
import codecs
import gzip
import locale
import os
import tarfile
from contextlib import contextmanager

import morfessor

from . import get_version
from .categorizationscheme import get_categories, CategorizedMorph
from .exception import InvalidCategoryError
from .flatcat import FlatcatModel
from .utils import _generator_progress, _is_string

PY3 = sys.version_info.major == 3

if PY3:
    from io import BytesIO as StringIO
else:
    from StringIO import StringIO

_logger = logging.getLogger(__name__)

BINARY_ENDINGS = ('.pickled', '.pickle', '.bin')
TARBALL_ENDINGS = ('.tar.gz', '.tgz')


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
                 category_separator='/',
                 strict=True):
        super(FlatcatIO, self).__init__(
            encoding=encoding,
            construction_separator=construction_separator,
            comment_start=comment_start,
            compound_separator=compound_separator,
            atom_separator=None)
        self.analysis_separator = analysis_separator
        self.category_separator = category_separator
        self._strict = strict
        self._version = get_version()

    def write_tarball_model_file(self, file_name, model):
        _logger.info("Saving model as tarball...")
        if '.tar.gz' not in file_name:
            _logger.warn('Tarball model misleadingly named: {}'.format(
                file_name))
        with TarGzModel(file_name, 'w') as tarmodel:
            with tarmodel.newmember('params') as member:
                self.write_parameter_file(member,
                                        model.get_params())
            with tarmodel.newmember('analysis') as member:
                self.write_segmentation_file(member,
                                           model.segmentations)
            if model._supervised:
                with tarmodel.newmember('annotations') as member:
                    self.write_annotations_file(
                         member,
                         model.annotations,
                         construction_sep=' ',
                         output_tags=True)

    def read_tarball_model_file(self, file_name, model=None):
        """Read model from a tarball."""
        if model is None:
            model = FlatcatModel()
        with TarGzModel(file_name, 'r') as tarmodel:
            for (name, fobj) in tarmodel.members():
                if name == 'params':
                    model.set_params(
                        self.read_parameter_file(fobj))
                elif name == 'analysis':
                    model.add_corpus_data(
                        self.read_segmentation_file(fobj))
                elif name == 'annotations':
                    model.add_annotations(
                        self.read_annotations_file(fobj))
                else:
                    _logger.warn(
                        'Unknown model component {}'.format(name))
        return model

    def read_any_model(self, file_name):
        """Read a complete model in either binary or tarball format.
           This method can NOT be used to initialize from a
           Morfessor 1.0 style segmentation"""
        if any(file_name.endswith(ending) for ending in BINARY_ENDINGS):
            model = self.read_binary_model_file(file_name)
        elif any(file_name.endswith(ending) for ending in TARBALL_ENDINGS):
            model = self.read_tarball_model_file(file_name)
        else:
            raise Exception(
                'No indentified file ending in "{}"'.format(file_name))
        model.initialize_hmm()
        return model

    def write_segmentation_file(self, file_name, segmentations,
                                construction_sep=None,
                                output_tags=True,
                                comment_string=''):
        """Write segmentation file.

        File format (single line, wrapped only for pep8):
        <count> <construction1><cat_sep><category1><cons_sep>...
                <constructionN><cat_sep><categoryN>
        """
        construction_sep = (construction_sep if construction_sep
                            else self.construction_separator)

        _logger.info("Saving analysis to '%s'..." % file_name)
        output_morph = _make_morph_formatter(
            self.category_separator, output_tags)
        with self._open_text_file_write(file_name) as file_obj:
            d = datetime.datetime.now().replace(microsecond=0)
            file_obj.write(
                '# Output from Morfessor {}{}, {!s}\n'.format(
                get_version(), comment_string, d))
            for count, morphs in segmentations:
                s = construction_sep.join(
                    [output_morph(m) for m in morphs])
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
            analysis = self.read_annotation(analyses_line,
                                             construction_sep,
                                             analysis_sep)
            annotations[compound].extend(analysis)
        _logger.info("Done.")
        return annotations

    def write_annotations_file(self,
                               file_name,
                               annotations,
                               construction_sep=' ',
                               analysis_sep=None,
                               output_tags=False):
        _logger.info("Writing annotations to '%s'..." % file_name)

        def _annotation_func(item):
            (compound, annotation) = item
            try:
                alts = annotation.alternatives
            except AttributeError:
                alts = annotation
            return (1, compound, alts, 0, 0)

        self.write_formatted_file(
            file_name,
            '{compound}\t{analysis}\n',
            sorted(annotations.items()),
            _annotation_func,
            analysis_sep=analysis_sep,
            output_tags=output_tags,
            construction_sep=construction_sep)
        _logger.info("Done.")

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
                analysis = self.read_annotation(
                    line[len(annotation_prefix):],
                    construction_sep=construction_sep,
                    analysis_sep=analysis_sep)[0]
                compound = ''.join([x.morph for x in analysis])
                yield (True, 1, compound, analysis)
            else:
                for compound in self.compound_sep_re.split(line):
                    if len(compound) > 0:
                        yield (False, 1, compound, self._split_atoms(compound))

    def write_lexicon_file(self, file_name, lexicon):
        """Write to a Lexicon file all constructions
        and their emission counts.
        """
        _logger.info("Saving model lexicon to '%s'..." % file_name)
        with self._open_text_file_write(file_name) as file_obj:
            for (construction, counts) in lexicon:
                count = sum(counts)
                file_obj.write('{}\t{}\t{}\n'.format(count,
                                                     construction,
                                                     '\t'.join('{}'.format(x)
                                                          for x in counts)))
        _logger.info("Done.")

    def write_formatted_file(self,
                             file_name,
                             line_format,
                             data,
                             data_func,
                             newline_func=None,
                             output_newlines=False,
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

        output_morph = _make_morph_formatter(category_sep, output_tags)

        with self._open_text_file_write(file_name) as fobj:
            for item in _generator_progress(data):
                if newline_func is not None and newline_func(item):
                    if output_newlines:
                        fobj.write("\n")
                    continue
                (count, compound, alternatives, logp, clogp) = data_func(item)

                analysis = []
                if len(alternatives) == 1:
                    constructions = alternatives[0]
                    num_morphs = len(constructions)
                    num_nonmorphemes = sum(1 for cmorph in constructions
                                           if cmorph.category == 'ZZZ')
                    num_letters = sum(len(cmorph.morph)
                                      for cmorph in constructions)
                else:
                    num_morphs = None
                    num_nonmorphemes = None
                    num_letters = None
                for constructions in alternatives:
                    if filter_tags is not None:
                        constructions = [cmorph for cmorph in constructions
                                        if cmorph.category not in filter_tags
                                            or len(cmorph) > filter_len]
                    constructions = [output_morph(cmorph)
                                     for cmorph in constructions]
                    analysis.append(construction_sep.join(constructions))
                analysis = analysis_sep.join(analysis)
                fobj.write(line_format.format(
                                analysis=analysis,
                                compound=compound,
                                count=count,
                                logprob=logp,
                                clogprob=clogp,
                                num_morphs=num_morphs,
                                num_nonmorphemes=num_nonmorphemes,
                                num_letters=num_letters))

    def read_annotation(self, line, construction_sep, analysis_sep=None):
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
            if self._strict and category not in get_categories():
                raise InvalidCategoryError(category)
        cmorph = CategorizedMorph(morph, category)
        return cmorph

    #### Copypasta from pre-12.2015 version of Baseline ###
    # (io API was changed in a noncompatible way)
    def read_corpus_files(self, file_names):
        """Read one or more corpus files.

        Yield for each compound found (1, compound, compound_atoms).

        """
        for file_name in file_names:
            for item in self.read_corpus_file(file_name):
                yield item

    def read_corpus_list_files(self, file_names):
        """Read one or more corpus list files.

        Yield for each compound found (count, compound, compound_atoms).

        """
        for file_name in file_names:
            for item in self.read_corpus_list_file(file_name):
                yield item

    def read_corpus_file(self, file_name):
        """Read one corpus file.

        For each compound, yield (1, compound, compound_atoms).
        After each line, yield (0, \"\\n\", ()).

        """
        _logger.info("Reading corpus from '%s'..." % file_name)
        for line in self._read_text_file(file_name, raw=True):
            for compound in self.compound_sep_re.split(line):
                if len(compound) > 0:
                    yield 1, compound, self._split_atoms(compound)
            yield 0, "\n", ()
        _logger.info("Done.")

    def read_corpus_list_file(self, file_name):
        """Read a corpus list file.

        Each line has the format:
        <count> <compound>

        Yield tuples (count, compound, compound_atoms) for each compound.

        """
        _logger.info("Reading corpus from list '%s'..." % file_name)
        for line in self._read_text_file(file_name):
            try:
                count, compound = line.split(None, 1)
                yield int(count), compound, self._split_atoms(compound)
            except ValueError:
                yield 1, line, self._split_atoms(line)
        _logger.info("Done.")

    #### This can be removed once it finds its way to Baseline ####
    #
    def _open_text_file_write(self, file_name_or_obj):
        """Open a file for writing with the appropriate compression/encoding"""
        if _is_string(file_name_or_obj):
            file_name = file_name_or_obj
            if file_name == '-':
                file_obj = sys.stdout
                if PY3:
                    return file_obj
            elif file_name.endswith('.gz'):
                file_obj = gzip.open(file_name, 'wb')
            elif file_name.endswith('.bz2'):
                file_obj = bz2.BZ2File(file_name, 'wb')
            else:
                file_obj = open(file_name, 'wb')
        else:
            file_obj = file_name_or_obj

        if self.encoding is None:
            # Take encoding from locale if not set so far
            self.encoding = locale.getpreferredencoding()
        return codecs.getwriter(self.encoding)(file_obj)

    def _open_text_file_read(self, file_name_or_obj):
        """Open a file for reading with the appropriate compression/encoding"""
        if _is_string(file_name_or_obj):
            file_name = file_name_or_obj
            if file_name == '-':
                if PY3:
                    return sys.stdin
                else:
                    class StdinUnicodeReader:
                        def __init__(self, encoding):
                            self.encoding = encoding
                            if self.encoding is None:
                                self.encoding = locale.getpreferredencoding()

                        def __iter__(self):
                            return self

                        def next(self):
                            l = sys.stdin.readline()
                            if not l:
                                raise StopIteration()
                            return l.decode(self.encoding)

                    return StdinUnicodeReader(self.encoding)
            else:
                if file_name.endswith('.gz'):
                    file_obj = gzip.open(file_name, 'rb')
                elif file_name.endswith('.bz2'):
                    file_obj = bz2.BZ2File(file_name, 'rb')
                else:
                    file_obj = open(file_name, 'rb')
        else:
            file_obj = file_name_or_obj
            if self.encoding is None:
                self.encoding = locale.getpreferredencoding()

        if self.encoding is None:
            # Try to determine encoding if not set so far
            self.encoding = self._find_encoding(file_name)
        inp = codecs.getreader(self.encoding)(file_obj)
        return inp

    # straight copypasta
    def _read_text_file(self, file_name, raw=False):
        """Read a text file with the appropriate compression and encoding.

        Comments and empty lines are skipped unless raw is True.

        """
        inp = self._open_text_file_read(file_name)
        try:
            for line in inp:
                line = line.rstrip()
                if not raw and \
                   (len(line) == 0 or line.startswith(self.comment_start)):
                    continue
                if self.lowercase:
                    yield line.lower()
                else:
                    yield line
        except KeyboardInterrupt:
            if file_name == '-':
                _logger.info("Finished reading from stdin")
                return
            else:
                raise

    def read_parameter_file(self, file_name):
        """Read learned or estimated parameters from a file"""
        params = {}
        line_re = re.compile(r'^([^:]*)\s*:\s*(.*)$')
        for line in self._read_text_file(file_name):
            m = line_re.match(line.rstrip())
            if m:
                key = m.group(1)
                val = m.group(2)
                try:
                    val = float(val)
                except ValueError:
                    pass
                params[key] = val
        return params


class TarGzMember(object):
    """File-like object that writes itself into the tarfile on closing"""
    def __init__(self, arcname, tarmodel):
        self.arcname = arcname
        self.tarmodel = tarmodel
        self.strio = None

    def __enter__(self):
        self.strio = StringIO()
        return self

    def __exit__(self, typ, value, trace):
        self.close()

    def close(self):
        if self.strio.closed:
            return
        info = tarfile.TarInfo(name=self.arcname)
        self.strio.seek(0, os.SEEK_END)
        info.size = self.strio.tell()
        self.strio.seek(0)
        self.tarmodel.tarfobj.addfile(tarinfo=info, fileobj=self.strio)
        self.strio.close()

    def write(self, *args, **kwargs):
        self.strio.write(*args, **kwargs)

    def __repr__(self):
        return '{} in {}'.format(
            self.arcname, self.tarmodel.filename)


class TarGzModel(object):
    """A wrapper to hide the ugliness of the tarfile API.

    Both TarGzModel itself and the method newmember are context managers:
    Writing a model requires a nested with statement.
    """

    def __init__(self, filename, mode):
        self.filename = filename
        if mode == 'w':
            self.mode = 'w|gz'
        else:
            self.mode = 'r|gz'
        self.tarfobj = None

    def __enter__(self):
        self.tarfobj = tarfile.open(self.filename, self.mode)
        return self

    def __exit__(self, typ, value, trace):
        self.tarfobj.close()

    def newmember(self, arcname):
        """Receive a new member to the .tar.gz archive.

        Arguments:
            arcname - the name of the file within the archive.
        Returns:
            a file-like object into which the contents can be written.
            This is a context manager: use a "with" statement.
        """

        assert 'w' in self.mode
        return TarGzMember(arcname, self)

    def members(self):
        """Generates the (name, contents) pairs for each file in
        the archive.

        The contents are in the form of file-like objects.
        The files are generated in the order they are in the archive:
        the recipient must be able to handle them in an arbitrary order.
        """

        assert 'r' in self.mode
        while True:
            info = self.tarfobj.next()
            if info is None:
                break
            fobj = self.tarfobj.extractfile(info)

            yield (info.name, fobj)

            fobj.close()
    #
    #### End of stuff belonging in Baseline ####


def _make_morph_formatter(category_sep, output_tags):
    if output_tags:
        def output_morph(cmorph):
            if cmorph.category is None:
                return cmorph.morph
            return '{}{}{}'.format(cmorph.morph,
                                    category_sep,
                                    cmorph.category)
    else:
        def output_morph(cmorph):
            try:
                return cmorph.morph
            except AttributeError:
                return cmorph
    return output_morph
