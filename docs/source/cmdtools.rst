Command line tools
==================

The installation process installs 5 scripts in the appropriate PATH.

flatcat
-------
The flatcat command is a full-featured script for training, updating models
and segmenting test data.

Initializing or loading existing model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FlatCat model needs to be initialized with an initial segmentation,
which can be produced e.g. using Morfessor Baseline.
Loading an existing FlatCat model is done by initializing using the saved model.
If you don't want to perform training on the loaded model, make sure to use
`flatcat-segment` or specify ``-m none`` to specify that no training needs to be done.

positional argument ``<init file>``
    Initialize or load a model from the specified file.
    By default a segmentation file is expected. The segmentation can
    be compressed using bz2/gzip.
    If the file name ends in ``.pickled``, it is treated as a saved
    binary FlatCat model.
``-L <file>, --load-parameters <file>``
    load hyper-parameters from the specified file.
    Alternatively the hyper-parameters can be given on the command line.
    A binary model stores the hyper-parameter values used in training,
    so you do not need to load or specify them separately.


Extending the model with additional data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--extend <file>``
    Input data to extend the model with. This can be an untagged corpus file,
    or an already tagged segmentation. Binary FlatCat models can not be used
    to extend a model.  Data can be read from standard input
    by specifying '-' as the file name. You can use --extend several times
    in order to append multiple files).
``-A <file>, --annotations <file>``
    Add annotated data from the specified file.
``-T <file>, --testdata <file>``
    Input corpus file(s) to analyze (text or bz2/gzipped text; use '-' for
    standard input; add several times in order to append multiple files). The
    file is read in the same manner as an input corpus file. See
    :ref:`data-format-options` for changing the delimiters used for
    separating compounds and atoms.


Training model options
~~~~~~~~~~~~~~~~~~~~~~

``-m <mode>, --mode <mode>``
    Morfessor FlatCat can run in different modes, each doing different actions on
    the model. The modes are:

    none
        Do initialize or train a model. Can be used when just loading a model
        for segmenting new data
    batch
        Loads an existing model (which is already initialized with training
        data) and run
        Can be used to retrain an already trained model, after extending it
        with more data.
    online
        Initialize a model, and then extend it in an on-line fashion.
    online+batch
        Initialization and online training followed by batch training.

``-d <type>, --dampening <type>``
    Method for changing the compound counts in the input data.
    Note, that if the counts in the intialization file are already dampened
    (e.g. by Morfessor Baseline), you should not specify dampening again.
    Options:

    none
        Do not alter the counts of compounds (token based training)
    log
        Change the count :math:`x` of a compound to :math:`\log(x)` (log-token
        based training)
    ones
        Treat all compounds as if they only occured once (type based training)

``-f <list>, --forcesplit <list>``
    A list of atoms that would always cause the compound to be split. By
    default only hyphens (``-``) would force a split. Note the notation of the
    argument list. To have no force split characters, use as an empty string as
    argument (``-f ""``). To split, for example, both hyphen (``-``) and
    apostrophe (``'``) use ``-f "-'"``
``--nosplit-re <regexp>``
    If the expression matches the two surrounding
    characters, do not allow splitting (default None)

``--skips``
    Use random skips for frequently seen compounds to
    speed up oneline training. Has no effect on batch training.

``--batch-minfreq <int>``
    Compound frequency threshold for batch training
    (default 1)
``--max-epochs <int>``
    Hard maximum of epochs in training
``--online-epochint <int>``
    Epoch interval for online training (default 10000)


Saving model
~~~~~~~~~~~~

``-s <file>``
    save the model as an analysis (can be compressed by specifying
    a file ending .gz or .bz2). Use together with ``-S <file>``.
``-S <file>, --save-parameters <file>``
    save hyper-parameters into file.
``--save-binary-model``
    save  :ref:`binary-model-def`.
    Not recommended for long-term storage of models, due to bit-rot.
``-x <file>, --lexicon <file>``
    save the morph lexicon

Examples
~~~~~~~~
Initialize a model from the Morfessor Baseline segmentation baseline_segmentation.txt,
batch train the model using a perplexity threshold of 10,
save the model as an analysis file analysis.gz and a hyper-parameter file parameters.txt,
and segment the test.txt set: ::

    flatcat baseline_segmentation.txt -p 10 -s analysis.gz -S parameters.txt -T test.txt --remove-nonmorphemes -o test.segmentation

flatcat-train
---------------
The flatcat-train command is a convenience command that enables easier
training for Morfessor FlatCat models.

The basic command structure is: ::

    flatcat-train [arguments] initialization-file

The arguments are identical to the ones for the `flatcat`_ command. The most
relevant are:

``-s <file>``
    save the model as an analysis (can be compressed by specifying
    a file ending .gz or .bz2). Use together with ``-S <file>``.
``-S <file>, --save-parameters <file>``
    save hyper-parameters into file.

Examples
~~~~~~~~
Train a Morfessor FlatCat model from a Morfessor Baseline segmentation in ISO_8859-15 encoding,
writing the log to logfile,
and saving the model as a binary file model.pickled: ::

    flatcat-train baseline_segmentation.txt --encoding=ISO_8859-15 -p 10 --logfile=log.log --save-binary-model model.pickled

flatcat-segment
-----------------
The flatcat-segment command is a convenience command that enables easier
segmentation of test data with a Morfessor FlatCat model.

The basic command structure is: ::

    flatcat-segment [arguments] model-file test-data [test-data ...]

The arguments are identical to the ones for the `flatcat`_ command. The most
relevant are:

``-L <file>``
    Load hyper-parameters from file. Not necessary if the model is saved in binary format.
``-o <file>``
    Output the segmentation of the test data into this file.
``--remove-nonmorphemes``
    Apply heuristics for non-morpheme removal to the segmentation output,
    to ensure that no morphemes categorized as non-morphemes (ZZZ) remain.
``--output-categories``
    Include the categories in the segmentation output.
    Default is to only output the surface form of the morphs.

Examples
~~~~~~~~
Loading a model from analysis.gz, hyper-parameters from parameters.txt
and segmenting the file test_corpus.txt: ::

    flatcat-segment analysis.gz -L parameters.txt --remove-nonmorphemes -o test_corpus.segmented test_corpus.txt

Include the categories in the output: ::

    flatcat-segment analysis.gz -L parameters.txt --output-categories -o test_corpus.segmented test_corpus.txt

Use FlatCat as a stemmer by removing prefixes and suffixes: ::

    flatcat-segment analysis.gz -L parameters.txt --filter-categories PRE,SUF -o test_corpus.segmented test_corpus.txt

flatcat-diagnostics
-------------------

The flatcat-diagnostics command is used to plot the diagnostic statistics
collected by giving the parameters ``--statsfile <file>`` and
``--stats-annotations <file>`` to `flatcat` or `flatcat-train`.

Examples
~~~~~~~~

Collect statistics during training,
using development set devset.segmentation: ::
    
    flatcat-train baseline_segmentation.txt -p 10 --save-binary-model model.pickled --statsfile stats.pickled --stats-annotations devset.segmentation

Plot the statistics: ::

    flatcat-diagnostics stats.pickled

.. _data-format-options:

flatcat-reformat
----------------

A reformatting tool which makes format conversion operations on category tagged data
a bit more convenient. Work in progress.

Data format command line options
--------------------------------


``--encoding <encoding>``
    Encoding of input and output files (if none is given, both the local
    encoding and UTF-8 are tried).
``--compound-separator <regexp>``
    compound separator regexp (default '\s+')
``--analysis-separator <str>``
    separator for different analyses in an annotation file. Use NONE for only
    allowing one analysis per line.
``--output-format <format>``
    format string for --output file (default: '{analysis}\\n'). Valid keywords
    are: ``{analysis}`` = constructions of the compound, ``{compound}`` =
    compound string, {count} = count of the compound (currently always 1),
    ``{logprob}`` = log-probability of the analysis,
    Valid escape sequences are ``\n`` (newline) and ``\t`` (tabular).
``--output-format-separator <str>``
    construction separator for analysis in ``--output`` file (default: ' ').
``--output-newlines``
    for each newline in input, print newline in ``--output`` file (default: 'False').

..  and ``{clogprob}`` = log-probability of the compound.



Universal command line options
------------------------------
``--verbose <int>  -v``
    verbose level; controls what is written to the standard error stream or log file (default 1)
``--logfile <file>``
    write log messages to file in addition to standard error stream
``--progressbar``
    Force the progressbar to be displayed (possibly lowers the log level for the standard error stream)
``--help``
    -h show this help message and exit
``--version``
    show version number and exit
