Quickstart guide
================

Setting up a virtual environment
--------------------------------

The Morfessor packages are created using the current Python packaging
standards, as described on http://docs.python.org/install/. Morfessor packages
are fully compatible with, and recommended to run in, virtual environments as
described on http://virtualenv.org.

If you don't want to use virtualenv, you can skip this step.

::

    virtualenv flatcat
    cd flatcat
    source bin/activate

Installation instructions
-------------------------

Morfessor FlatCat installation packages can be obtained from the
`Morpho project homepage`_ (latest stable version)
or the `FlatCat Github page`_ (all versions).

Morfessor FlatCat 1.0 is installed using setuptools library for Python.

.. or can be directly installed from the `Python Package Index (PyPI)`_.

Unpack the tarball or zip file, change into the newly created directory, and then run::

    python setup.py install

The setup command also installs Morfessor Baseline, which is a dependency of FlatCat.

.. A second method is to use the tool pip on the tarball or zip file directly::
.. 
..     pip install morfessor-VERSION.tar.gz



Installation from PyPI
----------------------

Morfessor FlatCat is also distributed through the `Python Package Index (PyPI)`_.
This means that tools like pip and easy_install can automatically download and
install the latest version of Morfessor.

Simply type::

    pip install Morfessor-FlatCat

To install Morfessor FlatCat.

Basic workflow
--------------

This quickstart workflow assumes that your training corpus is in ``data/train.txt``,
your gold standard segmentation annotations are in ``data/annotations.txt``,
and your testing wordlist is in ``data/test.txt``.
All of these files are encoded with the encoding of your shell locale.
We also assume that you wish to use Morfessor Baseline as initialization method.
If you have different needs, see :ref:`command-line-tools` for details.

We will be using the values 100 for the perplexity threshold,
1.0 for the unannotated corpus likelihood weight (alpha),
and 0.1 for the annotated corpus likelihood weight (beta).
You will need to adjust these values for your data set.

Perform the Morfessor Baseline segmentation::

    morfessor-train data/train.txt -S baseline.gz -w 1.0 -A data/annotations.txt -W 0.1

Perform Morfessor FlatCat training::

    flatcat-train baseline.gz -p 100 -w 1.0 -A data/annotations.txt -W 0.1 -s analysis.tar.gz

Segment the test data with the trained Morfessor FlatCat model::
    
    flatcat-segment analysis.tar.gz data/test.txt -o test_corpus.segmented --remove-nonmorphemes --output-categories 


.. _Morpho project homepage: http://morpho.aalto.fi/projects/morpho/
.. _FlatCat Github page: https://github.com/aalto-speech/flatcat/releases
.. _Python Package Index (PyPI): https://pypi.python.org/pypi/Morfessor
