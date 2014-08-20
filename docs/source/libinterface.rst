Python library interface to Morfessor FlatCat
=============================================

Morfessor FlatCat 1.0 contains a library interface in order to be integrated in other
python applications. The public members are documented below and should remain
relatively the same between Morfessor FlatCat versions. Private members are documented
in the code and can change anytime in releases.

The classes are documented below.

IO class
--------
.. automodule:: flatcat.io
   :members:

.. _flatcat-model-label:

Model classes
-------------
.. automodule:: flatcat.flatcat
   :members:


Code Examples for using library interface
=========================================

Initialize a semi-supervised model from a given segmentation and annotations
----------------------------------------------------------------------------
::
    
    import flatcat

    io = flatcat.FlatcatIO()

    morph_usage = flatcat.categorizationscheme.MorphUsageProperties()

    model = flatcat.FlatcatModel(morph_usage, corpusweight=1.0)

    model.add_corpus_data(io.read_segmentation_file('segmentation.txt'))

    model.add_annotations(io.read_annotations_file('annotations.txt'),
                          annotatedcorpusweight=1.0)

    model.initialize_hmm()

The model is now ready to be trained.


Segmenting new data using an existing model
-------------------------------------------

First printing only the segmentations, followed by the analysis with morph categories.

::

    import flatcat

    io = flatcat.FlatcatIO()

    model = io.read_binary_model_file('model.pickled')

    words = ['words', 'segmenting', 'morfessor', 'categories', 'semisupervised']

    for word in words:
        print(model.viterbi_segment(word))

    for word in words:
        print(model.viterbi_analyze(word))

