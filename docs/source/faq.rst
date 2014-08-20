Frequently asked questions
==========================

UnicodeError: Can not determine encoding of input files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the encoding of the input data is not detected automatically,
you need to specify it on the command line, e.g. ::
    
    --encoding=ISO_8859-15

The word counts in the output are smaller than I expected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the counts in the input data are already dampened,
you should not specify a dampening to FlatCat. ::

    --dampening none

The input does not seem to be segmented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you receive the following warning when loading a segmentation::

    #################### WARNING ####################
    The input does not seem to be segmented.
    Are you using the correct construction separator?

the reason might be that the data is in an unexpected format.
By default Morfessor FlatCat assumes that morphs are separated by
a plus sign surrounded on both sides by a space ' + '.
If your data uses e.g. only a space to separate morphs, you need to specify::

    --construction-separator ' '
