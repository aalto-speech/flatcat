General
=======

Morfessor FlatCat is a method for learning of morphological segmentation of natural language.
FlatCat uses a Hidden Markov Model with the states ``{prefix, stem, suffix, non-morpheme}``
for modeling the word internal ordering of morphs.
FlatCat can be trained in an unsupervised or semi-supervised fashion.

Publications
------------

The Morfessor FlatCat method is presented in the paper [Gronroos2014a]_.
The paper is available for download from
http://www.aclweb.org/anthology/C/C14/C14-1111.pdf

The Morfessor FlatCat method is described in detail in the Master's Thesis [Gronroos2014b]_.

Morfessor FlatCat was developed together with the Morfessor 2.0 software package.
The work done in Morfessor 2.0 is described in detail in the Morfessor 2.0
Technical Report [TechRep]_. The report is available for download from
http://urn.fi/URN:ISBN:978-952-60-5501-5.

Citing
------

The authors do kindly ask that you cite the Morfessor FlatCat paper
 [Gronroos2014a]_ when using this tool in academic publications.

In addition, when you refer to other Morfessor algorithms, you should cite the
respective publications where they have been introduced. For example, the first
Morfessor algorithm was published in [Creutz2002]_ and the semi-supervised
extension in [Kohonen2010]_. See [TechRep]_ for further information on the
relevant publications.

.. [Gronroos2014a] Grönroos, S.-A. and Virpioja, S. (2014). Morfessor FlatCat: An HMM-based method for unsupervised and semi-supervised learning of morphology. In Proceedings of the 25th International Conference on Computational Linguistics.  Pages 1177--1185, Dublin, Ireland, August 2014, Association for Computational Linguistics.

.. [Gronroos2014b] Grönroos, S.-A. (2014). Semi-supervised induction of a concatenative morphology with simple morphotactics: A model in the Morfessor family. Master's thesis, Aalto University, Helsinki.

.. [TechRep] Sami Virpioja, Peter Smit, Stig-Arne Grönroos, and Mikko Kurimo. Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline. Aalto University publication series SCIENCE + TECHNOLOGY, 25/2013. Aalto University, Helsinki, 2013. ISBN 978-952-60-5501-5.

.. [Creutz2002] Mathias Creutz and Krista Lagus. Unsupervised discovery of morphemes. In Proceedings of the Workshop on Morphological and Phonological Learning of ACL-02, pages 21-30, Philadelphia, Pennsylvania, 11 July, 2002. 

.. [Kohonen2010] Oskar Kohonen, Sami Virpioja and Krista Lagus. Semi-supervised learning of concatenative morphology. In Proceedings of the 11th Meeting of the ACL Special Interest Group on Computational Morphology and Phonology, pages 78-86, Uppsala, Sweden, July 2010. Association for Computational Linguistics.

