# TaBIIC - Taxonomy Building Through Iterative and Interactive Clustering

This repository contains the implementation of a tool as a proof of concept of the approach described in the paper
d’Aquin, M. (2023). TaBIIC: Taxonomy Building Through Iterative and Interactive Clustering. In Formal Ontology in Information Systems (pp. 155-168). IOS Press.
https://ebooks.iospress.nl/volumearticle/66098 


TaBIIC is a prototype tool to create an initial version of a taxonomy of concepts, with some ontological definitions attached to them, from tabular data. This idea can be summarised as follows: Taxonomies are made of concepts that represent subsets of items that are similar in the sense that they share some characteristics. If we have available a dataset corresponding to a set of items with characteristics, it is therefore natural to try to use clustering methods (such as K-Means or agglomerative clustering) to generate an initial version of such a taxonomy. However, doing so leaves very little control over the definition of the concepts included, and therefore will lead to significant efforts afterwards in refining, describing and organising them.

TaBIIC introduces such element of control by using clustering iteratively and interactively. What this means is that, as the taxonomy is being built, the user has the possibility to label concepts, fix their definitions (the characteristics that groupe them together), and further divide them using clustering. In this way, the taxonomy created is at the same time aligned to the data from which it was generated, and aligned with the users understanding and knowledge of the domain. It can in addition be loaded directly into an ontology editing tool (such as [Protégé](https://protege.stanford.edu/)) to be further refined.

## Running TaBIIC

At this point, TaBIIC can be seen as a proof of concept, and is therefore not very well documented. It should however be relatively easy to figure out the dependencies and run it: It is built in Python using `flask` to provide a web interface for interaction. To start the server, simply use

``python app.py``

then go to [http://localhost:5000](http://localhost:5000) to get to the interface.

## Using TaBIIC

The first thing the interface is going to ask is a file. TaBIIC should be able to load a standard CSV file assuming that the first line represents the labels of columns, that each other line represents an item, and that each column contains the values of the characteristics of items.

Once the file is loaded, a large red rectangle represents the concept of all the items in the dataset. You can use the left panel to name it and to choose the type of each variable (column). ID variables are ignored. You can then cut this first concept in two using clustering.

Once you have obtained initial clusters, you can name them as concepts and inspect what variables have values that are more different from the other cluster. By clicking on a variable, you can fix its value or range of values, so to provide a definition for the concept. The population (items grouped under that concept) will then be recalculated, and the other cluster given a definition corresponding to the complement of the definition given to this one.

You can then go ahead cuting those concepts as well.

Once satisfied with the taxonomy of concepts, their labels and their definitions, the taxonomy can be saved into an OWL file using the RDF/Turtle syntax. The taxonomic relations will be translated in this file into `rdfs:subClassOf` properties, and the definitions into restrictions in OWL. 
