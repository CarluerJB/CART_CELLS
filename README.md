# Tools for Gene Regulatory Network Analysis

## Description
This package contains a Python library to analyse Single-Cell data using cart tree, grn and clustering.
### CART_class
A pipeline allowing to build CART type decision tree on each gene for a single cell data set.
All the result are compiled and filtered according to user parameters.

### GRN class
A pipeline allowing to build GRN based on results from CART_class.
Using goatools high degree node' Go-term are analysed.
GRN parameters can be setup by the user.

### CLUSTERING_class
Using GRN_class analysis, the clustering is made on goatools association analysis' p-values.
Two way Dendrogram are build to group Go-term and genes.

## Installation
Make sure you python >= 3.11.0, and run the setup file to download all the needed file/library.

## Cookbook
run.sh contains a typical case of use of the full pipeline
main.py has been setup to use the full pipeline
PARAMETERS directory is free to be modified with your specific parameters.
Parameters has been splitted according to analysis step.