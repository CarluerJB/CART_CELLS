# GRN_CART
A pipeline allowing to build CART type decision tree on each gene for a single cell data set

# clustering.py
Allow to make a image representation of single cell data, clustering is made by cell and by TF

# CART_tree_demo.py
A minimalist CART_tree for demonstration purpose

# CART_tree.py
Allow to build a CART tree for each gene as the Y target, the X is the TFs.
results are save as txt tree map, as a pdf map representation and as .gv meta data representation

# compile_CART_results.py
Allow to create a summary of CART_tree results, t-test are made to test each first condition, and tukey is made to test each combination of first-second comdition.
results are stored in a csv table, with zero percentage

# check_CART_res.py
Allow to have a representation for a given gene of the CART tree condition, by representing subsets distributions. t-test and tukey test are used here to give more information to user.
