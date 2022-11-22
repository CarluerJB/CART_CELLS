import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split, cross_val_score
import graphviz   
import sys
import ast
import os

datas_source = "Lauras"
NORMALIZE = False
CLASS_TRANSFO = False
MIN_SAMPLE_CUT = 0.10
CLASS_WEIGHT = None

# IN/OUT PATH
data = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"
out_data = "/media/carluerj/Data/These/Results/GRN_inference/"
tf_list_path = "/media/carluerj/Data/These/DATA/gene_regulator_network/tf_list.txt"

# CREATE OUT DIR and FILE
# os.makedirs(out_data + "list_gene_005/" + "score", exist_ok=True)
# os.makedirs(out_data + "list_gene_005/" + "txt_tree", exist_ok=True)
# os.makedirs(out_data + "list_gene_005/" + "tree", exist_ok=True)
# with open(out_data + "list_gene_005/" + "list_gene.txt", 'w') as file:
#     pass
# Table operation
data_src_type = data.split('.')[1]
if data_src_type == "txt":
    df = pd.read_table(data, sep="\t", header=0)
elif data_src_type == "h5":
    df = pd.read_hdf(data)
else:
    raise NotImplementedError

tf_list = pd.read_table(tf_list_path, header=None)[0].to_list()

X = df.drop(tf_list).transpose()
Y = df.loc[tf_list].transpose()

# Table normalisation
if NORMALIZE:
    pass # TODO

# Table quantile to class
if CLASS_TRANSFO:
    Y_copy = Y.copy(deep=True)
    X_quant = np.quantile(X.to_numpy().flatten(), [0.33,0.66,1.0]).astype(np.int64())
    Y_quant = np.quantile(Y.to_numpy().flatten(), [0.33,0.66,1.0]).astype(np.int64())
    Y[(Y_copy <= Y_quant[2]) & (Y_copy >= Y_quant[1])] = 2
    Y[(Y_copy <= Y_quant[1]) & (Y_copy >= Y_quant[0])] = 1
    Y[(Y_copy <= Y_quant[0])] = 0

# Go through each genes
for elem, ielem in zip(Y.columns, range(len(Y.columns))):
    sys.stdout.write("\rWORKING ON elem n° {0} = {1}".format(ielem, elem))
    sys.stdout.flush()

    # Y_single is for 1 gene on 2K cells
    Y_single = Y[elem]
    perc_zero = len(Y_single[Y_single==0].index) / len(Y_single.index)
    
    # GENERAL CART TREE
    clf = tree.DecisionTreeClassifier(min_samples_split=MIN_SAMPLE_CUT, max_depth=3, min_samples_leaf = MIN_SAMPLE_CUT, class_weight=CLASS_WEIGHT, criterion="gini")
    clf = clf.fit(X, Y_single)
    tree_rules = export_text(clf, feature_names=list(X.columns.values), show_weights=False)

    # EVALUATE CART TREE MODEL
    x_train, x_test, y_train, y_test = train_test_split(X, Y_single, train_size=0.8)
    clf_eval = tree.DecisionTreeClassifier(min_samples_split=MIN_SAMPLE_CUT, max_depth=3, min_samples_leaf = MIN_SAMPLE_CUT, class_weight=CLASS_WEIGHT, criterion="gini")
    clf_eval = clf_eval.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Result saving to visual tree
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=X.columns,class_names=Y_single.index,filled=True, rounded=True,  special_characters=True)
    graph = graphviz.Source(dot_data, elem+".gv")
    graph.render(directory=out_data + 'list_gene_005/tree', view=False)
    with open(out_data + "list_gene_005/" + "txt_tree/" + elem + ".txt", "w") as file:
        file.write(tree_rules)
    with open(out_data + "list_gene_005/" + "score/" + elem + ".txt", "w") as file:
        for i in range(len(clf.tree_.impurity)):
                file.write(str(clf.tree_.impurity[i]) + "," + str(clf.tree_.n_node_samples[i]) + "\n")
    # Write the gene that has been focus in a text file + accuracy score for each
    # print(y_train.value_counts())
    # print(y_pred)
    # print(list(y_train.values))
    with open(out_data + "list_gene_005/" + "list_gene.txt", 'a') as file:
        file.write(elem + ";" + 
                    str(metrics.accuracy_score(y_test, y_pred)) + ";" + 
                    str(np.where(np.array(y_pred)==0.0)[0].size / np.array(y_pred).size) + ";" + 
                    str(np.where(np.array(y_train.to_numpy())==0.0)[0].size / np.array(y_train.values).size) + ";" + 
                    str(np.where(np.array(y_test.to_numpy())==0.0)[0].size / np.array(y_test.values).size) + ";" + 
                    str(perc_zero) + "\n")
    # exit(0)