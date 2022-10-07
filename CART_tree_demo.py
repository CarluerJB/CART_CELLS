import pandas as pd
from sklearn import tree
import graphviz   
import sys

# IN/OUT PATH
data = "demo_dataset.txt"

# Table operation
df = pd.read_table(data, sep="\t", header=0)
X = df.loc[df.index[-23:]].transpose() 
Y = df.drop([*df.index[-23:]]).transpose() # AGI only

# Write the gene that will be focus in a text file
with open("list_gene.txt", 'w') as file:
    for elem in Y.columns:
        file.write(elem + "\n")
i=0
# Go through each genes
for elem, ielem in zip(Y.columns, range(len(Y.columns))):

    sys.stdout.write("\rWORKING ON elem n° {0} = {1}".format(ielem, elem))
    sys.stdout.flush()

    # Y_single is for 1 gene on 2K cells
    Y_single = Y[elem]

    # CART TREE
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y_single)
    print(clf)
    # Result saving to visual tree
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=X.columns,class_names=Y_single.index,filled=True, rounded=True,  special_characters=True)
    graph = graphviz.Source(dot_data, elem+".gv")
    graph.render(directory='results_demo', view=False)