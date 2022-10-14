import networkx as nx
import pandas as pd

data = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"
compiled_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_BF/"

df = pd.read_table(data, sep="\t", header=0)
X = df.loc[df.index[-23:]].transpose() # TF
compiled_res = pd.read_table(compiled_data + "Final_score_table.txt", sep=",", header=0)