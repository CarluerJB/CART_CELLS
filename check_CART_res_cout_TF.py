import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"
out_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_BF/"
compiled_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_BF/"


df = pd.read_table(data, sep="\t", header=0)
X = df.loc[df.index[-23:]].transpose() # TF
compiled_res = pd.read_table(compiled_data + "Final_score_table.txt", sep=",", header=0)
TF_list_count = {elem : 0 for elem in X.columns}
# for TF in TF_list_count.keys():
count_TF_found = compiled_res['TF'].value_counts()
for key, value in count_TF_found.items():
    TF_list_count[key] = value

print(TF_list_count)
plt.bar(TF_list_count.keys(), TF_list_count.values())
plt.show()
