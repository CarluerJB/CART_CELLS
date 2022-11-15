from os import stat
from random import seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import sys
import seaborn as sns
from scipy import stats
from sklearn import metrics
import statsmodels.api as sm
from umap import UMAP


#data = "/media/carluerj/Data/These/DATA/gene_regulator_network/20K_prefiltred_contrib_filter_23_TF_002.txt"
data = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"
out_data = "/media/carluerj/Data/These/Results/GRN_inference/"
compiled_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_005/"

parameters = sys.argv[1:]
print(parameters)
bool_parameters = {
    
}

for key, val in bool_parameters.items():
    if key in parameters:
        bool_parameters[key] = True
        parameters.remove(key)

args = {key : value for key, value in zip(parameters[0::2], parameters[1::2])}


thres_model = float(args["-tm"]) if "-tm" in args.keys() else 0.0
thres_gini = float(args["-tg"]) if "-tg" in args.keys() else 1.0
thres_zero_te_tr = float(args["-tz"]) if "-tz" in args.keys() else 1.0
thres_node = float(args["-tn"]) if "-tn" in args.keys() else 10.0
thres_pval = float(args["-tp"]) if "-tp" in args.keys() else 3.0

sort_on_model = int(args["-ms"]) if "-ms" in args.keys() else None
sort_on_gini = int(args["-gs"]) if "-gs" in args.keys() else None
sort_on_zero_te_tr = int(args["-zs"]) if "-zs" in args.keys() else None
sort_on_means = int(args["-mes"]) if "-mes" in args.keys() else None
sort_on_node = int(args["-ns"]) if "-ns" in args.keys() else None
sort_on_iterscore = int(args["-iss"]) if "-iss" in args.keys() else None
sort_on_pvalscore = int(args["-pvs"]) if "-pvs" in args.keys() else None

res_file_pathname = args["-fn"] if "-fn" in args.keys() else None

sort_by = [None]*7
sort_order = [None]*7
if sort_on_model!=None:
    sort_by[sort_on_model] = "model_score"
    sort_order[sort_on_model] = False
if sort_on_gini!=None:
    sort_by[sort_on_gini] = "gini_score_0"
    sort_order[sort_on_gini] = True
if sort_on_zero_te_tr!=None:
    sort_by[sort_on_zero_te_tr] = "ratio"
    sort_order[sort_on_zero_te_tr] = True
if sort_on_means!=None:
    sort_by[sort_on_means] = "ratio_mean"
    sort_order[sort_on_means] = True
if sort_on_node!=None:
    sort_by[sort_on_node] = "node"
    sort_order[sort_on_node] = True
if sort_on_iterscore!=None:
    sort_by[sort_on_iterscore] = "iter_score"
    sort_order[sort_on_iterscore] = True
if sort_on_pvalscore!=None:
    sort_by[sort_on_pvalscore] = "p-val_TF1"
    sort_order[sort_on_pvalscore] = False
sort_by = [i for i in sort_by if i != None]
sort_order = [i for i in sort_order if i != None]

print(thres_gini)

# Read data
df = pd.read_table(data, sep="\t", header=0)
X = df.loc[df.index[-23:]].transpose() # TF
Y = df.drop([*df.index[-23:]]).transpose() # AGI
mean = Y.mean()
std = Y.std()
Y_norm = Y-mean/std
ind_dict = dict((k, i) for i, k in enumerate(Y_norm.index.values))
compiled_res = pd.read_table(compiled_data + "Final_score_table_MANNWHIT_TEST_FINAL_byline.txt", sep=",", header=0)

# Load parameter based on AGI
compiled_res['ratio'] = compiled_res['perc_zero_tot']
compiled_res['ratio_mean'] = (compiled_res['mean_TF1+'] + compiled_res['mean_TF1-'])/2
compiled_res['p-val_TF1'] = -np.log10(compiled_res['p-val_TF1'].values)
compiled_res['N2 anova p-val_TF1-TF2'] = -np.log10(compiled_res['N2 anova p-val_TF1-TF2'].values)
compiled_res['N3 anova p-val_TF1-TF2'] = -np.log10(compiled_res['N3 anova p-val_TF1-TF2'].values)

## FILTERING STEP
# global filtering rules
print(compiled_res.dtypes)
print(compiled_res[(compiled_res['gini_score_0'] <= thres_gini)])
print(compiled_res[((compiled_res['gini_score_0'] <= thres_gini) & (compiled_res['gini_score_1'] <= thres_gini) & (compiled_res['gini_score_2'] <= thres_gini))])
print(compiled_res[(compiled_res['model_score'] >= thres_model)])
print(compiled_res[(compiled_res['ratio'] >= thres_zero_te_tr)])
print(compiled_res[(compiled_res['p-val_TF1'] <= thres_pval)])
compiled_res = compiled_res.loc[compiled_res[(((compiled_res['gini_score_0'] <= thres_gini) & (compiled_res['gini_score_1'] <= thres_gini) & (compiled_res['gini_score_2'] <= thres_gini)) & 
                                (compiled_res['model_score'] >= thres_model) & 
                                (compiled_res['ratio'] <= thres_zero_te_tr)) & 
                                ((compiled_res['p-val_TF1'] >= thres_pval) | (compiled_res['N2 anova p-val_TF1-TF2'] >= thres_pval) | (compiled_res['N3 anova p-val_TF1-TF2'] >= thres_pval))].index]


## SORTING STEP
print(compiled_res)
compiled_res.sort_values(by=sort_by, ascending=sort_order, inplace=True)
candidate = compiled_res["AGI"].to_numpy()

with open(res_file_pathname, "w") as file:
    for elem in candidate:
        file.write(elem + "\n")
