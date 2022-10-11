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
compiled_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_BF/"

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

sort_on_model = int(args["-ms"]) if "-ms" in args.keys() else None
sort_on_gini = int(args["-gs"]) if "-gs" in args.keys() else None
sort_on_zero_te_tr = int(args["-zs"]) if "-zs" in args.keys() else None
sort_on_node = int(args["-ns"]) if "-ns" in args.keys() else None

sort_by = [None]*5
sort_order = [None]*5
if sort_on_model!=None:
    sort_by[sort_on_model] = "model_score"
    sort_order[sort_on_model] = False
if sort_on_gini!=None:
    sort_by[sort_on_gini] = "gini_score"
    sort_order[sort_on_gini] = True
if sort_on_zero_te_tr!=None:
    sort_by[sort_on_zero_te_tr] = "ratio"
    sort_order[sort_on_zero_te_tr] = True
if sort_on_node!=None:
    sort_by[sort_on_node] = "node"
    sort_order[sort_on_node] = True
sort_by = [i for i in sort_by if i != None]
sort_order = [i for i in sort_order if i != None]


# Read data
df = pd.read_table(data, sep="\t", header=0)
X = df.loc[df.index[-23:]].transpose() # TF
Y = df.drop([*df.index[-23:]]).transpose() # AGI
mean = Y.mean()
std = Y.std()
Y_norm = Y-mean/std
ind_dict = dict((k, i) for i, k in enumerate(Y_norm.index.values))
compiled_res = pd.read_table(compiled_data + "Final_score_table.txt", sep=",", header=0)

# Load parameter based on AGI
compiled_res = compiled_res.loc[(compiled_res['sign']=='<=')]
compiled_res['ratio'] = (compiled_res['perc_zero_test'] + compiled_res['perc_zero_pred'])/2


## FILTERING STEP
# filtering on GINI score
compiled_res = compiled_res.loc[compiled_res[compiled_res['gini_score'] <= thres_gini].index]

# filtering on model score
compiled_res = compiled_res.loc[compiled_res[compiled_res['model_score'] >= thres_model].index]

# filtering on ratio score
compiled_res = compiled_res.loc[compiled_res[compiled_res['ratio'] <= thres_zero_te_tr].index]

# filtering on node indice
compiled_res = compiled_res.loc[compiled_res[compiled_res['node'] <= thres_node].index]

## SORTING STEP
compiled_res.sort_values(by=sort_by, ascending=sort_order, inplace=True)
print(compiled_res)
