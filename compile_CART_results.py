from os import sep
import ast
from re import T
import sys
import pandas as pd
import graphviz
from sklearn import tree
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
np.seterr(divide = 'ignore') 
import warnings
warnings.filterwarnings("ignore")


in_dir = sys.argv[1]
out_data = sys.argv[1]
data_path = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"


AGI_name = pd.read_table(out_data + "list_gene.txt", header=None, sep=";")
data = pd.read_table(data_path, sep="\t", header=0)
X = data.loc[data.index[-23:]].transpose() # TF
Y = data.drop([*data.index[-23:]]).transpose() # AGI

df = pd.DataFrame({"gini_score" : [], "sample_size" : [], "AGI" : [], "node" : [], "model_score" : [], "TF" : [], "sign" : [], "lim" : [], "p-val" : []})#, "y_test" : [], "y_pred" : []})

for ielem, elem, score, y_test, y_pred in zip(range(len(AGI_name[0].values)), AGI_name[0].values, AGI_name[1].values, AGI_name[2].values, AGI_name[3].values):
    sys.stdout.write("\rWORKING ON elem n° {0} = {1}".format(ielem, elem))
    sys.stdout.flush()
    a = pd.read_table(out_data + "score/" + elem + ".txt", header=None, sep=",", names=["gini_score", "sample_size"])
    a["AGI"] = elem
    a["node"] = a.index
    a["model_score"] = score
    # a["y_test"] = str(y_test)
    # a["y_pred"] = str(y_pred)
    a["TF"] = None
    a["sign"] = None
    a["lim"] = None
    a["anova p-val_TF1"] = None
    a["anova p-val_TF2"] = None
    a["anova p-val_TF1-TF2"] = None
    a["p-val_++_+-"] = None
    a["p-val_++_-+"] = None
    a["p-val_++_--"] = None
    a["p-val_+-_-+"] = None
    a["p-val_+-_--"] = None
    a["p-val_-+_--"] = None
    a["perc_zero_test"] = np.where(np.array(ast.literal_eval(y_test))==0.0)[0].size / np.array(ast.literal_eval(y_test)).size
    a["perc_zero_pred"] = np.where(np.array(ast.literal_eval(y_pred))==0.0)[0].size / np.array(ast.literal_eval(y_pred)).size
    # Tree to table
    with open(out_data + "txt_tree/" + elem + ".txt", 'r') as file :
        tree_txt = file.readlines()
    cond = tree_txt[0][(tree_txt[0].find('|--- ') + len("|--- ")) : ]
    if cond.find(' <=') != -1:
            a.loc[0, "TF"] = cond[:cond.find(' <=')]
            a.loc[0, 'lim'] = None
            a.loc[0, 'sign'] = "root"
    i = 1
    node_i_list=[]
    for line in tree_txt:
        cond = line[(line.find('|--- ') + len("|--- ")) : ]
        if cond.find(' <=') != -1:
            a.loc[i, "TF"] = cond[:cond.find(' <=')]
            a.loc[i, 'lim'] = float(cond[cond.find(' <=')+4:-2])
            a.loc[i, 'sign'] = "<="
            node_i_list.append(i)
        elif cond.find(' >') != -1:
            a.loc[i, "TF"] = cond[:cond.find(' >')]
            a.loc[i, 'lim'] = float(cond[cond.find(' >')+3:-2])
            a.loc[i, 'sign'] = ">"
        else:
            continue
        i+=1
    # Compute t-test on TF1 sup / TF1 inf
    if len(node_i_list)>0:
        cond = a[a["node"]==node_i_list[0]]
        TF_1_sup = Y.loc[X[cond['TF'].values[0]] > cond['lim'].values[0], cond['AGI']]
        TF_1_inf = Y.loc[X[cond['TF'].values[0]] <= cond['lim'].values[0], cond['AGI']]
        t_test_res = stats.ttest_ind(np.nan_to_num(np.log10(TF_1_sup.values), neginf=0), np.nan_to_num(np.log10(TF_1_inf.values), neginf=0) , alternative="two-sided")
        a.loc[a["node"]==node_i_list[0], "p-val_TF1"] = list(t_test_res)[1]
        # Compute anova-test on (TF1 sup / TF1 inf) and (TF2 sup / TF2 inf)
        if len(node_i_list)>1:
            cond2 = a[a["node"]==node_i_list[1]]
            TF1_TF2 = pd.DataFrame({'target' : [0]*len(Y.index), 'TF1' : [0]*len(Y.index), 'TF2' : [0]*len(Y.index)})
            TF1_TF2.index = Y.index
            TF1_TF2['target'] = Y.loc[:, cond['AGI']]
            TF1_TF2.loc[(X[cond['TF'].values[0]] > cond['lim'].values[0]), 'TF1'] = 1
            TF1_TF2.loc[(X[cond2['TF'].values[0]] > cond2['lim'].values[0]), 'TF2'] = 1
            
            # ANOVA TW TEST
            model = ols('target ~ C(TF1) + C(TF2) + C(TF1):C(TF2)', data=TF1_TF2).fit()
            result = sm.stats.anova_lm(model, type=2)
            a.loc[a["node"]==2, "anova p-val_TF1"] = result.loc['C(TF1)','PR(>F)']
            a.loc[a["node"]==2, "anova p-val_TF2"] = result.loc['C(TF2)','PR(>F)']
            a.loc[a["node"]==2, "anova p-val_TF1-TF2"] = result.loc['C(TF1):C(TF2)','PR(>F)']

            
            # TUKEY TEST
            
            TF_1_sup_TF_2_sup = Y.loc[(X[cond['TF'].values[0]] > cond['lim'].values[0]) & (X[cond2['TF'].values[0]] > cond2['lim'].values[0]), cond['AGI']]#.to_numpy().flatten().tolist()
            TF_1_sup_TF_2_inf = Y.loc[(X[cond['TF'].values[0]] > cond['lim'].values[0]) & (X[cond2['TF'].values[0]] <= cond2['lim'].values[0]), cond['AGI']]#.to_numpy().flatten().tolist()
            TF_1_inf_TF_2_sup = Y.loc[(X[cond['TF'].values[0]] <= cond['lim'].values[0]) & (X[cond2['TF'].values[0]] > cond2['lim'].values[0]), cond['AGI']]#.to_numpy().flatten().tolist()
            TF_1_inf_TF_2_inf = Y.loc[(X[cond['TF'].values[0]] <= cond['lim'].values[0]) & (X[cond2['TF'].values[0]] <= cond2['lim'].values[0]), cond['AGI']]#.to_numpy().flatten().tolist()
            # result = stats.tukey_hsd(TF_1_sup_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_inf_TF_2_inf) 
            TF_1_sup_TF_2_sup['sign'] = "++"
            TF_1_sup_TF_2_inf['sign'] = "+-"
            TF_1_inf_TF_2_sup['sign'] = "-+"
            TF_1_inf_TF_2_inf['sign'] = "--"
            tukey_data = TF_1_sup_TF_2_sup.append([TF_1_sup_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_inf_TF_2_inf])
            result = pairwise_tukeyhsd(tukey_data[cond['AGI']], groups=tukey_data['sign'])
            # print(result)
            # exit(0)
            # print(np.unique(result.groups))
            i = 0
            if len(TF_1_sup_TF_2_sup.index)>0:
                if len(TF_1_sup_TF_2_inf.index)>0:
                    a.loc[a["node"]==node_i_list[1], "p-val_++_+-"] = result.pvalues[i]
                    i=i+1
                if len(TF_1_inf_TF_2_sup.index)>0:
                    a.loc[a["node"]==node_i_list[1], "p-val_++_-+"] = result.pvalues[i]
                    i=i+1
                if len(TF_1_inf_TF_2_inf.index)>0:
                    a.loc[a["node"]==node_i_list[1], "p-val_++_--"] = result.pvalues[i]
                    i=i+1
            if len(TF_1_sup_TF_2_inf.index)>0:
                if len(TF_1_inf_TF_2_sup.index)>0:
                    a.loc[a["node"]==node_i_list[1], "p-val_+-_-+"] = result.pvalues[i]
                    i=i+1
                if len(TF_1_inf_TF_2_inf.index)>0:
                    a.loc[a["node"]==node_i_list[1], "p-val_+-_--"] = result.pvalues[i]
                    i=i+1
            if len(TF_1_inf_TF_2_sup.index)>0:
                if len(TF_1_inf_TF_2_inf.index)>0:
                    a.loc[a["node"]==node_i_list[1], "p-val_-+_--"] = result.pvalues[i]
                    i=i+1
        if len(node_i_list)>2:
            cond3 = a[a["node"]==node_i_list[2]]
            TF1_TF3 = pd.DataFrame({'target' : [0]*len(Y.index), 'TF1' : [0]*len(Y.index), 'TF3' : [0]*len(Y.index)})
            TF1_TF3.index = Y.index
            TF1_TF3['target'] = Y.loc[:, cond['AGI']]
            TF1_TF3.loc[(X[cond['TF'].values[0]] > cond['lim'].values[0]), 'TF1'] = 1
            TF1_TF3.loc[(X[cond3['TF'].values[0]] > cond3['lim'].values[0]), 'TF3'] = 1
            
            # ANOVA TW TEST
            model = ols('target ~ C(TF1) + C(TF3) + C(TF1):C(TF3)', data=TF1_TF3).fit()
            result = sm.stats.anova_lm(model, type=2)
            a.loc[a["node"]==node_i_list[2], "anova p-val_TF1"] = result.loc['C(TF1)','PR(>F)']
            a.loc[a["node"]==node_i_list[2], "anova p-val_TF2"] = result.loc['C(TF3)','PR(>F)']
            a.loc[a["node"]==node_i_list[2], "anova p-val_TF1-TF2"] = result.loc['C(TF1):C(TF3)','PR(>F)']
            
            # TUKEY TEST
            TF_1_sup_TF_3_sup = Y.loc[(X[cond['TF'].values[0]] > cond['lim'].values[0]) & (X[cond3['TF'].values[0]] > cond3['lim'].values[0]), cond['AGI']]
            TF_1_sup_TF_3_inf = Y.loc[(X[cond['TF'].values[0]] > cond['lim'].values[0]) & (X[cond3['TF'].values[0]] <= cond3['lim'].values[0]), cond['AGI']]
            TF_1_inf_TF_3_sup = Y.loc[(X[cond['TF'].values[0]] <= cond['lim'].values[0]) & (X[cond3['TF'].values[0]] > cond3['lim'].values[0]), cond['AGI']]
            TF_1_inf_TF_3_inf = Y.loc[(X[cond['TF'].values[0]] <= cond['lim'].values[0]) & (X[cond3['TF'].values[0]] <= cond3['lim'].values[0]), cond['AGI']]

            TF_1_sup_TF_3_sup['sign'] = "++"
            TF_1_sup_TF_3_inf['sign'] = "+-"
            TF_1_inf_TF_3_sup['sign'] = "-+"
            TF_1_inf_TF_3_inf['sign'] = "--"
            tukey_data = TF_1_sup_TF_3_sup.append([TF_1_sup_TF_3_sup, TF_1_sup_TF_3_inf, TF_1_inf_TF_3_sup, TF_1_inf_TF_3_inf])
            result = pairwise_tukeyhsd(tukey_data[cond['AGI']], groups=tukey_data['sign'])

            i = 0
            if len(TF_1_sup_TF_3_sup.index)>0:
                if len(TF_1_sup_TF_3_inf.index)>0:
                    a.loc[a["node"]==node_i_list[2], "p-val_++_+-"] = result.pvalues[i]
                    i=i+1
                if len(TF_1_inf_TF_3_sup.index)>0:
                    a.loc[a["node"]==node_i_list[2], "p-val_++_-+"] = result.pvalues[i]
                    i=i+1
                if len(TF_1_inf_TF_3_inf.index)>0:
                    a.loc[a["node"]==node_i_list[2], "p-val_++_--"] = result.pvalues[i]
                    i=i+1
            if len(TF_1_sup_TF_3_inf.index)>0:
                if len(TF_1_inf_TF_3_sup.index)>0:
                    a.loc[a["node"]==node_i_list[2], "p-val_+-_-+"] = result.pvalues[i]
                    i=i+1
                if len(TF_1_inf_TF_3_inf.index)>0:
                    a.loc[a["node"]==node_i_list[2], "p-val_+-_--"] = result.pvalues[i]
                    i=i+1
            if len(TF_1_inf_TF_3_sup.index)>0:
                if len(TF_1_inf_TF_3_inf.index)>0:
                    a.loc[a["node"]==node_i_list[2], "p-val_-+_--"] = result.pvalues[i]
                    i=i+1
    df = pd.concat([df, a])
df_sorted = df[df["node"]<=5.0].sort_values(by=['AGI', "node", "gini_score"])
df_sorted.to_csv(out_data + "Final_score_table.txt", header=True, index=False, sep=",")
