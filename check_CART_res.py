from os import stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# THIS SCRIPT NEED 3 ARGS
# TF
# TF
# LIM_VAL
show_first_layer_res = False
full_first_layer = True
kde = True
common_norm = False
histplot_multiple = "dodge"
histplot_binwidth=0.05
colors_palette = sns.color_palette("tab10", 4)
plt_config = 1
ratio_sign = 10
show_zero = False
va_text_sign = 'top'

medianprops = dict(
    linewidth=4, 
    color="#747473",
    solid_capstyle="butt"
)
boxprops = dict(
    linewidth=2, 
    color="#747473"
)

#data = "/media/carluerj/Data/These/DATA/gene_regulator_network/20K_prefiltred_contrib_filter_23_TF_002.txt"
data = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"
out_data = "/media/carluerj/Data/These/Results/GRN_inference/"
compiled_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_BF/"

# Read data
df = pd.read_table(data, sep="\t", header=0)
X = df.loc[df.index[-23:]].transpose() # TF
Y = df.drop([*df.index[-23:]]).transpose() # AGI
compiled_res = pd.read_table(compiled_data + "Final_score_table.txt", sep=",", header=0)

# Load parameter based on AGI
TG = sys.argv[1]
compiled_res = compiled_res.loc[(compiled_res['AGI']==TG) & (compiled_res['sign']=='<=')]
# if len(sys.argv)>2:
#     TF_1 = sys.argv[2]
#     lim_val_1 = float(sys.argv[3])
#     nb_plot = 1
#     is_node1_sign = None
#     if len(compiled_res[compiled_res['node']==1].index)>=1:
#         if (compiled_res[compiled_res['node']==1]['p-val_TF1']<0.05).to_numpy()[0]:
#             is_node1_sign = '*'
#             if (compiled_res[compiled_res['node']==1]['p-val_TF1']<0.01).to_numpy()[0]:
#                 is_node1_sign = '**'
#     else:
#         is_node1_sign = "???"
#     if len(sys.argv) > 4:
#         show_first_layer_res = True
#         if len(sys.argv) >= 6:
#             full_first_layer = False
#             TF_2 = sys.argv[4]
#             lim_val_2 = float(sys.argv[5])
#             nb_plot = 2
#             is_node2_TF1_sign = None
#             is_node2_TF2_sign = None
#             is_node2_TF1_TF2_sign = None
#             if len(compiled_res[compiled_res['node']==2].index)>=1:
#                 if (compiled_res[compiled_res['node']==2]['p-val_TF1']<0.05).to_numpy()[0]:
#                     is_node2_TF1_sign = '*'
#                     if (compiled_res[compiled_res['node']==2]['p-val_TF1']<0.01).to_numpy()[0]:
#                         is_node2_TF1_sign = '**'

#                 if (compiled_res[compiled_res['node']==2]['p-val_TF2']<0.05).to_numpy()[0]:
#                     is_node2_TF2_sign = '*'
#                     if (compiled_res[compiled_res['node']==2]['p-val_TF2']<0.01).to_numpy()[0]:
#                         is_node2_TF2_sign = '**'

#                 if (compiled_res[compiled_res['node']==2]['p-val_TF1-TF2']<0.05).to_numpy()[0]:
#                     is_node2_TF1_TF2_sign = '*'
#                     if (compiled_res[compiled_res['node']==2]['p-val_TF1-TF2']<0.01).to_numpy()[0]:
#                         is_node2_TF1_TF2_sign = '**'
#             else:
#                 is_node2_TF1_sign = "???"
#                 is_node2_TF2_sign = "???"
#                 is_node2_TF1_TF2_sign = "???"
#         if len(sys.argv) <= 8:
#             TF_3 = sys.argv[6]
#             lim_val_3 = float(sys.argv[7])
#             nb_plot = 3
#             is_node3_TF1_sign = None
#             is_node3_TF2_sign = None
#             is_node3_TF1_TF2_sign = None
#             if len(compiled_res[compiled_res['node']==3].index)>=1:
#                 if (compiled_res[compiled_res['node']==3]['p-val_TF1']<0.05).to_numpy()[0]:
#                     is_node3_TF1_sign = '*'
#                     if (compiled_res[compiled_res['node']==3]['p-val_TF1']<0.01).to_numpy()[0]:
#                         is_node3_TF1_sign = '**'

#                 if (compiled_res[compiled_res['node']==3]['p-val_TF2']<0.05).to_numpy()[0]:
#                     is_node3_TF2_sign = '*'
#                     if (compiled_res[compiled_res['node']==3]['p-val_TF2']<0.01).to_numpy()[0]:
#                         is_node3_TF2_sign = '**'

#                 if (compiled_res[compiled_res['node']==3]['p-val_TF1-TF2']<0.05).to_numpy()[0]:
#                     is_node3_TF1_TF2_sign = '*'
#                     if (compiled_res[compiled_res['node']==3]['p-val_TF1-TF2']<0.01).to_numpy()[0]:
#                         is_node3_TF1_TF2_sign = '**'
#             else:
#                 is_node3_TF1_sign = "???"
#                 is_node3_TF2_sign = "???"
#                 is_node3_TF1_TF2_sign = "???"
# else:
nb_plot = len(compiled_res.index)
node_i_list = compiled_res['node'].values
print(node_i_list)
print(compiled_res)
full_first_layer = False
if nb_plot>0:
    TF_1 = compiled_res[compiled_res['node']==node_i_list[0]]['TF'].values[0]
    lim_val_1 = compiled_res[compiled_res['node']==node_i_list[0]]['lim'].values[0]
    is_node1_sign = None
    if (compiled_res[compiled_res['node']==node_i_list[0]]['p-val_TF1']<0.05).to_numpy()[0]:
        is_node1_sign = '*'
        if (compiled_res[compiled_res['node']==node_i_list[0]]['p-val_TF1']<0.01).to_numpy()[0]:
            is_node1_sign = '**'
            if (compiled_res[compiled_res['node']==node_i_list[0]]['p-val_TF1']<0.001).to_numpy()[0]:
                is_node1_sign = '***'
    if nb_plot>1:
        show_first_layer_res = True
        TF_2 = compiled_res[compiled_res['node']==node_i_list[1]]['TF'].values[0]
        lim_val_2 = compiled_res[compiled_res['node']==node_i_list[1]]['lim'].values[0]
        is_node2_pp_pm_sign = None
        is_node2_pp_mp_sign = None
        is_node2_pp_mm_sign = None
        is_node2_pm_mp_sign = None
        is_node2_pm_mm_sign = None
        is_node2_mp_mm_sign = None

        if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_+-']<0.05).to_numpy()[0]:
            is_node2_pp_pm_sign = '*'
            if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_+-']<0.01).to_numpy()[0]:
                is_node2_pp_pm_sign = '**'
                if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_+-']<0.001).to_numpy()[0]:
                    is_node2_pp_pm_sign = '***'
        if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_-+']<0.05).to_numpy()[0]:
            is_node2_pp_mp_sign = '*'
            if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_-+']<0.01).to_numpy()[0]:
                is_node2_pp_mp_sign = '**'
                if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_-+']<0.001).to_numpy()[0]:
                    is_node2_pp_mp_sign = '***'
        if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_--']<0.05).to_numpy()[0]:
            is_node2_pp_mm_sign = '*'
            if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_--']<0.01).to_numpy()[0]:
                is_node2_pp_mm_sign = '**'
                if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_++_--']<0.001).to_numpy()[0]:
                    is_node2_pp_mm_sign = '***'
        if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_+-_-+']<0.05).to_numpy()[0]:
            is_node2_pm_mp_sign = '*'
            if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_+-_-+']<0.01).to_numpy()[0]:
                is_node2_pm_mp_sign = '**'
                if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_+-_-+']<0.001).to_numpy()[0]:
                    is_node2_pm_mp_sign = '***'
        if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_+-_--']<0.05).to_numpy()[0]:
            is_node2_pm_mm_sign = '*'
            if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_+-_--']<0.01).to_numpy()[0]:
                is_node2_pm_mm_sign = '**'
                if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_+-_--']<0.001).to_numpy()[0]:
                    is_node2_pm_mm_sign = '***'
        if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_-+_--']<0.05).to_numpy()[0]:
            is_node2_mp_mm_sign = '*'
            if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_-+_--']<0.01).to_numpy()[0]:
                is_node2_mp_mm_sign = '**'
                if (compiled_res[compiled_res['node']==node_i_list[1]]['p-val_-+_--']<0.001).to_numpy()[0]:
                    is_node2_mp_mm_sign = '***'

        
        if nb_plot>2:
            full_first_layer = True
            TF_3 = compiled_res[compiled_res['node']==node_i_list[2]]['TF'].values[0]
            lim_val_3 = compiled_res[compiled_res['node']==node_i_list[2]]['lim'].values[0]
            is_node3_pp_pm_sign = None
            is_node3_pp_mp_sign = None
            is_node3_pp_mm_sign = None
            is_node3_pm_mp_sign = None
            is_node3_pm_mm_sign = None
            is_node3_mp_mm_sign = None
            
            if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_+-']<0.05).to_numpy()[0]:
                is_node3_pp_pm_sign = '*'
                if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_+-']<0.01).to_numpy()[0]:
                    is_node3_pp_pm_sign = '**'
                    if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_+-']<0.001).to_numpy()[0]:
                        is_node3_pp_pm_sign = '***'
            if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_-+']<0.05).to_numpy()[0]:
                is_node3_pp_mp_sign = '*'
                if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_-+']<0.01).to_numpy()[0]:
                    is_node3_pp_mp_sign = '**'
                    if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_-+']<0.001).to_numpy()[0]:
                        is_node3_pp_mp_sign = '***'
            if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_--']<0.05).to_numpy()[0]:
                is_node3_pp_mm_sign = '*'
                if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_--']<0.01).to_numpy()[0]:
                    is_node3_pp_mm_sign = '**'
                    if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_++_--']<0.001).to_numpy()[0]:
                        is_node3_pp_mm_sign = '***'
            if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_+-_-+']<0.05).to_numpy()[0]:
                is_node3_pm_mp_sign = '*'
                if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_+-_-+']<0.01).to_numpy()[0]:
                    is_node3_pm_mp_sign = '**'
                    if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_+-_-+']<0.001).to_numpy()[0]:
                        is_node3_pm_mp_sign = '***'
            if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_+-_--']<0.05).to_numpy()[0]:
                is_node3_pm_mm_sign = '*'
                if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_+-_--']<0.01).to_numpy()[0]:
                    is_node3_pm_mm_sign = '**'
                    if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_+-_--']<0.001).to_numpy()[0]:
                        is_node3_pm_mm_sign = '***'
            if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_-+_--']<0.05).to_numpy()[0]:
                is_node3_mp_mm_sign = '*'
                if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_-+_--']<0.01).to_numpy()[0]:
                    is_node3_mp_mm_sign = '**'
                    if (compiled_res[compiled_res['node']==node_i_list[2]]['p-val_-+_--']<0.001).to_numpy()[0]:
                        is_node3_mp_mm_sign = '***'

if plt_config==1:
    figure, axis = plt.subplots(nb_plot, 2)
    axis = axis.ravel()

# TF 1 
TF_1_sup = Y.loc[X[TF_1] > lim_val_1, TG]
TF_1_inf = Y.loc[X[TF_1] <= lim_val_1, TG]
TF_1_sup_norm = np.nan_to_num(np.log10(TF_1_sup.values), neginf=0)
TF_1_inf_norm = np.nan_to_num(np.log10(TF_1_inf.values), neginf=0)

if not show_zero:
    TF_1_sup_norm[TF_1_sup_norm==0] = np.nan
    TF_1_inf_norm[TF_1_inf_norm==0] = np.nan

sns.histplot([TF_1_inf_norm, TF_1_sup_norm][::-1], 
                kde=kde, legend=False, ax=axis[0], stat="percent", common_norm=common_norm, element="bars", multiple=histplot_multiple, binwidth=histplot_binwidth, palette=colors_palette[:2][::-1])#, binwidth=0.1)
axis[0].legend(title='TF', labels=[TF_1 + '<= ' + str(lim_val_1) + "\n(" + str(len(TF_1_inf.index)) + " sample)", TF_1 + '> ' + str(lim_val_1) + "\n(" + str(len(TF_1_sup.index)) + " sample)"])
sns.violinplot(data=[TF_1_inf, TF_1_sup], 
                ax = axis[1], 
                showmeans=False, 
                showmedians=False, 
                showextrema=False)
axis[1].boxplot(
    [TF_1_inf, TF_1_sup],
    positions=[0, 1],
    showfliers = True,
    showcaps = True, labels=None)
# TODO add adaptative height
if is_node1_sign!=None:
    x1, x2 = 0, 1
    y, h, col = np.max([TF_1_inf.max(), TF_1_sup.max()]) + 2, 2, 'k'
    axis[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    axis[1].text((x1+x2)*.5, y+h, is_node1_sign, ha='center', va=va_text_sign, color=col)

axis[1].set_xticklabels(['', '', TF_1+'<=' + str(lim_val_1), TF_1+'>' + str(lim_val_1)])

if show_first_layer_res:
    # TF1 + TF2

    TF_1_sup_TF_2_sup = Y.loc[(X[TF_1] > lim_val_1) & (X[TF_2] > lim_val_2), TG]
    TF_1_sup_TF_2_inf = Y.loc[(X[TF_1] > lim_val_1) & (X[TF_2] <= lim_val_2), TG]
    TF_1_inf_TF_2_sup = Y.loc[(X[TF_1] <= lim_val_1) & (X[TF_2] > lim_val_2), TG]
    TF_1_inf_TF_2_inf = Y.loc[(X[TF_1] <= lim_val_1) & (X[TF_2] <= lim_val_2), TG]

    TF_1_sup_TF_2_sup_norm = np.nan_to_num(np.log10(TF_1_sup_TF_2_sup), neginf=0)
    TF_1_sup_TF_2_inf_norm = np.nan_to_num(np.log10(TF_1_sup_TF_2_inf), neginf=0)
    TF_1_inf_TF_2_sup_norm = np.nan_to_num(np.log10(TF_1_inf_TF_2_sup), neginf=0)
    TF_1_inf_TF_2_inf_norm = np.nan_to_num(np.log10(TF_1_inf_TF_2_inf), neginf=0)

    if not show_zero:
        TF_1_sup_TF_2_sup_norm[TF_1_sup_TF_2_sup_norm==0] = np.nan
        TF_1_sup_TF_2_inf_norm[TF_1_sup_TF_2_inf_norm==0] = np.nan
        TF_1_inf_TF_2_sup_norm[TF_1_inf_TF_2_sup_norm==0] = np.nan
        TF_1_inf_TF_2_inf_norm[TF_1_inf_TF_2_inf_norm==0] = np.nan

    sns.histplot(data = [TF_1_inf_TF_2_inf_norm, TF_1_inf_TF_2_sup_norm, TF_1_sup_TF_2_inf_norm, TF_1_sup_TF_2_sup_norm][::-1],
                kde=kde, legend=False, ax=axis[2], stat="percent", common_norm=common_norm, element="bars", multiple=histplot_multiple, binwidth=histplot_binwidth, palette=colors_palette[::-1])
    axis[2].legend(title='TF', labels=[
                                        TF_1 + '-' + TF_2 + '-' + "\n(" + str(len(TF_1_sup_TF_2_sup.index)) + " sample)", 
                                        TF_1 + '-' + TF_2 + '+' + "\n(" + str(len(TF_1_sup_TF_2_inf.index)) + " sample)", 
                                        TF_1 + '+' + TF_2 + '-' + "\n(" + str(len(TF_1_inf_TF_2_sup.index)) + " sample)", 
                                        TF_1 + '+' + TF_2 + '+' + "\n(" + str(len(TF_1_inf_TF_2_inf.index)) + " sample)"])
    sns.violinplot([TF_1_inf_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_sup_TF_2_sup], 
                    ax = axis[3], 
                    showmeans=True, 
                    showmedians=True, 
                    showextrema=True)
    axis[3].boxplot(
        [TF_1_inf_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_sup_TF_2_sup],
        positions=[0, 1, 2, 3],
        showfliers = True, # Do not show the outliers beyond the caps.
        showcaps = True, labels=None)   # Do not show the caps
    
    max_val = np.max([TF_1_inf_TF_2_inf.max() if len(TF_1_inf_TF_2_inf.index)>0 else 0, 
                        TF_1_inf_TF_2_sup.max() if len(TF_1_inf_TF_2_sup.index)>0 else 0, 
                        TF_1_sup_TF_2_inf.max() if len(TF_1_sup_TF_2_inf.index)>0 else 0, 
                        TF_1_sup_TF_2_sup.max() if len(TF_1_sup_TF_2_sup.index)>0 else 0])
    height_p_sign = int(max_val/ratio_sign)
    y, h, col = max_val + 2, height_p_sign, 'k'
    y_add = 1
    if is_node2_pp_pm_sign!=None:
        x1, x2 = 2, 3
        axis[3].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        axis[3].text((x1+x2)*.5, y+h, is_node2_pp_pm_sign, ha='center', va=va_text_sign, color=col)
        y_add = 1
    if is_node2_mp_mm_sign!=None:
        x1, x2 = 0, 1
        axis[3].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        axis[3].text((x1+x2)*.5, y+h, is_node2_mp_mm_sign, ha='center', va=va_text_sign, color=col)
        y_add = 1
    
    if is_node2_pp_mp_sign!=None:
        y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
        x1, x2 = 1, 3
        axis[3].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        axis[3].text((x1+x2)*.5, y+h, is_node2_pp_mp_sign, ha='center', va=va_text_sign, color=col)
        y_add += 1
    
    if is_node2_pm_mm_sign!=None:
        y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
        x1, x2 = 0, 2
        axis[3].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        axis[3].text((x1+x2)*.5, y+h, is_node2_pm_mm_sign, ha='center', va=va_text_sign, color=col)
        y_add += 1
    
    if is_node2_pm_mp_sign!=None:
        y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
        x1, x2 = 1, 2
        axis[3].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        axis[3].text((x1+x2)*.5, y+h, is_node2_pm_mp_sign, ha='center', va=va_text_sign, color=col)
        y_add += 1
    
    if is_node2_pp_mm_sign!=None:
        y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
        x1, x2 = 0, 3
        axis[3].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        axis[3].text((x1+x2)*.5, y+h, is_node2_pp_mm_sign, ha='center', va=va_text_sign, color=col)
        y_add += 1
    axis[3].set_xticklabels(['', '', '', '', TF_1 + '-' + TF_2 + '-', TF_1 + '-' + TF_2 + '+', TF_1 + '+' + TF_2 + '-', TF_1+'+' + TF_2 + '+'])
    if full_first_layer:
        # TF1 + TF3

        TF_1_sup_TF_3_sup = Y.loc[(X[TF_1] > lim_val_1) & (X[TF_3] > lim_val_3), TG]
        TF_1_sup_TF_3_inf = Y.loc[(X[TF_1] > lim_val_1) & (X[TF_3] <= lim_val_3), TG]
        TF_1_inf_TF_3_sup = Y.loc[(X[TF_1] <= lim_val_1) & (X[TF_3] > lim_val_3), TG]
        TF_1_inf_TF_3_inf = Y.loc[(X[TF_1] <= lim_val_1) & (X[TF_3] <= lim_val_3), TG]

        TF_1_sup_TF_3_sup_norm = np.nan_to_num(np.log10(TF_1_sup_TF_3_sup), neginf=0)
        TF_1_sup_TF_3_inf_norm = np.nan_to_num(np.log10(TF_1_sup_TF_3_inf), neginf=0)
        TF_1_inf_TF_3_sup_norm = np.nan_to_num(np.log10(TF_1_inf_TF_3_sup), neginf=0)
        TF_1_inf_TF_3_inf_norm = np.nan_to_num(np.log10(TF_1_inf_TF_3_inf), neginf=0)

        if not show_zero:
            TF_1_sup_TF_3_sup_norm[TF_1_sup_TF_3_sup_norm==0] = np.nan
            TF_1_sup_TF_3_inf_norm[TF_1_sup_TF_3_inf_norm==0] = np.nan
            TF_1_inf_TF_3_sup_norm[TF_1_inf_TF_3_sup_norm==0] = np.nan
            TF_1_inf_TF_3_inf_norm[TF_1_inf_TF_3_inf_norm==0] = np.nan

        sns.histplot(data = [TF_1_inf_TF_3_inf_norm, TF_1_inf_TF_3_sup_norm, TF_1_sup_TF_3_inf_norm, TF_1_sup_TF_3_sup_norm][::-1],
                    kde=kde, legend=False, ax=axis[4], stat="percent", common_norm=common_norm, element="bars", multiple=histplot_multiple, binwidth=histplot_binwidth, palette=colors_palette[::-1])
        axis[4].legend(title='TF', labels=[
                                            TF_1 + '-' + TF_3 + '-' + "\n(" + str(len(TF_1_sup_TF_3_sup.index)) + " sample)", 
                                            TF_1 + '-' + TF_3 + '+' + "\n(" + str(len(TF_1_sup_TF_3_inf.index)) + " sample)", 
                                            TF_1 + '+' + TF_3 + '-' + "\n(" + str(len(TF_1_inf_TF_3_sup.index)) + " sample)", 
                                            TF_1 + '+' + TF_3 + '+' + "\n(" + str(len(TF_1_inf_TF_3_inf.index)) + " sample)"])
        sns.violinplot([TF_1_inf_TF_3_inf, TF_1_inf_TF_3_sup, TF_1_sup_TF_3_inf, TF_1_sup_TF_3_sup], ax = axis[5], showmeans=True, showmedians=True, showextrema=True)
        axis[5].boxplot(
            [TF_1_inf_TF_3_inf, TF_1_inf_TF_3_sup, TF_1_sup_TF_3_inf, TF_1_sup_TF_3_sup],
            positions=[0, 1, 2, 3],
            showfliers = True, # Do not show the outliers beyond the caps.
            showcaps = True, labels=None)   # Do not show the caps
        print(TF_1_inf_TF_3_sup)
        print(TF_1_inf_TF_3_sup.max())
        max_val = np.max([TF_1_inf_TF_3_inf.max() if len(TF_1_inf_TF_3_inf.index)>0 else 0, 
                            TF_1_inf_TF_3_sup.max() if len(TF_1_inf_TF_3_sup.index)>0 else 0, 
                            TF_1_sup_TF_3_inf.max() if len(TF_1_sup_TF_3_inf.index)>0 else 0, 
                            TF_1_sup_TF_3_sup.max() if len(TF_1_sup_TF_3_sup.index)>0 else 0])
        print(max_val)
        height_p_sign = int(max_val/ratio_sign)
        y, h, col = max_val + 2, height_p_sign, 'k'
        y_add = 1
        if is_node3_pp_pm_sign!=None:
            x1, x2 = 2, 3
            axis[5].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[5].text((x1+x2)*.5, y+h, is_node3_pp_pm_sign, ha='center', va=va_text_sign, color=col)
            y_add = 2
        if is_node3_mp_mm_sign!=None:
            x1, x2 = 0, 1
            axis[5].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[5].text((x1+x2)*.5, y+h, is_node3_mp_mm_sign, ha='center', va=va_text_sign, color=col)
            y_add = 2
        
        if is_node3_pp_mp_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 1, 3
            axis[5].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[5].text((x1+x2)*.5, y+h, is_node3_pp_mp_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        
        if is_node3_pm_mm_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 0, 2
            axis[5].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[5].text((x1+x2)*.5, y+h, is_node3_pm_mm_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        
        if is_node3_pm_mp_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 1, 2
            axis[5].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[5].text((x1+x2)*.5, y+h, is_node3_pm_mp_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        
        if is_node3_pp_mm_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 0, 3
            axis[5].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[5].text((x1+x2)*.5, y+h, is_node3_pp_mm_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        axis[5].set_xticklabels(['', '', '', '', TF_1 + '-' + TF_3 + '-', TF_1 + '-' + TF_3 + '+', TF_1 + '+' + TF_3 + '-', TF_1+'+' + TF_3 + '+'])
plt.suptitle(str(TG))
plt.show()