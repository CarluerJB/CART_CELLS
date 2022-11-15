from os import stat
from random import seed
from turtle import width
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

# THIS SCRIPT NEED 3 ARGS
# TF
# TF
# LIM_VAL
show_first_layer_res = False
full_first_layer = True
kde = True
common_norm = False
log_y_axis = False
histplot_multiple = "dodge"
histplot_binwidth=0.2
colors_palette = sns.color_palette("tab10", 4)
ratio_sign = 10
show_zero = False
va_text_sign = 'top'
run_UMAP = False
cell_width = 212
cell_height = 22
swatch_width = 48
xaxis_label_size = 8
plot_method = "print" # Choice between "print" and "show"

#data = "/media/carluerj/Data/These/DATA/gene_regulator_network/20K_prefiltred_contrib_filter_23_TF_002.txt"
data = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"
out_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_005/"
compiled_data = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_005/"
embedded_data_path = "/media/carluerj/Data/These/Results/GRN_inference/list_gene_BF/UMAP_embedding.npy"

# Read data
df = pd.read_table(data, sep="\t", header=0)
X = df.loc[df.index[-23:]].transpose() # TF
Y = df.drop([*df.index[-23:]]).transpose() # AGI
mean = Y.mean()
std = Y.std()
Y_norm = Y-mean/std
ind_dict = dict((k, i) for i, k in enumerate(Y_norm.index.values))
compiled_res = pd.read_table(compiled_data + "Final_score_table_MANNWHIT_TEST_FINAL_byline.txt", sep=",", header=0, na_values="NaN")

# Load parameter based on AGI
TG = sys.argv[1]
if TG not in Y.columns:
    with open(out_data + "not_found_candidates.txt", "a") as file:
        file.write(TG + "\n")
    exit(0)
compiled_res = compiled_res.loc[(compiled_res['AGI']==TG)]

# RUN UMAP
if run_UMAP:
    model = UMAP(n_components=2, min_dist=0.3, n_neighbors=30, metric="correlation")
    embedded_data = model.fit_transform(Y_norm)
    np.save(embedded_data_path, embedded_data)
else:
    embedded_data = np.load(embedded_data_path)

nb_plot = pd.notna(compiled_res[['TF1','TF2','TF3']]).sum().sum()
full_first_layer = False
if nb_plot>0:
    TF_1 = compiled_res['TF1'].values[0]
    lim_val_1 = compiled_res['lim1'].values[0]
    is_node1_sign = None
    if (compiled_res['p-val_TF1']<0.05).to_numpy()[0]:
        is_node1_sign = '*'
        if (compiled_res['p-val_TF1']<0.01).to_numpy()[0]:
            is_node1_sign = '**'
            if (compiled_res['p-val_TF1']<0.001).to_numpy()[0]:
                is_node1_sign = '***'
    if nb_plot>1:
        show_first_layer_res = True
        TF_2 = compiled_res['TF2'].values[0]
        lim_val_2 = compiled_res['lim2'].values[0]
        is_node2_pp_pm_sign = None
        is_node2_pp_mp_sign = None
        is_node2_pp_mm_sign = None
        is_node2_pm_mp_sign = None
        is_node2_pm_mm_sign = None
        is_node2_mp_mm_sign = None
        print(TF_1)
        print(lim_val_1)
        print(TF_2)
        print(lim_val_2)
        if (compiled_res['N2_p-val_++_+-']<0.05).to_numpy()[0]:
            is_node2_pp_pm_sign = '*'
            if (compiled_res['N2_p-val_++_+-']<0.01).to_numpy()[0]:
                is_node2_pp_pm_sign = '**'
                if (compiled_res['N2_p-val_++_+-']<0.001).to_numpy()[0]:
                    is_node2_pp_pm_sign = '***'
        if (compiled_res['N2_p-val_++_-+']<0.05).to_numpy()[0]:
            is_node2_pp_mp_sign = '*'
            if (compiled_res['N2_p-val_++_-+']<0.01).to_numpy()[0]:
                is_node2_pp_mp_sign = '**'
                if (compiled_res['N2_p-val_++_-+']<0.001).to_numpy()[0]:
                    is_node2_pp_mp_sign = '***'
        if (compiled_res['N2_p-val_++_--']<0.05).to_numpy()[0]:
            is_node2_pp_mm_sign = '*'
            if (compiled_res['N2_p-val_++_--']<0.01).to_numpy()[0]:
                is_node2_pp_mm_sign = '**'
                if (compiled_res['N2_p-val_++_--']<0.001).to_numpy()[0]:
                    is_node2_pp_mm_sign = '***'
        if (compiled_res['N2_p-val_+-_-+']<0.05).to_numpy()[0]:
            is_node2_pm_mp_sign = '*'
            if (compiled_res['N2_p-val_+-_-+']<0.01).to_numpy()[0]:
                is_node2_pm_mp_sign = '**'
                if (compiled_res['N2_p-val_+-_-+']<0.001).to_numpy()[0]:
                    is_node2_pm_mp_sign = '***'
        if (compiled_res['N2_p-val_+-_--']<0.05).to_numpy()[0]:
            is_node2_pm_mm_sign = '*'
            if (compiled_res['N2_p-val_+-_--']<0.01).to_numpy()[0]:
                is_node2_pm_mm_sign = '**'
                if (compiled_res['N2_p-val_+-_--']<0.001).to_numpy()[0]:
                    is_node2_pm_mm_sign = '***'
        if (compiled_res['N2_p-val_-+_--']<0.05).to_numpy()[0]:
            is_node2_mp_mm_sign = '*'
            if (compiled_res['N2_p-val_-+_--']<0.01).to_numpy()[0]:
                is_node2_mp_mm_sign = '**'
                if (compiled_res['N2_p-val_-+_--']<0.001).to_numpy()[0]:
                    is_node2_mp_mm_sign = '***'

        
        if nb_plot>2:
            full_first_layer = True
            TF_3 = compiled_res['TF3'].values[0]
            lim_val_3 = compiled_res['lim3'].values[0]
            is_node3_pp_pm_sign = None
            is_node3_pp_mp_sign = None
            is_node3_pp_mm_sign = None
            is_node3_pm_mp_sign = None
            is_node3_pm_mm_sign = None
            is_node3_mp_mm_sign = None
            
            if (compiled_res['N3_p-val_++_+-']<0.05).to_numpy()[0]:
                is_node3_pp_pm_sign = '*'
                if (compiled_res['N3_p-val_++_+-']<0.01).to_numpy()[0]:
                    is_node3_pp_pm_sign = '**'
                    if (compiled_res['N3_p-val_++_+-']<0.001).to_numpy()[0]:
                        is_node3_pp_pm_sign = '***'
            if (compiled_res['N3_p-val_++_-+']<0.05).to_numpy()[0]:
                is_node3_pp_mp_sign = '*'
                if (compiled_res['N3_p-val_++_-+']<0.01).to_numpy()[0]:
                    is_node3_pp_mp_sign = '**'
                    if (compiled_res['N3_p-val_++_-+']<0.001).to_numpy()[0]:
                        is_node3_pp_mp_sign = '***'
            if (compiled_res['N3_p-val_++_--']<0.05).to_numpy()[0]:
                is_node3_pp_mm_sign = '*'
                if (compiled_res['N3_p-val_++_--']<0.01).to_numpy()[0]:
                    is_node3_pp_mm_sign = '**'
                    if (compiled_res['N3_p-val_++_--']<0.001).to_numpy()[0]:
                        is_node3_pp_mm_sign = '***'
            if (compiled_res['N3_p-val_+-_-+']<0.05).to_numpy()[0]:
                is_node3_pm_mp_sign = '*'
                if (compiled_res['N3_p-val_+-_-+']<0.01).to_numpy()[0]:
                    is_node3_pm_mp_sign = '**'
                    if (compiled_res['N3_p-val_+-_-+']<0.001).to_numpy()[0]:
                        is_node3_pm_mp_sign = '***'
            if (compiled_res['N3_p-val_+-_--']<0.05).to_numpy()[0]:
                is_node3_pm_mm_sign = '*'
                if (compiled_res['N3_p-val_+-_--']<0.01).to_numpy()[0]:
                    is_node3_pm_mm_sign = '**'
                    if (compiled_res['N3_p-val_+-_--']<0.001).to_numpy()[0]:
                        is_node3_pm_mm_sign = '***'
            if (compiled_res['N3_p-val_-+_--']<0.05).to_numpy()[0]:
                is_node3_mp_mm_sign = '*'
                if (compiled_res['N3_p-val_-+_--']<0.01).to_numpy()[0]:
                    is_node3_mp_mm_sign = '**'
                    if (compiled_res['N3_p-val_-+_--']<0.001).to_numpy()[0]:
                        is_node3_mp_mm_sign = '***'


figure, axis = plt.subplots(nb_plot, 4, figsize=(15, 10))
axis = axis.ravel()
# TF 1 
TF_1_sup = Y.loc[X[TF_1] > lim_val_1, TG]
TF_1_inf = Y.loc[X[TF_1] <= lim_val_1, TG]
TF_1_sup_norm = np.nan_to_num(np.log10(TF_1_sup.values), neginf=0)
TF_1_inf_norm = np.nan_to_num(np.log10(TF_1_inf.values), neginf=0)

if not show_zero:
    TF_1_sup_norm[TF_1_sup==0] = np.nan
    TF_1_inf_norm[TF_1_inf==0] = np.nan
else:
    TF_1_sup_norm = TF_1_sup_norm + 1
    TF_1_inf_norm = TF_1_inf_norm + 1

sns.histplot([TF_1_inf_norm if len(TF_1_inf_norm)>0 else np.nan, TF_1_sup_norm if len(TF_1_sup_norm)>0 else np.nan][::-1], 
                kde=kde, legend=True, ax=axis[0], stat="percent", common_norm=common_norm, element="bars", multiple=histplot_multiple, binwidth=histplot_binwidth, palette=colors_palette[:2][::-1])#, binwidth=0.1)
# axis[0].legend(title='TF', labels=[TF_1 + '<= ' + str(lim_val_1) + "\n(" + str(len(TF_1_inf.index)) + " samples)", TF_1 + '> ' + str(lim_val_1) + "\n(" + str(len(TF_1_sup.index)) + " samples)"])
legend = axis[0].get_legend()
handles = legend.legendHandles
legend.remove()
label_list = []
handle_list = []
label_list.append(TF_1 + ' - ' + "\n(" + str(len(TF_1_inf.index)) + " samples)")
label_list.append(TF_1 + ' + ' + "\n(" + str(len(TF_1_sup.index)) + " samples)")
for i, elem in enumerate(label_list):
    row = i
    y = row * cell_height
    swatch_start_x = cell_width
    text_pos_x = cell_width + swatch_width + 7
    axis[0].text(text_pos_x, y, elem, fontsize=14,
        horizontalalignment='left',
        verticalalignment='center')
    handle_list.append(Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                height=18, facecolor=colors_palette[i], edgecolor='0.7'))
axis[0].legend(handles=handle_list, labels=label_list)


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

if log_y_axis:
    axis[1].set(yscale="symlog")
# TODO add adaptative height
if is_node1_sign!=None:
    x1, x2 = 0, 1
    y, h, col = axis[1].get_ylim()[1] + 2, 2, 'k'
    axis[1].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    axis[1].text((x1+x2)*.5, y+h, is_node1_sign, ha='center', va=va_text_sign, color=col)

axis[1].set_xticklabels(['', '', TF_1+' - ', TF_1+' + '])
axis[1].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)

axis[2].boxplot(
    [TF_1_inf, TF_1_sup],
    positions=[0, 1],
    showfliers = False,
    showcaps = True, labels=None)
if log_y_axis:
    axis[2].set(yscale="symlog")

axis[2].set_xticklabels([TF_1+' - ', TF_1+' + '])
axis[2].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)

# UMAP
cell_sup = list(set(Y_norm.index.values).intersection(set(TF_1_sup.index.values)))
cell_inf = list(set(Y_norm.index.values).intersection(set(TF_1_inf.index.values)))

i_cell_sup = [ind_dict[x] for x in cell_sup]
i_cell_inf = [ind_dict[x] for x in cell_inf]

# plt.scatter(embedded_data[:,0], embedded_data[:,1], s=5, c="green", marker='.')
axis[3].scatter(x=embedded_data[:,0][i_cell_sup], y=embedded_data[:,1][i_cell_sup], s=3, color=colors_palette[:2][::-1][0], marker='.')
axis[3].scatter(x=embedded_data[:,0][i_cell_inf], y=embedded_data[:,1][i_cell_inf], s=3, color=colors_palette[:2][::-1][1], marker='.')

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
        TF_1_sup_TF_2_sup_norm[TF_1_sup_TF_2_sup==0] = np.nan
        TF_1_sup_TF_2_inf_norm[TF_1_sup_TF_2_inf==0] = np.nan
        TF_1_inf_TF_2_sup_norm[TF_1_inf_TF_2_sup==0] = np.nan
        TF_1_inf_TF_2_inf_norm[TF_1_inf_TF_2_inf==0] = np.nan
    else:
        TF_1_sup_TF_2_sup_norm = TF_1_sup_TF_2_sup_norm + 1
        TF_1_sup_TF_2_inf_norm = TF_1_sup_TF_2_inf_norm + 1
        TF_1_inf_TF_2_sup_norm = TF_1_inf_TF_2_sup_norm + 1
        TF_1_inf_TF_2_inf_norm = TF_1_inf_TF_2_inf_norm + 1

    sns.histplot(data = [TF_1_inf_TF_2_inf_norm, TF_1_inf_TF_2_sup_norm, TF_1_sup_TF_2_inf_norm, TF_1_sup_TF_2_sup_norm][::-1],
                kde=kde, legend=True, ax=axis[4], stat="percent", common_norm=common_norm, element="bars", multiple=histplot_multiple, binwidth=histplot_binwidth, palette=colors_palette[::-1])
    legend = axis[4].get_legend()
    handles = legend.legendHandles
    legend.remove()
    label_list = []
    handle_list = []
    label_list.append(TF_1 + ' - ' + TF_2 + ' - ' + "\n(" + str(len(TF_1_inf_TF_2_inf.index)) + " samples)")
    label_list.append(TF_1 + ' - ' + TF_2 + ' + ' + "\n(" + str(len(TF_1_inf_TF_2_sup.index)) + " samples)")
    label_list.append(TF_1 + ' + ' + TF_2 + ' - ' + "\n(" + str(len(TF_1_sup_TF_2_inf.index)) + " samples)")
    label_list.append(TF_1 + ' + ' + TF_2 + ' + ' + "\n(" + str(len(TF_1_sup_TF_2_sup.index)) + " samples)")
    for i, elem in enumerate(label_list):
        row = i
        y = row * cell_height
        swatch_start_x = cell_width
        text_pos_x = cell_width + swatch_width + 7
        axis[4].text(text_pos_x, y, elem, fontsize=14,
            horizontalalignment='left',
            verticalalignment='center')
        handle_list.append(Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                    height=18, facecolor=colors_palette[i], edgecolor='0.7'))
    axis[4].legend(handles=handle_list, labels=label_list)
    # axis[4].legend(title='TF', labels=[
    #                                     TF_1 + '-' + TF_2 + '-' + "\n(" + str(len(TF_1_sup_TF_2_sup.index)) + " samples)", 
    #                                     TF_1 + '-' + TF_2 + '+' + "\n(" + str(len(TF_1_sup_TF_2_inf.index)) + " samples)", 
    #                                     TF_1 + '+' + TF_2 + '-' + "\n(" + str(len(TF_1_inf_TF_2_sup.index)) + " samples)", 
    #                                     TF_1 + '+' + TF_2 + '+' + "\n(" + str(len(TF_1_inf_TF_2_inf.index)) + " samples)"])
    sns.violinplot([TF_1_inf_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_sup_TF_2_sup], 
                    ax = axis[5], 
                    showmeans=True, 
                    showmedians=True, 
                    showextrema=True)
    axis[5].boxplot(
        [TF_1_inf_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_sup_TF_2_sup],
        positions=[0, 1, 2, 3],
        showfliers = True, # Do not show the outliers beyond the caps.
        showcaps = True, labels=None)   # Do not show the caps
    if log_y_axis:
        axis[5].set(yscale="symlog")
    
    
    max_val_ax5 = axis[5].get_ylim()[1]
    height_p_sign_ax5 = int(max_val_ax5/ratio_sign)
    y_ax5, h_ax5, col_ax5 = max_val_ax5 + 2, height_p_sign_ax5, 'k'

    axis[6].boxplot(
        [TF_1_inf_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_sup_TF_2_sup],
        positions=[0, 1, 2, 3],
        showfliers = False, # Do not show the outliers beyond the caps.
        showcaps = True, labels=None)   # Do not show the caps
    if log_y_axis:
        axis[6].set(yscale="symlog")
    


    y_add = 1
    if is_node2_pp_pm_sign!=None:
        x1, x2 = 2, 3
        axis[5].plot([x1, x1, x2, x2], [y_ax5, y_ax5+h_ax5, y_ax5+h_ax5, y_ax5], lw=1.5, c=col_ax5)
        axis[5].text((x1+x2)*.5, y_ax5+h_ax5, is_node2_pp_pm_sign, ha='center', va=va_text_sign, color=col_ax5)
        y_add = 1
    if is_node2_mp_mm_sign!=None:
        x1, x2 = 0, 1
        axis[5].plot([x1, x1, x2, x2], [y_ax5, y_ax5+h_ax5, y_ax5+h_ax5, y_ax5], lw=1.5, c=col_ax5)
        axis[5].text((x1+x2)*.5, y_ax5+h_ax5, is_node2_mp_mm_sign, ha='center', va=va_text_sign, color=col_ax5)
        y_add = 1
    
    if is_node2_pp_mp_sign!=None:
        x1, x2 = 1, 3
        y_ax5, h_ax5, col_ax5 = max_val_ax5 + 2 + (y_add*height_p_sign_ax5) + int((y_add*height_p_sign_ax5)/3), height_p_sign_ax5, 'k'
        axis[5].plot([x1, x1, x2, x2], [y_ax5, y_ax5+h_ax5, y_ax5+h_ax5, y_ax5], lw=1.5, c=col_ax5)
        axis[5].text((x1+x2)*.5, y_ax5+h_ax5, is_node2_pp_mp_sign, ha='center', va=va_text_sign, color=col_ax5)
        y_add += 1
    
    if is_node2_pm_mm_sign!=None:
        x1, x2 = 0, 2
        y_ax5, h_ax5, col_ax5 = max_val_ax5 + 2 + (y_add*height_p_sign_ax5) + int((y_add*height_p_sign_ax5)/3), height_p_sign_ax5, 'k'
        axis[5].plot([x1, x1, x2, x2], [y_ax5, y_ax5+h_ax5, y_ax5+h_ax5, y_ax5], lw=1.5, c=col_ax5)
        axis[5].text((x1+x2)*.5, y_ax5+h_ax5, is_node2_pm_mm_sign, ha='center', va=va_text_sign, color=col_ax5)
        y_add += 1
    
    if is_node2_pm_mp_sign!=None:
        x1, x2 = 1, 2
        y_ax5, h_ax5, col_ax5 = max_val_ax5 + 2 + (y_add*height_p_sign_ax5) + int((y_add*height_p_sign_ax5)/3), height_p_sign_ax5, 'k'
        axis[5].plot([x1, x1, x2, x2], [y_ax5, y_ax5+h_ax5, y_ax5+h_ax5, y_ax5], lw=1.5, c=col_ax5)
        axis[5].text((x1+x2)*.5, y_ax5+h_ax5, is_node2_pm_mp_sign, ha='center', va=va_text_sign, color=col_ax5)
        y_add += 1
    
    if is_node2_pp_mm_sign!=None:
        x1, x2 = 0, 3
        y_ax5, h_ax5, col_ax5 = max_val_ax5 + 2 + (y_add*height_p_sign_ax5) + int((y_add*height_p_sign_ax5)/3), height_p_sign_ax5, 'k'
        axis[5].plot([x1, x1, x2, x2], [y_ax5, y_ax5+h_ax5, y_ax5+h_ax5, y_ax5], lw=1.5, c=col_ax5)
        axis[5].text((x1+x2)*.5, y_ax5+h_ax5, is_node2_pp_mm_sign, ha='center', va=va_text_sign, color=col_ax5)
        y_add += 1
    axis[5].set_xticklabels(['', '', '', '', TF_1 + ' - ' + TF_2 + ' - ', TF_1 + ' - ' + TF_2 + ' + ', TF_1 + ' + ' + TF_2 + ' - ', TF_1+' + ' + TF_2 + ' + '])
    axis[5].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
    axis[6].set_xticklabels([TF_1 + ' - ' + TF_2 + ' - ', TF_1 + ' - ' + TF_2 + ' + ', TF_1 + ' + ' + TF_2 + ' - ', TF_1+' + ' + TF_2 + ' + '])
    axis[6].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
    cell_sup_sup = list(set(Y_norm.index.values).intersection(set(TF_1_sup_TF_2_sup.index.values)))
    cell_sup_inf = list(set(Y_norm.index.values).intersection(set(TF_1_sup_TF_2_inf.index.values)))
    cell_inf_sup = list(set(Y_norm.index.values).intersection(set(TF_1_inf_TF_2_sup.index.values)))
    cell_inf_inf = list(set(Y_norm.index.values).intersection(set(TF_1_inf_TF_2_inf.index.values)))


    i_cell_sup_sup = [ind_dict[x] for x in cell_sup_sup]
    i_cell_sup_inf = [ind_dict[x] for x in cell_sup_inf]
    i_cell_inf_sup = [ind_dict[x] for x in cell_inf_sup]
    i_cell_inf_inf = [ind_dict[x] for x in cell_inf_inf]

    # plt.scatter(embedded_data[:,0], embedded_data[:,1], s=5, c="green", marker='.')
    axis[7].scatter(x=embedded_data[:,0][i_cell_sup_sup], y=embedded_data[:,1][i_cell_sup_sup], s=3, color=colors_palette[::-1][0], marker='.')
    axis[7].scatter(x=embedded_data[:,0][i_cell_sup_inf], y=embedded_data[:,1][i_cell_sup_inf], s=3, color=colors_palette[::-1][1], marker='.')
    axis[7].scatter(x=embedded_data[:,0][i_cell_inf_sup], y=embedded_data[:,1][i_cell_inf_sup], s=3, color=colors_palette[::-1][2], marker='.')
    axis[7].scatter(x=embedded_data[:,0][i_cell_inf_inf], y=embedded_data[:,1][i_cell_inf_inf], s=3, color=colors_palette[::-1][3], marker='.')
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
            TF_1_sup_TF_3_sup_norm[TF_1_sup_TF_3_sup==0] = np.nan
            TF_1_sup_TF_3_inf_norm[TF_1_sup_TF_3_inf==0] = np.nan
            TF_1_inf_TF_3_sup_norm[TF_1_inf_TF_3_sup==0] = np.nan
            TF_1_inf_TF_3_inf_norm[TF_1_inf_TF_3_inf==0] = np.nan
        else:
            TF_1_sup_TF_3_sup_norm = TF_1_sup_TF_3_sup_norm + 1
            TF_1_sup_TF_3_inf_norm = TF_1_sup_TF_3_inf_norm + 1
            TF_1_inf_TF_3_sup_norm = TF_1_inf_TF_3_sup_norm + 1
            TF_1_inf_TF_3_inf_norm = TF_1_inf_TF_3_inf_norm + 1
        
        
        sns.histplot(data = [TF_1_inf_TF_3_inf_norm, TF_1_inf_TF_3_sup_norm, TF_1_sup_TF_3_inf_norm, TF_1_sup_TF_3_sup_norm][::-1],
                    kde=kde, legend=True, ax=axis[8], stat="percent", common_norm=common_norm, element="bars", multiple=histplot_multiple, binwidth=histplot_binwidth, palette=colors_palette[::-1], label="data")
        
        label_list = []
        handle_list = []
        legend = axis[8].get_legend()
        handles = legend.legendHandles
        legend.remove()
        label_list.append(TF_1 + ' - ' + TF_3 + ' - ' + "\n(" + str(len(TF_1_inf_TF_3_inf.index)) + " samples)")
        label_list.append(TF_1 + ' - ' + TF_3 + ' + ' + "\n(" + str(len(TF_1_inf_TF_3_sup.index)) + " samples)")
        label_list.append(TF_1 + ' + ' + TF_3 + ' - ' + "\n(" + str(len(TF_1_sup_TF_3_inf.index)) + " samples)")
        label_list.append(TF_1 + ' + ' + TF_3 + ' + ' + "\n(" + str(len(TF_1_sup_TF_3_sup.index)) + " samples)")
        for i, elem in enumerate(label_list):
            row = i
            y = row * cell_height
            swatch_start_x = cell_width
            text_pos_x = cell_width + swatch_width + 7
            axis[8].text(text_pos_x, y, elem, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')
            handle_list.append(Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors_palette[i], edgecolor='0.7'))
        axis[8].legend(handles=handle_list, labels=label_list)
        
        sns.violinplot([TF_1_inf_TF_3_inf, TF_1_inf_TF_3_sup, TF_1_sup_TF_3_inf, TF_1_sup_TF_3_sup], ax = axis[9], showmeans=True, showmedians=True, showextrema=True)
        axis[9].boxplot(
            [TF_1_inf_TF_3_inf, TF_1_inf_TF_3_sup, TF_1_sup_TF_3_inf, TF_1_sup_TF_3_sup],
            positions=[0, 1, 2, 3],
            showfliers = True, # Do not show the outliers beyond the caps.
            showcaps = True, labels=None)   # Do not show the caps
        if log_y_axis:
            axis[9].set(yscale="symlog")

        axis[10].boxplot(
            [TF_1_inf_TF_3_inf, TF_1_inf_TF_3_sup, TF_1_sup_TF_3_inf, TF_1_sup_TF_3_sup],
            positions=[0, 1, 2, 3],
            showfliers = False, # Do not show the outliers beyond the caps.
            showcaps = True, labels=None)   # Do not show the caps
        if log_y_axis:
            axis[10].set(yscale="symlog")
        

        max_val = axis[9].get_ylim()[1]
        height_p_sign = int(max_val/ratio_sign)
        y, h, col = max_val + 2, height_p_sign, 'k'
        y_add = 1
        if is_node3_pp_pm_sign!=None:
            x1, x2 = 2, 3
            axis[9].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[9].text((x1+x2)*.5, y+h, is_node3_pp_pm_sign, ha='center', va=va_text_sign, color=col)
            y_add = 2
        if is_node3_mp_mm_sign!=None:
            x1, x2 = 0, 1
            axis[9].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[9].text((x1+x2)*.5, y+h, is_node3_mp_mm_sign, ha='center', va=va_text_sign, color=col)
            y_add = 2
        
        if is_node3_pp_mp_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 1, 3
            axis[9].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[9].text((x1+x2)*.5, y+h, is_node3_pp_mp_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        
        if is_node3_pm_mm_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 0, 2
            axis[9].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[9].text((x1+x2)*.5, y+h, is_node3_pm_mm_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        
        if is_node3_pm_mp_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 1, 2
            axis[9].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[9].text((x1+x2)*.5, y+h, is_node3_pm_mp_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        
        if is_node3_pp_mm_sign!=None:
            y, h, col = max_val + 2 + (y_add*height_p_sign) + int((y_add*height_p_sign)/3), height_p_sign, 'k'
            x1, x2 = 0, 3
            axis[9].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            axis[9].text((x1+x2)*.5, y+h, is_node3_pp_mm_sign, ha='center', va=va_text_sign, color=col)
            y_add += 1
        axis[9].set_xticklabels(['', '', '', '', TF_1 + ' - ' + TF_3 + ' - ', TF_1 + ' - ' + TF_3 + ' + ', TF_1 + ' + ' + TF_3 + ' - ', TF_1+' + ' + TF_3 + ' + '])
        axis[9].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
        axis[10].set_xticklabels([TF_1 + ' - ' + TF_3 + ' - ', TF_1 + ' - ' + TF_3 + ' + ', TF_1 + ' + ' + TF_3 + ' - ', TF_1+' + ' + TF_3 + ' + '])
        axis[10].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
        cell_sup_sup = list(set(Y_norm.index.values).intersection(set(TF_1_sup_TF_3_sup.index.values)))
        cell_sup_inf = list(set(Y_norm.index.values).intersection(set(TF_1_sup_TF_3_inf.index.values)))
        cell_inf_sup = list(set(Y_norm.index.values).intersection(set(TF_1_inf_TF_3_sup.index.values)))
        cell_inf_inf = list(set(Y_norm.index.values).intersection(set(TF_1_inf_TF_3_inf.index.values)))


        i_cell_sup_sup = [ind_dict[x] for x in cell_sup_sup]
        i_cell_sup_inf = [ind_dict[x] for x in cell_sup_inf]
        i_cell_inf_sup = [ind_dict[x] for x in cell_inf_sup]
        i_cell_inf_inf = [ind_dict[x] for x in cell_inf_inf]

        axis[11].scatter(x=embedded_data[:,0][i_cell_sup_sup], y=embedded_data[:,1][i_cell_sup_sup], s=3, color=colors_palette[::-1][0], marker='.')
        axis[11].scatter(x=embedded_data[:,0][i_cell_sup_inf], y=embedded_data[:,1][i_cell_sup_inf], s=3, color=colors_palette[::-1][1], marker='.')
        axis[11].scatter(x=embedded_data[:,0][i_cell_inf_sup], y=embedded_data[:,1][i_cell_inf_sup], s=3, color=colors_palette[::-1][2], marker='.')
        axis[11].scatter(x=embedded_data[:,0][i_cell_inf_inf], y=embedded_data[:,1][i_cell_inf_inf], s=3, color=colors_palette[::-1][3], marker='.')
# plt.suptitle(str(TG) + "\nCART Threshold : " + str(TF_1) + " : " + str(lim_val_1)+ "\n" + str(TF_2) + " : " + str(lim_val_2)+ "\n" + str(TF_3) + " : " + str(lim_val_3))

if full_first_layer:
    plt.suptitle(str(TG) + "\nCART Threshold : \n" + '{:<10s}{:<4s}\n'.format(str(TF_1), str(lim_val_1)) + '{:<10s}{:<4s}\n'.format(str(TF_2), str(lim_val_2)) + '{:<10s}{:<4s}\n'.format(str(TF_3), str(lim_val_3)))
elif show_first_layer_res:
    plt.suptitle(str(TG) + "\nCART Threshold : \n" + '{:<10s}{:<4s}\n'.format(str(TF_1), str(lim_val_1)) + '{:<10s}{:<4s}\n'.format(str(TF_2), str(lim_val_2)))
else:
    plt.suptitle(str(TG) + "\nCART Threshold : \n" + '{:<10s}{:<4s}\n'.format(str(TF_1), str(lim_val_1)))
if plot_method=="show":
    plt.show()
elif plot_method=="print":
    plt.savefig(out_data + "candidates_by_pval/" + str(TG) + ".pdf")
else:
    raise(NotImplementedError())