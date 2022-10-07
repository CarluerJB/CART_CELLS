from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)

data = "/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix.txt"
out_data = "/media/carluerj/Data/These/Results/GRN_inference/"

df = pd.read_table(data, sep="\t", header=0)
df_control = df.loc['RFP']
df_TF = df.loc[df.index[-28:-1]]
df_TF.drop(["FBH4", "GATA17L", "HSFB2A", "HYH"], inplace=True)
print(df_TF)

df_TF_log = df_TF.apply(np.log10, axis=1)
df_TF_log.replace(-np.inf, 0, inplace=True)
print(df_TF_log)
data = df_TF
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
model = model.fit(data)

model2 = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
model2 = model2.fit(data.transpose())


vmin = np.min(data.to_numpy())
vmax = np.max(data.to_numpy())
pos_per_point=10
dpi=960

fig, axs = plt.subplots(2, 2, figsize=(8, 5.6), gridspec_kw={'width_ratios': [12, 1], 'height_ratios': [12, 2]})

a=plot_dendrogram(model, truncate_mode="level", p=23, get_leaves=True, ax=axs[0][1], no_labels=True, labels=data.index.to_numpy(), orientation="right", leaf_font_size=2.7)
a2=plot_dendrogram(model2, truncate_mode="level", p=2313, get_leaves=True, ax=axs[1][0], no_labels=True, labels=data.columns.to_numpy(), orientation="bottom", leaf_font_size=2.7)
img = axs[0][0].imshow(data.iloc[a['leaves'][::-1],a2['leaves'][::]].to_numpy(), cmap='coolwarm', interpolation='none', aspect='auto', vmin=vmin, vmax=vmax)
axs[0][0].get_yaxis().set_visible(False)
axs[0][0].get_xaxis().set_visible(False)
axs[0][1].get_xaxis().set_visible(False)
axs[1][0].get_xaxis().set_visible(False)
ax1p=axs[0][1].twinx()
ax1p.get_yaxis().set_visible(True)
ax1p.set_yticks([*np.arange(0.5, len(data.index)+0.5), 23], minor=False)
ax1p.set_yticklabels([*data.index.to_numpy()[a['leaves']], ''], fontsize=5)

# ax2=axs[0][0].twiny()
# ax2.set_xticks(range(int(len(data.columns[::pos_per_point]))))
# ax2.tick_params(width=0.1)
# ax2.set_xticklabels(data.iloc[:,a2['leaves']].columns[::pos_per_point], rotation=90, fontsize=2)

axs[1][1].set_axis_off()
plt.tight_layout()
plt.savefig(out_data + "TF_clust_log_filtered_4FT_no_log" + ".png", dpi=dpi)
# plt.show()