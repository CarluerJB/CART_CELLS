# ===============================
# AUTHOR     : CARLUER Jean-Baptiste
# CREATE DATE     : 2022-2023
# PURPOSE     : Thesis in BioInformatics
# SPECIAL NOTES:
# ===============================
# Change History:
#
# # =================================

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import sys


class DENDROGRAM_TW:
    def __init__(self, save_dir_path, ge_matrix_path=None, TF_list_path=None):
        self.save_dir_path = save_dir_path
        if ge_matrix_path != None:
            self.path_output = self.save_dir_path + "network/tw_dendro_gc.png"
            self.ge_matrix_path = ge_matrix_path
            self.tf_list_path = TF_list_path
        else:
            self.path_output = (
                self.save_dir_path + "network/source_clust_to_GO_clust.png"
            )
        if TF_list_path != None:
            self.tf_list_path = TF_list_path
        self.path_list_src = self.save_dir_path + "network/resume_TF_target.txt"
        self.path_GO_src = self.save_dir_path + "network/association_study/"

        self.parameter_file_path = "PARAMETERS/PARAM_CLUST_DEFAULT.txt"
        self.load_and_init_parameter()
        self.show_parameter()

    ## PARAMETERS METHODS
    # Init default parameters and load new parameters if user define some
    def load_and_init_parameter(self):
        self.GO_TERM_FMT = "BP"
        self.PVAL = (
            "p_bonferroni"  # p_uncorrected, p_bonferroni, p_sidak, p_holm, p_fdr_bh
        )
        self.PVAL_THRES = 0.001
        self.X_POS_PER_PT = 1
        self.Y_POS_PER_PT = 1
        self.DPI = 960
        with open(self.parameter_file_path, "r") as parameter_file:
            for line in parameter_file:
                if ("#" in line) or (line == "\n"):
                    continue
                target_param, param = line.split(" ")
                if target_param == "GO_TERM_FMT":
                    self.GO_TERM_FMT = param[:-1]
                if target_param == "PVAL":
                    self.PVAL = param[:-1]
                if target_param == "PVAL_THRES":
                    self.PVAL_THRES = float(param)
                if target_param == "X_POS_PER_PT":
                    self.X_POS_PER_PT = int(param)
                if target_param == "Y_POS_PER_PT":
                    self.Y_POS_PER_PT = int(param)
                if target_param == "DPI":
                    self.DPI = int(param)

    def show_parameter(self):
        print("\n\tPARAMETERS : ")
        print("\t\tGO_TERM_FMT : ", self.GO_TERM_FMT)
        print("\t\tPVAL : ", self.PVAL)
        print("\t\tPVAL_THRES : ", self.PVAL_THRES)
        print("\t\tX_POS_PER_PT : ", self.X_POS_PER_PT)
        print("\t\tY_POS_PER_PT : ", self.Y_POS_PER_PT)
        print("\t\tDPI : ", self.DPI)
        print("\n")

    ## DATA LOADING METHODS
    # Load output of GRN.run_find_enrichment
    def load_GO_data(self):
        list_src = pd.read_table(self.path_list_src, sep=",", header=0)
        self.source_go_term_df = pd.DataFrame(
            {elem: [] for _, elem in list_src["source"].items()}
        )
        for id, elem in list_src["source"].items():
            try:
                go_association = pd.read_table(
                    self.path_GO_src + elem + ".txt", sep="\t", header=0
                )[["# GO", "NS", self.PVAL, "name"]]
            except:
                continue
            go_association.drop(
                go_association[go_association[self.PVAL] > self.PVAL_THRES].index,
                inplace=True,
            )
            go_association["# GO"] = go_association["# GO"].str.replace(
                ".", "", regex=False
            )
            go_association.drop(
                go_association[go_association["NS"] != self.GO_TERM_FMT].index,
                axis=0,
                inplace=True,
            )
            go_association.index = go_association["name"]
            go_association.drop(["# GO", "NS", "name"], axis=1, inplace=True)
            already_present_index = self.source_go_term_df.index.intersection(
                go_association.index
            )
            missing_index = go_association.index.difference(
                self.source_go_term_df.index
            )
            self.source_go_term_df = pd.concat(
                [
                    self.source_go_term_df,
                    go_association.loc[
                        missing_index,
                    ],
                ]
            )
            self.source_go_term_df.loc[
                already_present_index, elem
            ] = go_association.loc[already_present_index, self.PVAL]
            self.source_go_term_df[elem] = -np.log10(self.source_go_term_df[elem])
            self.source_go_term_df.drop([self.PVAL], axis=1, inplace=True)
        self.source_go_term_df.fillna(0, inplace=True)
        self.source_go_term_df.drop(
            self.source_go_term_df.columns[self.source_go_term_df.sum() <= 0.0],
            axis=1,
            inplace=True,
        )
        self.source_go_term_df.drop(
            self.source_go_term_df.index[self.source_go_term_df.sum(axis=1) <= 0.0],
            axis=0,
            inplace=True,
        )

    # Load Gene expression matrix
    # txt and h5 formats are supported
    def load_GE_matrix(self):
        data_src_type = self.ge_matrix_path.split(".")[1]
        if data_src_type == "txt":
            self.ge_matrix = pd.read_table(self.ge_matrix_path, sep="\t", header=0)
        elif data_src_type == "h5":
            self.ge_matrix = pd.read_hdf(self.ge_matrix_path)
        else:
            raise NotImplementedError

    # Load the list of TF which can be found in GE_matrix
    def load_tf_list(self):
        self.tf_list = pd.read_table(self.tf_list_path, header=None)[0].to_list()

    ## TABLE MANIPULATION METHODS
    # Extract TF (list which come from load_tf_list) from GE matrix
    def extract_TF_from_GE_matrix(self):
        self.GE_TF_matrix = self.ge_matrix.loc[self.tf_list].transpose()

    ## PLOTTING METHODS
    # PLOT TF vs GO term of AGI linked to TF
    def plot_clustering_GOTW(self):
        model = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage="ward"
        )
        model = model.fit(self.source_go_term_df)

        model2 = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage="ward"
        )
        model2 = model2.fit(self.source_go_term_df.transpose())

        vmin = np.min(self.source_go_term_df.to_numpy())
        vmax = np.max(self.source_go_term_df.to_numpy())

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(8, 6),
            gridspec_kw={"height_ratios": [12, 1], "width_ratios": [12, 2]},
        )

        a = plot_dendrogram(
            model,
            truncate_mode="level",
            p=len(self.source_go_term_df.index),
            get_leaves=True,
            ax=axs[0][1],
            no_labels=True,
            labels=self.source_go_term_df.index.to_numpy(),
            orientation="right",
            leaf_font_size=2.7,
        )
        a2 = plot_dendrogram(
            model2,
            truncate_mode="level",
            p=len(self.source_go_term_df.columns),
            get_leaves=True,
            ax=axs[1][0],
            no_labels=True,
            labels=self.source_go_term_df.columns.to_numpy(),
            orientation="bottom",
            leaf_font_size=2.7,
        )

        img = axs[0][0].imshow(
            self.source_go_term_df.iloc[a["leaves"][::-1], a2["leaves"][::]].to_numpy(),
            cmap="coolwarm",
            interpolation="none",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax1 = axs[0][0].twinx()
        ax1.get_yaxis().set_visible(True)
        ax1.set_yticks(
            [
                *np.arange(
                    0.5, len(self.source_go_term_df.index[:: self.Y_POS_PER_PT]) + 0.5
                ),
                len(self.source_go_term_df.index[:: self.Y_POS_PER_PT]),
            ],
            minor=False,
        )
        ax1.set_yticklabels(
            [
                *self.source_go_term_df.index.to_numpy()[
                    a["leaves"][:: self.Y_POS_PER_PT]
                ],
                "",
            ],
            fontsize=20
            / len(
                self.source_go_term_df.index.to_numpy()[
                    a["leaves"][:: self.Y_POS_PER_PT]
                ]
            ),
        )
        ax1.tick_params(width=0.1)

        ax2 = axs[0][0].twiny()
        ax2.get_xaxis().set_visible(True)
        ax2.set_xticks(
            np.arange(
                int(len(self.source_go_term_df.columns[:: self.X_POS_PER_PT])) / 2,
                step=0.5,
            ),
            self.source_go_term_df.iloc[:, a2["leaves"]].columns[:: self.X_POS_PER_PT],
            rotation=90,
            fontsize=20
            / len(
                self.source_go_term_df.iloc[:, a2["leaves"]].columns[
                    :: self.X_POS_PER_PT
                ]
            ),
        )
        ax2.tick_params(width=0.1)
        # ax2.set_xticklabels(
        #     self.source_go_term_df.iloc[:, a2["leaves"]].columns[:: self.X_POS_PER_PT],
        #     rotation=90,
        #     fontsize=20
        #     / len(
        #         self.source_go_term_df.iloc[:, a2["leaves"]].columns[
        #             :: self.X_POS_PER_PT
        #         ]
        #     ),
        # )

        axs[1][1].set_axis_off()
        plt.tight_layout()
        plt.savefig(self.path_output, dpi=self.DPI)

    # PLOT TF vs Cells using heatmap and dendrograms
    def plot_clustering_TW(self, log10=True):
        sys.stdout.write("\r GENERATING TW Clustering ")
        sys.stdout.flush()
        if log10:
            self.GE_TF_matrix = self.GE_TF_matrix.apply(np.log10, axis=1)
            self.GE_TF_matrix.replace(-np.inf, 0, inplace=True)
        print("1")
        model = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage="ward"
        )
        model = model.fit(self.GE_TF_matrix)
        print("2")
        model2 = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage="ward"
        )
        model2 = model2.fit(self.GE_TF_matrix.transpose())
        print("3")
        vmin = np.min(self.GE_TF_matrix.to_numpy())
        vmax = np.max(self.GE_TF_matrix.to_numpy())

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(8, 5.6),
            gridspec_kw={"width_ratios": [12, 1], "height_ratios": [12, 2]},
        )
        print("4")
        a = plot_dendrogram(
            model,
            truncate_mode="level",
            p=len(self.GE_TF_matrix.index),
            get_leaves=True,
            ax=axs[0][1],
            no_labels=True,
            labels=self.GE_TF_matrix.index.to_numpy(),
            orientation="right",
            leaf_font_size=2.7,
        )
        a2 = plot_dendrogram(
            model2,
            truncate_mode="level",
            p=len(self.GE_TF_matrix.columns),
            get_leaves=True,
            ax=axs[1][0],
            no_labels=True,
            labels=self.GE_TF_matrix.columns.to_numpy(),
            orientation="bottom",
            leaf_font_size=2.7,
        )
        img = axs[0][0].imshow(
            self.GE_TF_matrix.iloc[a["leaves"][::-1], a2["leaves"][::]].to_numpy(),
            cmap="coolwarm",
            interpolation="none",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        print("5")
        axs[0][0].get_yaxis().set_visible(False)
        axs[0][0].get_xaxis().set_visible(False)
        axs[0][1].get_xaxis().set_visible(False)
        axs[1][0].get_xaxis().set_visible(False)
        ax1p = axs[0][1].twinx()
        ax1p.get_yaxis().set_visible(True)
        ax1p.set_yticks(
            [*np.arange(0.5, len(self.GE_TF_matrix.index) + 0.5), 23], minor=False
        )
        ax1p.set_yticklabels(
            [*self.GE_TF_matrix.index.to_numpy()[a["leaves"]], ""], fontsize=5
        )
        axs[1][1].set_axis_off()

        plt.tight_layout()
        plt.savefig(self.path_output, dpi=self.DPI)


# Plotting function for dendrogram
def plot_dendrogram(model, **kwargs):
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
    return dendrogram(linkage_matrix, **kwargs)
