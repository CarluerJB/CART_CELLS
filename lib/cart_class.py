import os
import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
import graphviz
import re
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import sys
import time
import seaborn as sns
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

np.seterr(divide="ignore")
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


class CART_TREE:
    def __init__(
        self,
        ge_matrix_path,
        tf_list_path,
        save_dir_path,
        analysis_type="genes",
        parameter_file_path="PARAMETERS/PARAM_CART_DEFAULT.txt",
        target_sub_list_path=None,
    ):
        self.SORT_ORDER_REF = {
            "model_score": False,
            "gini_score_0": True,
            "ratio": True,
            "ratio_mean": True,
            "node": True,
            "iter_score": True,
            "p-val_1": False,
        }
        self.run_UMAP = False
        self.CLASS_TRANSFO = False
        self.MIN_SAMPLE_CUT = 0.10
        self.CRITERION = "gini"
        self.THRES_MODEL = 0.0
        self.THRES_CRITERION = 1.0
        self.THRES_ZERO_TE_TR = 1.0
        self.THRES_PVAL = 3.0
        self.CLASS_WEIGHT = None

        self.parameter_file_path = parameter_file_path
        self.ge_matrix_path = ge_matrix_path
        self.tf_list_path = tf_list_path
        self.save_dir_path = save_dir_path
        self.embedded_data_path = self.save_dir_path + "embedding.npy"
        self.create_out_dir()
        self.load_GE_matrix()
        self.load_tf_list()
        self.analysis_type = analysis_type
        if self.analysis_type == "genes":
            self.Y_txt = "AGI"
            self.X_txt = "TF"
            self.Y = self.ge_matrix.drop(self.tf_list).transpose()
            self.X = self.ge_matrix.loc[self.tf_list].transpose()
            self.Y_norm = self.Y - self.Y.mean() / self.Y.std()
            self.Y_target_list = self.Y.columns
            if target_sub_list_path != None:
                self.Y_target_list = list(
                    set(
                        pd.read_table(target_sub_list_path, header=None)[
                            0
                        ].values.tolist()
                    ).intersection(self.Y.columns.values.tolist())
                )
        else:
            self.X = self.ge_matrix.drop(self.tf_list).transpose()
            self.Y = self.ge_matrix.loc[self.tf_list].transpose()
            self.Y_norm = self.Y - self.Y.mean() / self.Y.std()
            self.X_txt = "AGI"
            self.Y_txt = "TF"
        if self.CLASS_TRANSFO == True:
            self.y_to_quantile()
        self.compiled_table = pd.DataFrame(
            {
                "AGI": [],
                "model_score": [],
                "gini_score_0": [],
                "gini_score_1": [],
                "gini_score_2": [],
                "TF1": [],
                "TF2": [],
                "TF3": [],
                "sign1": [],
                "sign2": [],
                "sign3": [],
                "lim1": [],
                "lim2": [],
                "lim3": [],
                "N2 anova p-val_1": [],
                "N2 anova p-val_2": [],
                "N2 anova p-val_1-2": [],
                "N3 anova p-val_1": [],
                "N3 anova p-val_2": [],
                "N3 anova p-val_1-2": [],
                "N2_p-val_++_+-": [],
                "N2_p-val_++_-+": [],
                "N2_p-val_++_--": [],
                "N2_p-val_+-_-+": [],
                "N2_p-val_+-_--": [],
                "N2_p-val_-+_--": [],
                "N3_p-val_++_+-": [],
                "N3_p-val_++_-+": [],
                "N3_p-val_++_--": [],
                "N3_p-val_+-_-+": [],
                "N3_p-val_+-_--": [],
                "N3_p-val_-+_--": [],
                "mean_1+": [],
                "mean_1-": [],
                "mean_1+2+": [],
                "mean_1-2+": [],
                "mean_1+2-": [],
                "mean_1-2-": [],
                "mean_1+3+": [],
                "mean_1-3+": [],
                "mean_1+3-": [],
                "mean_1-3-": [],
                "perc_zero_train": [],
                "perc_zero_test": [],
                "perc_zero_pred": [],
                "perc_zero_tot": [],
            }
        )
        self._perc_cart_tree_interface = 0.0
        self._time_per_tree = 0.1

    def load_parameter_file(self):
        with open(self.parameter_file_path, "r") as parameter_file:
            for line in parameter_file:
                if ("#" in line) or (line == "\n"):
                    continue
                target_param, param = line.split(" ")
                if target_param == "CLASS_TRANSFO":
                    self.CLASS_TRANSFO = param == "True"
                if target_param == "MIN_SAMPLE_CUT":
                    self.MIN_SAMPLE_CUT = float(param)
                if target_param == "CRITERION":
                    self.CRITERION = param[:-1]
                if target_param == "THRES_MODEL":
                    self.THRES_MODEL = float(param)
                if target_param == "THRES_CRITERION":
                    self.THRES_CRITERION = float(param)
                if target_param == "THRES_ZERO_TE_TR":
                    self.THRES_ZERO_TE_TR = float(param)
                if target_param == "THRES_PVAL":
                    self.THRES_PVAL = float(param)
                if target_param == "CLASS_WEIGHT":
                    self.CLASS_WEIGHT = None if param[:-1] == "None" else "balanced"

    def show_parameter(self):
        print("\n\tPARAMETERS : ")
        print("\t\tCLASS_TRANSFO : ", self.CLASS_TRANSFO)
        print("\t\tMIN_SAMPLE_CUT : ", self.MIN_SAMPLE_CUT)
        print("\t\tCRITERION : ", self.CRITERION)
        print("\t\tCLASS_WEIGHT : ", self.CLASS_WEIGHT)
        print("\t\tTHRES_MODEL : ", self.THRES_MODEL)
        print("\t\tTHRES_CRITERION : ", self.THRES_CRITERION)
        print("\t\tTHRES_ZERO_TE_TR : ", self.THRES_ZERO_TE_TR)
        print("\t\tTHRES_PVAL : ", self.THRES_PVAL)
        print("\n")

    def create_out_dir(self):
        os.makedirs(self.save_dir_path + "score", exist_ok=True)
        os.makedirs(self.save_dir_path + "txt_tree", exist_ok=True)
        os.makedirs(self.save_dir_path + "tree", exist_ok=True)
        os.makedirs(self.save_dir_path + "CARTLUPLOT/", exist_ok=True)

    def load_GE_matrix(self):
        data_src_type = self.ge_matrix_path.split(".")[1]
        if data_src_type == "txt":
            self.ge_matrix = pd.read_table(self.ge_matrix_path, sep="\t", header=0)
        elif data_src_type == "h5":
            self.ge_matrix = pd.read_hdf(self.ge_matrix_path)
        else:
            raise NotImplementedError

    def load_tf_list(self):
        self.tf_list = pd.read_table(self.tf_list_path, header=None)[0].to_list()

    def subsample_columns(self, AGI_list):
        self.Y = self.Y[AGI_list]

    def y_to_quantile(self):
        Y_copy = self.Y.copy(deep=True)
        Y_quant = np.quantile(self.Y.to_numpy().flatten(), [0.33, 0.66, 1.0]).astype(
            np.int64()
        )
        self.Y[(Y_copy <= Y_quant[2]) & (Y_copy >= Y_quant[1])] = 2
        self.Y[(Y_copy <= Y_quant[1]) & (Y_copy >= Y_quant[0])] = 1
        self.Y[(Y_copy <= Y_quant[0])] = 0

    def eval_model(self, Y_id=None):
        if Y_id is None:
            Y_single = self.Y
        else:
            Y_single = self.Y[Y_id]
        x_train, x_test, y_train, y_test = train_test_split(
            self.X, Y_single, train_size=0.8, random_state=RANDOM_STATE
        )
        clf_eval = tree.DecisionTreeClassifier(
            min_samples_split=self.MIN_SAMPLE_CUT,
            max_depth=3,
            min_samples_leaf=self.MIN_SAMPLE_CUT,
            criterion=self.CRITERION,
            random_state=RANDOM_STATE,
            class_weight=self.CLASS_WEIGHT,
        )
        clf_eval = clf_eval.fit(x_train, y_train)
        y_pred = clf_eval.predict(x_test)
        self.model_evaluation = {
            "accuracy": metrics.accuracy_score(y_test, y_pred),
            "perc_0_pred": np.where(np.array(y_pred) == 0.0)[0].size
            / np.array(y_pred).size,
            "perc_0_train": np.where(np.array(y_train.to_numpy()) == 0.0)[0].size
            / np.array(y_train.values).size,
            "perc_0_test": np.where(np.array(y_test.to_numpy()) == 0.0)[0].size
            / np.array(y_test.values).size,
            "perc_0_total": len(Y_single[Y_single == 0].index) / len(Y_single.index),
        }

    def generate_CART_tree(self, Y_id=None):
        if Y_id is None:
            Y_single = self.Y
        else:
            Y_single = self.Y[Y_id]
            self._perc_cart_tree_interface += 1
        sys.stdout.write(
            "\r GENERATING CART tree for: {0} ({1}%, time left: {2} sec)".format(
                Y_id,
                round(
                    self._perc_cart_tree_interface * 100 / len(self.Y_target_list), 3
                ),
                round(
                    len(self.Y_target_list) * self._time_per_tree
                    - self._perc_cart_tree_interface * self._time_per_tree,
                    3,
                ),
            )
        )
        sys.stdout.flush()
        self.start = time.time()
        self.clf = tree.DecisionTreeClassifier(
            min_samples_split=self.MIN_SAMPLE_CUT,
            max_depth=3,
            min_samples_leaf=self.MIN_SAMPLE_CUT,
            criterion=self.CRITERION,
            random_state=RANDOM_STATE,
            class_weight=self.CLASS_WEIGHT,
        )
        self.clf = self.clf.fit(self.X, Y_single)
        print("Done")

    def save_CART_tree(self, Y_id=None):
        if Y_id is None:
            Y_id = "default"
            Y_single = self.Y
        else:
            Y_single = self.Y[Y_id]
        tree_rules = export_text(
            self.clf, feature_names=list(self.X.columns.values), show_weights=False
        )
        dot_data = tree.export_graphviz(
            self.clf,
            out_file=None,
            feature_names=self.X.columns,
            class_names=Y_single.index,
            filled=True,
            rounded=True,
            special_characters=True,
        )
        graph = graphviz.Source(dot_data, Y_id + ".gv")
        graph.render(directory=self.save_dir_path + "tree", view=False)
        with open(self.save_dir_path + "txt_tree/" + Y_id + ".txt", "w") as file:
            file.write(tree_rules)
        self.tree_rules = tree_rules.splitlines()
        self._CART_scoring = pd.DataFrame({"gini_score": [], "sample_size": []})
        with open(self.save_dir_path + "score/" + Y_id + ".txt", "w") as file:
            for i in range(len(self.clf.tree_.impurity)):
                file.write(
                    str(self.clf.tree_.impurity[i])
                    + ","
                    + str(self.clf.tree_.n_node_samples[i])
                    + "\n"
                )
                self._CART_scoring.loc[i, "gini_score"] = self.clf.tree_.impurity[i]
                self._CART_scoring.loc[i, "sample_size"] = int(
                    self.clf.tree_.n_node_samples[i]
                )
        with open(self.save_dir_path + "list_gene.txt", "a") as file:
            file.write(
                Y_id
                + ";"
                + str(self.model_evaluation["accuracy"])
                + ";"
                + str(self.model_evaluation["perc_0_pred"])
                + ";"
                + str(self.model_evaluation["perc_0_train"])
                + ";"
                + str(self.model_evaluation["perc_0_train"])
                + ";"
                + str(self.model_evaluation["perc_0_total"])
                + "\n"
            )

    def save_cartlu_plot(self, TG):
        # TODO deal with this
        show_first_layer_res = False
        full_first_layer = True
        kde = True
        common_norm = False
        log_y_axis = False
        histplot_multiple = "dodge"
        histplot_binwidth = None
        colors_palette = sns.color_palette("tab10", 4)
        ratio_sign = 10
        show_zero = False
        va_text_sign = "top"
        cell_width = 212
        cell_height = 22
        swatch_width = 48
        xaxis_label_size = 8
        legend_fontsize = 6.5
        plot_method = "print"
        path = Path(self.embedded_data_path)
        if path.is_file():
            self.run_UMAP = False
        else:
            self.run_UMAP = True
        if self.run_UMAP:
            model = UMAP(
                n_components=2, min_dist=0.3, n_neighbors=30, metric="correlation"
            )
            embedded_data = model.fit_transform(self.Y_norm.fillna(0))
            np.save(self.embedded_data_path, embedded_data)
            self.run_UMAP = False
        else:
            embedded_data = np.load(self.embedded_data_path)
        nb_plot = pd.notna(self.compiled_row[["TF1", "TF2", "TF3"]]).sum().sum()
        full_first_layer = False
        if nb_plot > 0:
            TF_1 = self.compiled_row["TF1"].values[0]
            lim_val_1 = self.compiled_row["lim1"].values[0]
            is_node1_sign = None
            if (self.compiled_row["p-val_1"] < 0.05).to_numpy()[0]:
                is_node1_sign = "*"
                if (self.compiled_row["p-val_1"] < 0.01).to_numpy()[0]:
                    is_node1_sign = "**"
                    if (self.compiled_row["p-val_1"] < 0.001).to_numpy()[0]:
                        is_node1_sign = "***"
            if nb_plot > 1:
                show_first_layer_res = True
                TF_2 = self.compiled_row["TF2"].values[0]
                lim_val_2 = self.compiled_row["lim2"].values[0]
                is_node2_pp_pm_sign = None
                is_node2_pp_mp_sign = None
                is_node2_pp_mm_sign = None
                is_node2_pm_mp_sign = None
                is_node2_pm_mm_sign = None
                is_node2_mp_mm_sign = None
                if (self.compiled_row["N2_p-val_++_+-"] < 0.05).to_numpy()[0]:
                    is_node2_pp_pm_sign = "*"
                    if (self.compiled_row["N2_p-val_++_+-"] < 0.01).to_numpy()[0]:
                        is_node2_pp_pm_sign = "**"
                        if (self.compiled_row["N2_p-val_++_+-"] < 0.001).to_numpy()[0]:
                            is_node2_pp_pm_sign = "***"
                if (self.compiled_row["N2_p-val_++_-+"] < 0.05).to_numpy()[0]:
                    is_node2_pp_mp_sign = "*"
                    if (self.compiled_row["N2_p-val_++_-+"] < 0.01).to_numpy()[0]:
                        is_node2_pp_mp_sign = "**"
                        if (self.compiled_row["N2_p-val_++_-+"] < 0.001).to_numpy()[0]:
                            is_node2_pp_mp_sign = "***"
                if (self.compiled_row["N2_p-val_++_--"] < 0.05).to_numpy()[0]:
                    is_node2_pp_mm_sign = "*"
                    if (self.compiled_row["N2_p-val_++_--"] < 0.01).to_numpy()[0]:
                        is_node2_pp_mm_sign = "**"
                        if (self.compiled_row["N2_p-val_++_--"] < 0.001).to_numpy()[0]:
                            is_node2_pp_mm_sign = "***"
                if (self.compiled_row["N2_p-val_+-_-+"] < 0.05).to_numpy()[0]:
                    is_node2_pm_mp_sign = "*"
                    if (self.compiled_row["N2_p-val_+-_-+"] < 0.01).to_numpy()[0]:
                        is_node2_pm_mp_sign = "**"
                        if (self.compiled_row["N2_p-val_+-_-+"] < 0.001).to_numpy()[0]:
                            is_node2_pm_mp_sign = "***"
                if (self.compiled_row["N2_p-val_+-_--"] < 0.05).to_numpy()[0]:
                    is_node2_pm_mm_sign = "*"
                    if (self.compiled_row["N2_p-val_+-_--"] < 0.01).to_numpy()[0]:
                        is_node2_pm_mm_sign = "**"
                        if (self.compiled_row["N2_p-val_+-_--"] < 0.001).to_numpy()[0]:
                            is_node2_pm_mm_sign = "***"
                if (self.compiled_row["N2_p-val_-+_--"] < 0.05).to_numpy()[0]:
                    is_node2_mp_mm_sign = "*"
                    if (self.compiled_row["N2_p-val_-+_--"] < 0.01).to_numpy()[0]:
                        is_node2_mp_mm_sign = "**"
                        if (self.compiled_row["N2_p-val_-+_--"] < 0.001).to_numpy()[0]:
                            is_node2_mp_mm_sign = "***"

                if nb_plot > 2:
                    full_first_layer = True
                    TF_3 = self.compiled_row["TF3"].values[0]
                    lim_val_3 = self.compiled_row["lim3"].values[0]
                    is_node3_pp_pm_sign = None
                    is_node3_pp_mp_sign = None
                    is_node3_pp_mm_sign = None
                    is_node3_pm_mp_sign = None
                    is_node3_pm_mm_sign = None
                    is_node3_mp_mm_sign = None

                    if (self.compiled_row["N3_p-val_++_+-"] < 0.05).to_numpy()[0]:
                        is_node3_pp_pm_sign = "*"
                        if (self.compiled_row["N3_p-val_++_+-"] < 0.01).to_numpy()[0]:
                            is_node3_pp_pm_sign = "**"
                            if (self.compiled_row["N3_p-val_++_+-"] < 0.001).to_numpy()[
                                0
                            ]:
                                is_node3_pp_pm_sign = "***"
                    if (self.compiled_row["N3_p-val_++_-+"] < 0.05).to_numpy()[0]:
                        is_node3_pp_mp_sign = "*"
                        if (self.compiled_row["N3_p-val_++_-+"] < 0.01).to_numpy()[0]:
                            is_node3_pp_mp_sign = "**"
                            if (self.compiled_row["N3_p-val_++_-+"] < 0.001).to_numpy()[
                                0
                            ]:
                                is_node3_pp_mp_sign = "***"
                    if (self.compiled_row["N3_p-val_++_--"] < 0.05).to_numpy()[0]:
                        is_node3_pp_mm_sign = "*"
                        if (self.compiled_row["N3_p-val_++_--"] < 0.01).to_numpy()[0]:
                            is_node3_pp_mm_sign = "**"
                            if (self.compiled_row["N3_p-val_++_--"] < 0.001).to_numpy()[
                                0
                            ]:
                                is_node3_pp_mm_sign = "***"
                    if (self.compiled_row["N3_p-val_+-_-+"] < 0.05).to_numpy()[0]:
                        is_node3_pm_mp_sign = "*"
                        if (self.compiled_row["N3_p-val_+-_-+"] < 0.01).to_numpy()[0]:
                            is_node3_pm_mp_sign = "**"
                            if (self.compiled_row["N3_p-val_+-_-+"] < 0.001).to_numpy()[
                                0
                            ]:
                                is_node3_pm_mp_sign = "***"
                    if (self.compiled_row["N3_p-val_+-_--"] < 0.05).to_numpy()[0]:
                        is_node3_pm_mm_sign = "*"
                        if (self.compiled_row["N3_p-val_+-_--"] < 0.01).to_numpy()[0]:
                            is_node3_pm_mm_sign = "**"
                            if (self.compiled_row["N3_p-val_+-_--"] < 0.001).to_numpy()[
                                0
                            ]:
                                is_node3_pm_mm_sign = "***"
                    if (self.compiled_row["N3_p-val_-+_--"] < 0.05).to_numpy()[0]:
                        is_node3_mp_mm_sign = "*"
                        if (self.compiled_row["N3_p-val_-+_--"] < 0.01).to_numpy()[0]:
                            is_node3_mp_mm_sign = "**"
                            if (self.compiled_row["N3_p-val_-+_--"] < 0.001).to_numpy()[
                                0
                            ]:
                                is_node3_mp_mm_sign = "***"
        else:
            return
        figure, axis = plt.subplots(nb_plot, 4, figsize=(15, 10))
        axis = axis.ravel()
        TF_1_sup = self.Y.loc[self.X[TF_1] > lim_val_1, TG]
        TF_1_inf = self.Y.loc[self.X[TF_1] <= lim_val_1, TG]
        TF_1_sup_norm = np.nan_to_num(np.log10(TF_1_sup.values), neginf=0)
        TF_1_inf_norm = np.nan_to_num(np.log10(TF_1_inf.values), neginf=0)
        if not show_zero:
            TF_1_sup_norm[TF_1_sup == 0] = np.nan
            TF_1_inf_norm[TF_1_inf == 0] = np.nan
        else:
            TF_1_sup_norm = TF_1_sup_norm + 1
            TF_1_inf_norm = TF_1_inf_norm + 1
        try:
            sns.histplot(
                [
                    TF_1_inf_norm if len(TF_1_inf_norm) > 0 else np.nan,
                    TF_1_sup_norm if len(TF_1_sup_norm) > 0 else np.nan,
                ][::-1],
                kde=kde,
                legend=True,
                ax=axis[0],
                stat="percent",
                common_norm=common_norm,
                element="bars",
                multiple=histplot_multiple,
                binwidth=histplot_binwidth,
                palette=colors_palette[:2][::-1],
            )
        except:
            sns.histplot(
                [
                    TF_1_inf_norm if len(TF_1_inf_norm) > 0 else np.nan,
                    TF_1_sup_norm if len(TF_1_sup_norm) > 0 else np.nan,
                ][::-1],
                kde=kde != True,
                legend=True,
                ax=axis[0],
                stat="percent",
                common_norm=common_norm,
                element="bars",
                multiple=histplot_multiple,
                binwidth=histplot_binwidth,
                palette=colors_palette[:2][::-1],
            )
        kde = True
        histplot_binwidth = 0.2
        legend = axis[0].get_legend()
        handles = legend.legendHandles
        legend.remove()
        label_list = []
        handle_list = []
        label_list.append(TF_1 + " - " + "\n(" + str(len(TF_1_inf.index)) + " samples)")
        label_list.append(TF_1 + " + " + "\n(" + str(len(TF_1_sup.index)) + " samples)")
        for i, elem in enumerate(label_list):
            row = i
            y = row * cell_height
            swatch_start_x = cell_width
            text_pos_x = cell_width + swatch_width + 7
            axis[0].text(
                text_pos_x,
                y,
                elem,
                fontsize=legend_fontsize,
                horizontalalignment="left",
                verticalalignment="center",
            )
            handle_list.append(
                Rectangle(
                    xy=(swatch_start_x, y - 9),
                    width=swatch_width,
                    height=18,
                    facecolor=colors_palette[i],
                    edgecolor="0.7",
                )
            )
        axis[0].legend(handles=handle_list, labels=label_list, fontsize=legend_fontsize)

        sns.violinplot(
            data=[TF_1_inf, TF_1_sup],
            ax=axis[1],
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        axis[1].boxplot(
            [TF_1_inf, TF_1_sup],
            positions=[0, 1],
            showfliers=True,
            showcaps=True,
            labels=None,
        )
        if log_y_axis:
            axis[1].set(yscale="symlog")
        # TODO add adaptative height
        if is_node1_sign != None:
            x1, x2 = 0, 1
            y, h, col = axis[1].get_ylim()[1] + 2, 2, "k"
            axis[1].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            axis[1].text(
                (x1 + x2) * 0.5,
                y + h,
                is_node1_sign,
                ha="center",
                va=va_text_sign,
                color=col,
            )

        axis[1].set_xticklabels(["", "", TF_1 + " - ", TF_1 + " + "])
        axis[1].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)

        axis[2].boxplot(
            [TF_1_inf, TF_1_sup],
            positions=[0, 1],
            showfliers=False,
            showcaps=True,
            labels=None,
        )
        if log_y_axis:
            axis[2].set(yscale="symlog")

        axis[2].set_xticklabels([TF_1 + " - ", TF_1 + " + "])
        axis[2].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)

        # UMAP
        cell_sup = list(
            set(self.Y_norm.index.values).intersection(set(TF_1_sup.index.values))
        )
        cell_inf = list(
            set(self.Y_norm.index.values).intersection(set(TF_1_inf.index.values))
        )
        ind_dict = dict((k, i) for i, k in enumerate(self.Y_norm.index.values))
        i_cell_sup = [ind_dict[x] for x in cell_sup]
        i_cell_inf = [ind_dict[x] for x in cell_inf]

        # plt.scatter(embedded_data[:,0], embedded_data[:,1], s=5, c="green", marker='.')
        axis[3].scatter(
            x=embedded_data[:, 0][i_cell_sup],
            y=embedded_data[:, 1][i_cell_sup],
            s=1,
            color=colors_palette[:2][::-1][0],
            marker=".",
            alpha=0.05,
        )
        axis[3].scatter(
            x=embedded_data[:, 0][i_cell_inf],
            y=embedded_data[:, 1][i_cell_inf],
            s=1,
            color=colors_palette[:2][::-1][1],
            marker=".",
            alpha=0.05,
        )

        if show_first_layer_res:
            # TF1 + TF2

            TF_1_sup_TF_2_sup = self.Y.loc[
                (self.X[TF_1] > lim_val_1) & (self.X[TF_2] > lim_val_2), TG
            ]
            TF_1_sup_TF_2_inf = self.Y.loc[
                (self.X[TF_1] > lim_val_1) & (self.X[TF_2] <= lim_val_2), TG
            ]
            TF_1_inf_TF_2_sup = self.Y.loc[
                (self.X[TF_1] <= lim_val_1) & (self.X[TF_2] > lim_val_2), TG
            ]
            TF_1_inf_TF_2_inf = self.Y.loc[
                (self.X[TF_1] <= lim_val_1) & (self.X[TF_2] <= lim_val_2), TG
            ]

            TF_1_sup_TF_2_sup_norm = np.nan_to_num(
                np.log10(TF_1_sup_TF_2_sup), neginf=0
            )
            TF_1_sup_TF_2_inf_norm = np.nan_to_num(
                np.log10(TF_1_sup_TF_2_inf), neginf=0
            )
            TF_1_inf_TF_2_sup_norm = np.nan_to_num(
                np.log10(TF_1_inf_TF_2_sup), neginf=0
            )
            TF_1_inf_TF_2_inf_norm = np.nan_to_num(
                np.log10(TF_1_inf_TF_2_inf), neginf=0
            )

            if not show_zero:
                TF_1_sup_TF_2_sup_norm[TF_1_sup_TF_2_sup == 0] = np.nan
                TF_1_sup_TF_2_inf_norm[TF_1_sup_TF_2_inf == 0] = np.nan
                TF_1_inf_TF_2_sup_norm[TF_1_inf_TF_2_sup == 0] = np.nan
                TF_1_inf_TF_2_inf_norm[TF_1_inf_TF_2_inf == 0] = np.nan
            else:
                TF_1_sup_TF_2_sup_norm = TF_1_sup_TF_2_sup_norm + 1
                TF_1_sup_TF_2_inf_norm = TF_1_sup_TF_2_inf_norm + 1
                TF_1_inf_TF_2_sup_norm = TF_1_inf_TF_2_sup_norm + 1
                TF_1_inf_TF_2_inf_norm = TF_1_inf_TF_2_inf_norm + 1
            if len(np.unique(TF_1_sup_TF_2_sup_norm)) < 3:
                kde = False
                histplot_binwidth = None
            if len(np.unique(TF_1_sup_TF_2_inf_norm)) < 3:
                kde = False
                histplot_binwidth = None
            if len(np.unique(TF_1_inf_TF_2_sup_norm)) < 3:
                kde = False
                histplot_binwidth = None
            if len(np.unique(TF_1_inf_TF_2_inf_norm)) < 3:
                kde = False
                histplot_binwidth = None
            sns.histplot(
                data=[
                    TF_1_inf_TF_2_inf_norm
                    if len(TF_1_inf_TF_2_inf_norm) > 0
                    else np.nan,
                    TF_1_inf_TF_2_sup_norm
                    if len(TF_1_inf_TF_2_sup_norm) > 0
                    else np.nan,
                    TF_1_sup_TF_2_inf_norm
                    if len(TF_1_sup_TF_2_inf_norm) > 0
                    else np.nan,
                    TF_1_sup_TF_2_sup_norm
                    if len(TF_1_sup_TF_2_sup_norm) > 0
                    else np.nan,
                ][::-1],
                kde=kde,
                legend=True,
                ax=axis[4],
                stat="percent",
                common_norm=common_norm,
                element="bars",
                multiple=histplot_multiple,
                binwidth=histplot_binwidth,
                palette=colors_palette[::-1],
            )
            kde = True
            histplot_binwidth = 0.2
            legend = axis[4].get_legend()
            handles = legend.legendHandles
            legend.remove()
            label_list = []
            handle_list = []
            label_list.append(
                TF_1
                + " - "
                + TF_2
                + " - "
                + "\n("
                + str(len(TF_1_inf_TF_2_inf.index))
                + " samples)"
            )
            label_list.append(
                TF_1
                + " - "
                + TF_2
                + " + "
                + "\n("
                + str(len(TF_1_inf_TF_2_sup.index))
                + " samples)"
            )
            label_list.append(
                TF_1
                + " + "
                + TF_2
                + " - "
                + "\n("
                + str(len(TF_1_sup_TF_2_inf.index))
                + " samples)"
            )
            label_list.append(
                TF_1
                + " + "
                + TF_2
                + " + "
                + "\n("
                + str(len(TF_1_sup_TF_2_sup.index))
                + " samples)"
            )
            for i, elem in enumerate(label_list):
                row = i
                y = row * cell_height
                swatch_start_x = cell_width
                text_pos_x = cell_width + swatch_width + 7
                axis[4].text(
                    text_pos_x,
                    y,
                    elem,
                    fontsize=legend_fontsize,
                    horizontalalignment="left",
                    verticalalignment="center",
                )
                handle_list.append(
                    Rectangle(
                        xy=(swatch_start_x, y - 9),
                        width=swatch_width,
                        height=18,
                        facecolor=colors_palette[i],
                        edgecolor="0.7",
                    )
                )
            axis[4].legend(
                handles=handle_list, labels=label_list, fontsize=legend_fontsize
            )
            sns.violinplot(
                [
                    TF_1_inf_TF_2_inf,
                    TF_1_inf_TF_2_sup,
                    TF_1_sup_TF_2_inf,
                    TF_1_sup_TF_2_sup,
                ],
                ax=axis[5],
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )
            axis[5].boxplot(
                [
                    TF_1_inf_TF_2_inf,
                    TF_1_inf_TF_2_sup,
                    TF_1_sup_TF_2_inf,
                    TF_1_sup_TF_2_sup,
                ],
                positions=[0, 1, 2, 3],
                showfliers=True,  # Do not show the outliers beyond the caps.
                showcaps=True,
                labels=None,
            )  # Do not show the caps
            if log_y_axis:
                axis[5].set(yscale="symlog")

            max_val_ax5 = axis[5].get_ylim()[1]
            height_p_sign_ax5 = int(max_val_ax5 / ratio_sign)
            y_ax5, h_ax5, col_ax5 = max_val_ax5 + 2, height_p_sign_ax5, "k"

            axis[6].boxplot(
                [
                    TF_1_inf_TF_2_inf,
                    TF_1_inf_TF_2_sup,
                    TF_1_sup_TF_2_inf,
                    TF_1_sup_TF_2_sup,
                ],
                positions=[0, 1, 2, 3],
                showfliers=False,  # Do not show the outliers beyond the caps.
                showcaps=True,
                labels=None,
            )  # Do not show the caps
            if log_y_axis:
                axis[6].set(yscale="symlog")

            y_add = 1
            if is_node2_pp_pm_sign != None:
                x1, x2 = 2, 3
                axis[5].plot(
                    [x1, x1, x2, x2],
                    [y_ax5, y_ax5 + h_ax5, y_ax5 + h_ax5, y_ax5],
                    lw=1.5,
                    c=col_ax5,
                )
                axis[5].text(
                    (x1 + x2) * 0.5,
                    y_ax5 + h_ax5,
                    is_node2_pp_pm_sign,
                    ha="center",
                    va=va_text_sign,
                    color=col_ax5,
                )
                y_add = 1
            if is_node2_mp_mm_sign != None:
                x1, x2 = 0, 1
                axis[5].plot(
                    [x1, x1, x2, x2],
                    [y_ax5, y_ax5 + h_ax5, y_ax5 + h_ax5, y_ax5],
                    lw=1.5,
                    c=col_ax5,
                )
                axis[5].text(
                    (x1 + x2) * 0.5,
                    y_ax5 + h_ax5,
                    is_node2_mp_mm_sign,
                    ha="center",
                    va=va_text_sign,
                    color=col_ax5,
                )
                y_add = 1

            if is_node2_pp_mp_sign != None:
                x1, x2 = 1, 3
                y_ax5, h_ax5, col_ax5 = (
                    max_val_ax5
                    + 2
                    + (y_add * height_p_sign_ax5)
                    + int((y_add * height_p_sign_ax5) / 3),
                    height_p_sign_ax5,
                    "k",
                )
                axis[5].plot(
                    [x1, x1, x2, x2],
                    [y_ax5, y_ax5 + h_ax5, y_ax5 + h_ax5, y_ax5],
                    lw=1.5,
                    c=col_ax5,
                )
                axis[5].text(
                    (x1 + x2) * 0.5,
                    y_ax5 + h_ax5,
                    is_node2_pp_mp_sign,
                    ha="center",
                    va=va_text_sign,
                    color=col_ax5,
                )
                y_add += 1

            if is_node2_pm_mm_sign != None:
                x1, x2 = 0, 2
                y_ax5, h_ax5, col_ax5 = (
                    max_val_ax5
                    + 2
                    + (y_add * height_p_sign_ax5)
                    + int((y_add * height_p_sign_ax5) / 3),
                    height_p_sign_ax5,
                    "k",
                )
                axis[5].plot(
                    [x1, x1, x2, x2],
                    [y_ax5, y_ax5 + h_ax5, y_ax5 + h_ax5, y_ax5],
                    lw=1.5,
                    c=col_ax5,
                )
                axis[5].text(
                    (x1 + x2) * 0.5,
                    y_ax5 + h_ax5,
                    is_node2_pm_mm_sign,
                    ha="center",
                    va=va_text_sign,
                    color=col_ax5,
                )
                y_add += 1

            if is_node2_pm_mp_sign != None:
                x1, x2 = 1, 2
                y_ax5, h_ax5, col_ax5 = (
                    max_val_ax5
                    + 2
                    + (y_add * height_p_sign_ax5)
                    + int((y_add * height_p_sign_ax5) / 3),
                    height_p_sign_ax5,
                    "k",
                )
                axis[5].plot(
                    [x1, x1, x2, x2],
                    [y_ax5, y_ax5 + h_ax5, y_ax5 + h_ax5, y_ax5],
                    lw=1.5,
                    c=col_ax5,
                )
                axis[5].text(
                    (x1 + x2) * 0.5,
                    y_ax5 + h_ax5,
                    is_node2_pm_mp_sign,
                    ha="center",
                    va=va_text_sign,
                    color=col_ax5,
                )
                y_add += 1

            if is_node2_pp_mm_sign != None:
                x1, x2 = 0, 3
                y_ax5, h_ax5, col_ax5 = (
                    max_val_ax5
                    + 2
                    + (y_add * height_p_sign_ax5)
                    + int((y_add * height_p_sign_ax5) / 3),
                    height_p_sign_ax5,
                    "k",
                )
                axis[5].plot(
                    [x1, x1, x2, x2],
                    [y_ax5, y_ax5 + h_ax5, y_ax5 + h_ax5, y_ax5],
                    lw=1.5,
                    c=col_ax5,
                )
                axis[5].text(
                    (x1 + x2) * 0.5,
                    y_ax5 + h_ax5,
                    is_node2_pp_mm_sign,
                    ha="center",
                    va=va_text_sign,
                    color=col_ax5,
                )
                y_add += 1
            axis[5].set_xticklabels(
                [
                    "",
                    "",
                    "",
                    "",
                    TF_1 + " - " + TF_2 + " - ",
                    TF_1 + " - " + TF_2 + " + ",
                    TF_1 + " + " + TF_2 + " - ",
                    TF_1 + " + " + TF_2 + " + ",
                ]
            )
            axis[5].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
            axis[6].set_xticklabels(
                [
                    TF_1 + " - " + TF_2 + " - ",
                    TF_1 + " - " + TF_2 + " + ",
                    TF_1 + " + " + TF_2 + " - ",
                    TF_1 + " + " + TF_2 + " + ",
                ]
            )
            axis[6].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
            cell_sup_sup = list(
                set(self.Y_norm.index.values).intersection(
                    set(TF_1_sup_TF_2_sup.index.values)
                )
            )
            cell_sup_inf = list(
                set(self.Y_norm.index.values).intersection(
                    set(TF_1_sup_TF_2_inf.index.values)
                )
            )
            cell_inf_sup = list(
                set(self.Y_norm.index.values).intersection(
                    set(TF_1_inf_TF_2_sup.index.values)
                )
            )
            cell_inf_inf = list(
                set(self.Y_norm.index.values).intersection(
                    set(TF_1_inf_TF_2_inf.index.values)
                )
            )

            i_cell_sup_sup = [ind_dict[x] for x in cell_sup_sup]
            i_cell_sup_inf = [ind_dict[x] for x in cell_sup_inf]
            i_cell_inf_sup = [ind_dict[x] for x in cell_inf_sup]
            i_cell_inf_inf = [ind_dict[x] for x in cell_inf_inf]

            # plt.scatter(embedded_data[:,0], embedded_data[:,1], s=5, c="green", marker='.')
            axis[7].scatter(
                x=embedded_data[:, 0][i_cell_sup_sup],
                y=embedded_data[:, 1][i_cell_sup_sup],
                s=1,
                color=colors_palette[::-1][0],
                marker=".",
                alpha=0.05,
            )
            axis[7].scatter(
                x=embedded_data[:, 0][i_cell_sup_inf],
                y=embedded_data[:, 1][i_cell_sup_inf],
                s=1,
                color=colors_palette[::-1][1],
                marker=".",
                alpha=0.05,
            )
            axis[7].scatter(
                x=embedded_data[:, 0][i_cell_inf_sup],
                y=embedded_data[:, 1][i_cell_inf_sup],
                s=1,
                color=colors_palette[::-1][2],
                marker=".",
                alpha=0.05,
            )
            axis[7].scatter(
                x=embedded_data[:, 0][i_cell_inf_inf],
                y=embedded_data[:, 1][i_cell_inf_inf],
                s=1,
                color=colors_palette[::-1][3],
                marker=".",
                alpha=0.05,
            )
            if full_first_layer:
                # TF1 + TF3

                TF_1_sup_TF_3_sup = self.Y.loc[
                    (self.X[TF_1] > lim_val_1) & (self.X[TF_3] > lim_val_3), TG
                ]
                TF_1_sup_TF_3_inf = self.Y.loc[
                    (self.X[TF_1] > lim_val_1) & (self.X[TF_3] <= lim_val_3), TG
                ]
                TF_1_inf_TF_3_sup = self.Y.loc[
                    (self.X[TF_1] <= lim_val_1) & (self.X[TF_3] > lim_val_3), TG
                ]
                TF_1_inf_TF_3_inf = self.Y.loc[
                    (self.X[TF_1] <= lim_val_1) & (self.X[TF_3] <= lim_val_3), TG
                ]

                TF_1_sup_TF_3_sup_norm = np.nan_to_num(
                    np.log10(TF_1_sup_TF_3_sup), neginf=0
                )
                TF_1_sup_TF_3_inf_norm = np.nan_to_num(
                    np.log10(TF_1_sup_TF_3_inf), neginf=0
                )
                TF_1_inf_TF_3_sup_norm = np.nan_to_num(
                    np.log10(TF_1_inf_TF_3_sup), neginf=0
                )
                TF_1_inf_TF_3_inf_norm = np.nan_to_num(
                    np.log10(TF_1_inf_TF_3_inf), neginf=0
                )

                if not show_zero:
                    TF_1_sup_TF_3_sup_norm[TF_1_sup_TF_3_sup == 0] = np.nan
                    TF_1_sup_TF_3_inf_norm[TF_1_sup_TF_3_inf == 0] = np.nan
                    TF_1_inf_TF_3_sup_norm[TF_1_inf_TF_3_sup == 0] = np.nan
                    TF_1_inf_TF_3_inf_norm[TF_1_inf_TF_3_inf == 0] = np.nan
                else:
                    TF_1_sup_TF_3_sup_norm = TF_1_sup_TF_3_sup_norm + 1
                    TF_1_sup_TF_3_inf_norm = TF_1_sup_TF_3_inf_norm + 1
                    TF_1_inf_TF_3_sup_norm = TF_1_inf_TF_3_sup_norm + 1
                    TF_1_inf_TF_3_inf_norm = TF_1_inf_TF_3_inf_norm + 1
                if len(np.unique(TF_1_sup_TF_3_sup_norm)) < 3:
                    kde = False
                    histplot_binwidth = None
                if len(np.unique(TF_1_sup_TF_3_inf_norm)) < 3:
                    kde = False
                    histplot_binwidth = None
                if len(np.unique(TF_1_inf_TF_3_sup_norm)) < 3:
                    kde = False
                    histplot_binwidth = None
                if len(np.unique(TF_1_inf_TF_3_inf_norm)) < 3:
                    kde = False
                    histplot_binwidth = None
                sns.histplot(
                    data=[
                        TF_1_inf_TF_3_inf_norm
                        if len(TF_1_inf_TF_3_inf_norm) > 0
                        else np.nan,
                        TF_1_inf_TF_3_sup_norm
                        if len(TF_1_inf_TF_3_sup_norm) > 0
                        else np.nan,
                        TF_1_sup_TF_3_inf_norm
                        if len(TF_1_sup_TF_3_inf_norm) > 0
                        else np.nan,
                        TF_1_sup_TF_3_sup_norm
                        if len(TF_1_sup_TF_3_sup_norm) > 0
                        else np.nan,
                    ][::-1],
                    kde=kde,
                    legend=True,
                    ax=axis[8],
                    stat="percent",
                    common_norm=common_norm,
                    element="bars",
                    multiple=histplot_multiple,
                    binwidth=histplot_binwidth,
                    palette=colors_palette[::-1],
                    label="data",
                )
                kde = True
                histplot_binwidth = 0.2
                label_list = []
                handle_list = []
                legend = axis[8].get_legend()
                handles = legend.legendHandles
                legend.remove()
                label_list.append(
                    TF_1
                    + " - "
                    + TF_3
                    + " - "
                    + "\n("
                    + str(len(TF_1_inf_TF_3_inf.index))
                    + " samples)"
                )
                label_list.append(
                    TF_1
                    + " - "
                    + TF_3
                    + " + "
                    + "\n("
                    + str(len(TF_1_inf_TF_3_sup.index))
                    + " samples)"
                )
                label_list.append(
                    TF_1
                    + " + "
                    + TF_3
                    + " - "
                    + "\n("
                    + str(len(TF_1_sup_TF_3_inf.index))
                    + " samples)"
                )
                label_list.append(
                    TF_1
                    + " + "
                    + TF_3
                    + " + "
                    + "\n("
                    + str(len(TF_1_sup_TF_3_sup.index))
                    + " samples)"
                )
                for i, elem in enumerate(label_list):
                    row = i
                    y = row * cell_height
                    swatch_start_x = cell_width
                    text_pos_x = cell_width + swatch_width + 7
                    axis[8].text(
                        text_pos_x,
                        y,
                        elem,
                        fontsize=legend_fontsize,
                        horizontalalignment="left",
                        verticalalignment="center",
                    )
                    handle_list.append(
                        Rectangle(
                            xy=(swatch_start_x, y - 9),
                            width=swatch_width,
                            height=18,
                            facecolor=colors_palette[i],
                            edgecolor="0.7",
                        )
                    )
                axis[8].legend(
                    handles=handle_list, labels=label_list, fontsize=legend_fontsize
                )

                sns.violinplot(
                    [
                        TF_1_inf_TF_3_inf,
                        TF_1_inf_TF_3_sup,
                        TF_1_sup_TF_3_inf,
                        TF_1_sup_TF_3_sup,
                    ],
                    ax=axis[9],
                    showmeans=True,
                    showmedians=True,
                    showextrema=True,
                )
                axis[9].boxplot(
                    [
                        TF_1_inf_TF_3_inf,
                        TF_1_inf_TF_3_sup,
                        TF_1_sup_TF_3_inf,
                        TF_1_sup_TF_3_sup,
                    ],
                    positions=[0, 1, 2, 3],
                    showfliers=True,  # Do not show the outliers beyond the caps.
                    showcaps=True,
                    labels=None,
                )  # Do not show the caps
                if log_y_axis:
                    axis[9].set(yscale="symlog")

                axis[10].boxplot(
                    [
                        TF_1_inf_TF_3_inf,
                        TF_1_inf_TF_3_sup,
                        TF_1_sup_TF_3_inf,
                        TF_1_sup_TF_3_sup,
                    ],
                    positions=[0, 1, 2, 3],
                    showfliers=False,  # Do not show the outliers beyond the caps.
                    showcaps=True,
                    labels=None,
                )  # Do not show the caps
                if log_y_axis:
                    axis[10].set(yscale="symlog")

                max_val = axis[9].get_ylim()[1]
                height_p_sign = int(max_val / ratio_sign)
                y, h, col = max_val + 2, height_p_sign, "k"
                y_add = 1
                if is_node3_pp_pm_sign != None:
                    x1, x2 = 2, 3
                    axis[9].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                    axis[9].text(
                        (x1 + x2) * 0.5,
                        y + h,
                        is_node3_pp_pm_sign,
                        ha="center",
                        va=va_text_sign,
                        color=col,
                    )
                    y_add = 2
                if is_node3_mp_mm_sign != None:
                    x1, x2 = 0, 1
                    axis[9].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                    axis[9].text(
                        (x1 + x2) * 0.5,
                        y + h,
                        is_node3_mp_mm_sign,
                        ha="center",
                        va=va_text_sign,
                        color=col,
                    )
                    y_add = 2

                if is_node3_pp_mp_sign != None:
                    y, h, col = (
                        max_val
                        + 2
                        + (y_add * height_p_sign)
                        + int((y_add * height_p_sign) / 3),
                        height_p_sign,
                        "k",
                    )
                    x1, x2 = 1, 3
                    axis[9].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                    axis[9].text(
                        (x1 + x2) * 0.5,
                        y + h,
                        is_node3_pp_mp_sign,
                        ha="center",
                        va=va_text_sign,
                        color=col,
                    )
                    y_add += 1

                if is_node3_pm_mm_sign != None:
                    y, h, col = (
                        max_val
                        + 2
                        + (y_add * height_p_sign)
                        + int((y_add * height_p_sign) / 3),
                        height_p_sign,
                        "k",
                    )
                    x1, x2 = 0, 2
                    axis[9].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                    axis[9].text(
                        (x1 + x2) * 0.5,
                        y + h,
                        is_node3_pm_mm_sign,
                        ha="center",
                        va=va_text_sign,
                        color=col,
                    )
                    y_add += 1

                if is_node3_pm_mp_sign != None:
                    y, h, col = (
                        max_val
                        + 2
                        + (y_add * height_p_sign)
                        + int((y_add * height_p_sign) / 3),
                        height_p_sign,
                        "k",
                    )
                    x1, x2 = 1, 2
                    axis[9].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                    axis[9].text(
                        (x1 + x2) * 0.5,
                        y + h,
                        is_node3_pm_mp_sign,
                        ha="center",
                        va=va_text_sign,
                        color=col,
                    )
                    y_add += 1

                if is_node3_pp_mm_sign != None:
                    y, h, col = (
                        max_val
                        + 2
                        + (y_add * height_p_sign)
                        + int((y_add * height_p_sign) / 3),
                        height_p_sign,
                        "k",
                    )
                    x1, x2 = 0, 3
                    axis[9].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                    axis[9].text(
                        (x1 + x2) * 0.5,
                        y + h,
                        is_node3_pp_mm_sign,
                        ha="center",
                        va=va_text_sign,
                        color=col,
                    )
                    y_add += 1
                axis[9].set_xticklabels(
                    [
                        "",
                        "",
                        "",
                        "",
                        TF_1 + " - " + TF_3 + " - ",
                        TF_1 + " - " + TF_3 + " + ",
                        TF_1 + " + " + TF_3 + " - ",
                        TF_1 + " + " + TF_3 + " + ",
                    ]
                )
                axis[9].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
                axis[10].set_xticklabels(
                    [
                        TF_1 + " - " + TF_3 + " - ",
                        TF_1 + " - " + TF_3 + " + ",
                        TF_1 + " + " + TF_3 + " - ",
                        TF_1 + " + " + TF_3 + " + ",
                    ]
                )
                axis[10].xaxis.set_tick_params(labelsize=xaxis_label_size, rotation=10)
                cell_sup_sup = list(
                    set(self.Y_norm.index.values).intersection(
                        set(TF_1_sup_TF_3_sup.index.values)
                    )
                )
                cell_sup_inf = list(
                    set(self.Y_norm.index.values).intersection(
                        set(TF_1_sup_TF_3_inf.index.values)
                    )
                )
                cell_inf_sup = list(
                    set(self.Y_norm.index.values).intersection(
                        set(TF_1_inf_TF_3_sup.index.values)
                    )
                )
                cell_inf_inf = list(
                    set(self.Y_norm.index.values).intersection(
                        set(TF_1_inf_TF_3_inf.index.values)
                    )
                )

                i_cell_sup_sup = [ind_dict[x] for x in cell_sup_sup]
                i_cell_sup_inf = [ind_dict[x] for x in cell_sup_inf]
                i_cell_inf_sup = [ind_dict[x] for x in cell_inf_sup]
                i_cell_inf_inf = [ind_dict[x] for x in cell_inf_inf]

                axis[11].scatter(
                    x=embedded_data[:, 0][i_cell_sup_sup],
                    y=embedded_data[:, 1][i_cell_sup_sup],
                    s=1,
                    color=colors_palette[::-1][0],
                    marker=".",
                    alpha=0.05,
                )
                axis[11].scatter(
                    x=embedded_data[:, 0][i_cell_sup_inf],
                    y=embedded_data[:, 1][i_cell_sup_inf],
                    s=1,
                    color=colors_palette[::-1][1],
                    marker=".",
                    alpha=0.05,
                )
                axis[11].scatter(
                    x=embedded_data[:, 0][i_cell_inf_sup],
                    y=embedded_data[:, 1][i_cell_inf_sup],
                    s=1,
                    color=colors_palette[::-1][2],
                    marker=".",
                    alpha=0.05,
                )
                axis[11].scatter(
                    x=embedded_data[:, 0][i_cell_inf_inf],
                    y=embedded_data[:, 1][i_cell_inf_inf],
                    s=1,
                    color=colors_palette[::-1][3],
                    marker=".",
                    alpha=0.05,
                )
        if full_first_layer:
            plt.suptitle(
                str(TG)
                + "\nCART Threshold : \n"
                + "{:<10s}{:<4s}\n".format(str(TF_1), str(lim_val_1))
                + "{:<10s}{:<4s}\n".format(str(TF_2), str(lim_val_2))
                + "{:<10s}{:<4s}\n".format(str(TF_3), str(lim_val_3))
            )
        elif show_first_layer_res:
            plt.suptitle(
                str(TG)
                + "\nCART Threshold : \n"
                + "{:<10s}{:<4s}\n".format(str(TF_1), str(lim_val_1))
                + "{:<10s}{:<4s}\n".format(str(TF_2), str(lim_val_2))
            )
        else:
            plt.suptitle(
                str(TG)
                + "\nCART Threshold : \n"
                + "{:<10s}{:<4s}\n".format(str(TF_1), str(lim_val_1))
            )
        if plot_method == "show":
            plt.show()
        elif plot_method == "print":
            plt.savefig(self.save_dir_path + "CARTLUPLOT/" + str(TG) + ".png")
        else:
            raise (NotImplementedError())

    def save_compiled_results(self, compiled_row=None, append=False):
        if compiled_row is not None:
            if append == True:
                compiled_row.to_csv(
                    self.save_dir_path + "compiled_table_a.csv",
                    header=False,
                    index=False,
                    sep=",",
                    mode="a",
                )
            else:
                self.compiled_table.to_csv(
                    self.save_dir_path + "compiled_table.csv",
                    header=True,
                    index=False,
                    sep=",",
                )
        else:
            self.compiled_table.to_csv(
                self.save_dir_path + "compiled_table.csv",
                header=True,
                index=False,
                sep=",",
            )

    def compile_cart_results(self, Y_id):
        compiled_row = pd.DataFrame(
            {
                "AGI": [Y_id],
                "model_score": [self.model_evaluation["accuracy"]],
                "gini_score_0": [None],
                "gini_score_1": [None],
                "gini_score_2": [None],
                "TF1": [None],
                "TF2": [None],
                "TF3": [None],
                "sign1": [None],
                "sign2": [None],
                "sign3": [None],
                "lim1": [None],
                "lim2": [None],
                "lim3": [None],
                "N2 anova p-val_1": [None],
                "N2 anova p-val_2": [None],
                "N2 anova p-val_1-2": [None],
                "N3 anova p-val_1": [None],
                "N3 anova p-val_2": [None],
                "N3 anova p-val_1-2": [None],
                "N2_p-val_++_+-": [None],
                "N2_p-val_++_-+": [None],
                "N2_p-val_++_--": [None],
                "N2_p-val_+-_-+": [None],
                "N2_p-val_+-_--": [None],
                "N2_p-val_-+_--": [None],
                "N3_p-val_++_+-": [None],
                "N3_p-val_++_-+": [None],
                "N3_p-val_++_--": [None],
                "N3_p-val_+-_-+": [None],
                "N3_p-val_+-_--": [None],
                "N3_p-val_-+_--": [None],
                "mean_1+": [None],
                "mean_1-": [None],
                "mean_1+2+": [None],
                "mean_1-2+": [None],
                "mean_1+2-": [None],
                "mean_1-2-": [None],
                "mean_1+3+": [None],
                "mean_1-3+": [None],
                "mean_1+3-": [None],
                "mean_1-3-": [None],
                "perc_zero_train": [self.model_evaluation["perc_0_train"]],
                "perc_zero_test": [self.model_evaluation["perc_0_test"]],
                "perc_zero_pred": [self.model_evaluation["perc_0_pred"]],
                "perc_zero_tot": [self.model_evaluation["perc_0_total"]],
            }
        )
        cond = self.tree_rules[0][(self.tree_rules[0].find("|--- ") + len("|--- ")) :]
        node_i_list = []
        i = 1
        if cond.find(" <=") != -1:
            compiled_row.loc[0, self.X_txt + "1"] = cond[: cond.find(" <=")]
            compiled_row.loc[0, "lim1"] = float(cond[cond.find(" <=") + 4 : -1])
            compiled_row.loc[0, "sign1"] = "<="
            compiled_row.loc[0, "gini_score_0"] = self._CART_scoring.loc[
                0, "gini_score"
            ]
            node_i_list.append(1)

        for line in self.tree_rules:
            if re.search("^\|   \|--- ", line):
                pass
            else:
                continue
            cond = line[(line.find("|   |--- ") + len("|   |--- ")) :]
            if cond.find(" <=") != -1:
                compiled_row.loc[0, self.X_txt + str(i + 1)] = cond[: cond.find(" <=")]
                compiled_row.loc[0, "lim" + str(i + 1)] = float(
                    cond[cond.find(" <=") + 4 : -1]
                )
                compiled_row.loc[0, "sign" + str(i + 1)] = "<="
                compiled_row.loc[0, "gini_score_" + str(i)] = self._CART_scoring.loc[
                    i, "gini_score"
                ]
                node_i_list.append(i + 1)
            else:
                continue
            i += 1
            if i > 3:
                break
        if len(node_i_list) > 0:
            TF_1_sup = self.Y.loc[
                self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                > compiled_row["lim" + str(node_i_list[0])].values[0],
                compiled_row[self.Y_txt],
            ]
            TF_1_inf = self.Y.loc[
                self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                <= compiled_row["lim" + str(node_i_list[0])].values[0],
                compiled_row[self.Y_txt],
            ]
            t_test_res = stats.mannwhitneyu(
                x=np.nan_to_num(np.log10(TF_1_sup.values), neginf=0),
                y=np.nan_to_num(np.log10(TF_1_inf.values), neginf=0),
                alternative="two-sided",
            )
            compiled_row.loc[0, "p-val_1"] = list(t_test_res)[1]
            compiled_row.loc[0, "mean_1+"] = TF_1_sup.mean().values
            compiled_row.loc[0, "mean_1-"] = TF_1_inf.mean().values
            # Compute anova-test on (TF1 sup / TF1 inf) and (TF2 sup / TF2 inf)
            if len(node_i_list) > 1:
                # cond2 = compiled_row[compiled_row["node"]==node_i_list[1]]
                TF1_TF2 = pd.DataFrame(
                    {
                        "target": [0] * len(self.Y.index),
                        self.X_txt + "1": [0] * len(self.Y.index),
                        self.X_txt + "2": [0] * len(self.Y.index),
                    }
                )
                TF1_TF2.index = self.Y.index
                TF1_TF2["target"] = self.Y.loc[:, compiled_row[self.Y_txt]]
                TF1_TF2.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        > compiled_row["lim" + str(node_i_list[0])].values[0]
                    ),
                    self.X_txt + "1",
                ] = 1
                TF1_TF2.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[1])].values[0]]
                        > compiled_row["lim" + str(node_i_list[1])].values[0]
                    ),
                    self.X_txt + "2",
                ] = 1
                # ANOVA TW TEST
                model = ols(
                    "target ~ C("
                    + self.X_txt
                    + "1) + C("
                    + self.X_txt
                    + "2) + C("
                    + self.X_txt
                    + "1):C("
                    + self.X_txt
                    + "2)",
                    data=TF1_TF2,
                ).fit()
                result = sm.stats.anova_lm(model, type=2)
                compiled_row.loc[0, "N2 anova p-val_1"] = result.loc[
                    "C(" + self.X_txt + "1)", "PR(>F)"
                ]
                compiled_row.loc[0, "N2 anova p-val_2"] = result.loc[
                    "C(" + self.X_txt + "2)", "PR(>F)"
                ]
                compiled_row.loc[0, "N2 anova p-val_1-2"] = result.loc[
                    "C(" + self.X_txt + "1):C(" + self.X_txt + "2)", "PR(>F)"
                ]

                # TUKEY TEST

                TF_1_sup_TF_2_sup = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        > compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[1])].values[0]]
                        > compiled_row["lim" + str(node_i_list[1])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]  # .to_numpy().flatten().tolist()
                TF_1_sup_TF_2_inf = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        > compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[1])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[1])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]  # .to_numpy().flatten().tolist()
                TF_1_inf_TF_2_sup = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[1])].values[0]]
                        > compiled_row["lim" + str(node_i_list[1])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]  # .to_numpy().flatten().tolist()
                TF_1_inf_TF_2_inf = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[1])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[1])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]  # .to_numpy().flatten().tolist()

                # TF_1_sup_TF_2_sup['sign'] = "++"
                # TF_1_sup_TF_2_inf['sign'] = "+-"
                # TF_1_inf_TF_2_sup['sign'] = "-+"
                # TF_1_inf_TF_2_inf['sign'] = "--"
                # tukey_data = TF_1_sup_TF_2_sup.append([TF_1_sup_TF_2_sup, TF_1_sup_TF_2_inf, TF_1_inf_TF_2_sup, TF_1_inf_TF_2_inf])
                # result = pairwise_tukeyhsd(tukey_datcompiled_row[cond[Y_txt]], groups=tukey_datcompiled_row['sign'])
                pp_pm = 1.0
                pp_mp = 1.0
                pp_mm = 1.0
                pm_mp = 1.0
                pm_mm = 1.0
                mp_mm = 1.0
                if not TF_1_sup_TF_2_sup[compiled_row[self.Y_txt]].empty:
                    if not TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]].empty:
                        pp_pm = stats.mannwhitneyu(
                            TF_1_sup_TF_2_sup[compiled_row[self.Y_txt]],
                            TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                    if not TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]].empty:
                        pp_mp = stats.mannwhitneyu(
                            TF_1_sup_TF_2_sup[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                    if not TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]].empty:
                        pp_mm = stats.mannwhitneyu(
                            TF_1_sup_TF_2_sup[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                if not TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]].empty:
                    if not TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]].empty:
                        pm_mp = stats.mannwhitneyu(
                            TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                    if not TF_1_inf_TF_2_inf[compiled_row[self.Y_txt]].empty:
                        pm_mm = stats.mannwhitneyu(
                            TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_2_inf[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                if not TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]].empty:
                    if not TF_1_inf_TF_2_inf[compiled_row[self.Y_txt]].empty:
                        mp_mm = stats.mannwhitneyu(
                            TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_2_inf[compiled_row[self.Y_txt]],
                        ).pvalue[0]

                compiled_row.loc[0, "mean_1+2+"] = TF_1_sup_TF_2_sup.mean().values
                compiled_row.loc[0, "mean_1-2+"] = TF_1_inf_TF_2_sup.mean().values
                compiled_row.loc[0, "mean_1+2-"] = TF_1_sup_TF_2_inf.mean().values
                compiled_row.loc[0, "mean_1-2-"] = TF_1_inf_TF_2_inf.mean().values
                i = 0
                if len(TF_1_sup_TF_2_sup.index) > 0:
                    if len(TF_1_sup_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_++_+-"] = pp_pm
                        i = i + 1
                    if len(TF_1_inf_TF_2_sup.index) > 0:
                        compiled_row.loc[0, "N2_p-val_++_-+"] = pp_mp
                        i = i + 1
                    if len(TF_1_inf_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_++_--"] = pp_mm
                        i = i + 1
                if len(TF_1_sup_TF_2_inf.index) > 0:
                    if len(TF_1_inf_TF_2_sup.index) > 0:
                        compiled_row.loc[0, "N2_p-val_+-_-+"] = pm_mp
                        i = i + 1
                    if len(TF_1_inf_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_+-_--"] = pm_mm
                        i = i + 1
                if len(TF_1_inf_TF_2_sup.index) > 0:
                    if len(TF_1_inf_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_-+_--"] = mp_mm
                        i = i + 1
            if len(node_i_list) > 2:
                # cond3 = compiled_row[compiled_row["node"]==node_i_list[2]]
                TF1_TF3 = pd.DataFrame(
                    {
                        "target": [0] * len(self.Y.index),
                        self.X_txt + "1": [0] * len(self.Y.index),
                        self.X_txt + "3": [0] * len(self.Y.index),
                    }
                )
                TF1_TF3.index = self.Y.index
                TF1_TF3["target"] = self.Y.loc[:, compiled_row[self.Y_txt]]
                TF1_TF3.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        > compiled_row["lim" + str(node_i_list[0])].values[0]
                    ),
                    self.X_txt + "1",
                ] = 1
                TF1_TF3.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[2])].values[0]]
                        > compiled_row["lim" + str(node_i_list[2])].values[0]
                    ),
                    self.X_txt + "3",
                ] = 1

                # ANOVA TW TEST
                model = ols(
                    "target ~ C("
                    + self.X_txt
                    + "1) + C("
                    + self.X_txt
                    + "3) + C("
                    + self.X_txt
                    + "1):C("
                    + self.X_txt
                    + "3)",
                    data=TF1_TF3,
                ).fit()
                result = sm.stats.anova_lm(model, type=2)
                compiled_row.loc[0, "N3 anova p-val_1"] = result.loc[
                    "C(" + self.X_txt + "1)", "PR(>F)"
                ]
                compiled_row.loc[0, "N3 anova p-val_2"] = result.loc[
                    "C(" + self.X_txt + "3)", "PR(>F)"
                ]
                compiled_row.loc[0, "N3 anova p-val_1-2"] = result.loc[
                    "C(" + self.X_txt + "1):C(" + self.X_txt + "3)", "PR(>F)"
                ]

                # TUKEY TEST
                TF_1_sup_TF_3_sup = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        > compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[2])].values[0]]
                        > compiled_row["lim" + str(node_i_list[2])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]
                TF_1_sup_TF_3_inf = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        > compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[2])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[2])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]
                TF_1_inf_TF_3_sup = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[2])].values[0]]
                        > compiled_row["lim" + str(node_i_list[2])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]
                TF_1_inf_TF_3_inf = self.Y.loc[
                    (
                        self.X[compiled_row[self.X_txt + str(node_i_list[0])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[0])].values[0]
                    )
                    & (
                        self.X[compiled_row[self.X_txt + str(node_i_list[2])].values[0]]
                        <= compiled_row["lim" + str(node_i_list[2])].values[0]
                    ),
                    compiled_row[self.Y_txt],
                ]
                pp_pm = 1.0
                pp_mp = 1.0
                pp_mm = 1.0
                pm_mp = 1.0
                pm_mm = 1.0
                mp_mm = 1.0

                if not TF_1_sup_TF_3_sup[compiled_row[self.Y_txt]].empty:
                    if not TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]].empty:
                        pp_pm = stats.mannwhitneyu(
                            TF_1_sup_TF_3_sup[compiled_row[self.Y_txt]],
                            TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                    if not TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]].empty:
                        pp_mp = stats.mannwhitneyu(
                            TF_1_sup_TF_3_sup[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                    if not TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]].empty:
                        pp_mm = stats.mannwhitneyu(
                            TF_1_sup_TF_3_sup[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                if TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]].empty:
                    if TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]].empty:
                        pm_mp = stats.mannwhitneyu(
                            TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                    if TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]].empty:
                        pm_mm = stats.mannwhitneyu(
                            TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                if TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]].empty:
                    if TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]].empty:
                        mp_mm = stats.mannwhitneyu(
                            TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]],
                            TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]],
                        ).pvalue[0]
                compiled_row.loc[0, "mean_1+3+"] = TF_1_sup_TF_3_sup.mean().values
                compiled_row.loc[0, "mean_1-3+"] = TF_1_inf_TF_3_sup.mean().values
                compiled_row.loc[0, "mean_1+3-"] = TF_1_sup_TF_3_inf.mean().values
                compiled_row.loc[0, "mean_1-3-"] = TF_1_inf_TF_3_inf.mean().values

                i = 0
                if len(TF_1_sup_TF_3_sup.index) > 0:
                    if len(TF_1_sup_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_++_+-"] = pp_pm
                        i = i + 1
                    if len(TF_1_inf_TF_3_sup.index) > 0:
                        compiled_row.loc[0, "N3_p-val_++_-+"] = pp_mp
                        i = i + 1
                    if len(TF_1_inf_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_++_--"] = pp_mm
                        i = i + 1
                if len(TF_1_sup_TF_3_inf.index) > 0:
                    if len(TF_1_inf_TF_3_sup.index) > 0:
                        compiled_row.loc[0, "N3_p-val_+-_-+"] = pm_mp
                        i = i + 1
                    if len(TF_1_inf_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_+-_--"] = pm_mm
                        i = i + 1
                if len(TF_1_inf_TF_3_sup.index) > 0:
                    if len(TF_1_inf_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_-+_--"] = mp_mm
                        i = i + 1
            self.compiled_table = pd.concat([self.compiled_table, compiled_row])
            self.save_compiled_results(compiled_row, append=True)
        self.compiled_row = compiled_row.copy(deep=True)
        self.end = time.time()
        self._time_per_tree = max(self.end - self.start, self._time_per_tree)

    def filter_cart_results(
        self, sort_by=[],
    ):
        self.filtered_table = self.compiled_table.copy(deep=True)
        sort_order = [True] * len(self.SORT_ORDER_REF.keys())
        for ielem, elem in zip(range(len(sort_by)), sort_by):
            sort_order[ielem] = self.SORT_ORDER_REF[elem]
        # data refactor + reshaping
        self.filtered_table["ratio"] = self.filtered_table["perc_zero_tot"]
        self.filtered_table["ratio_mean"] = (
            self.filtered_table["mean_1+"] + self.filtered_table["mean_1-"]
        ) / 2
        self.filtered_table["p-val_1"] = -np.log10(
            self.filtered_table["p-val_1"].values
        )
        self.filtered_table["N2 anova p-val_1-2"] = -np.log10(
            self.filtered_table["N2 anova p-val_1-2"].to_numpy().astype(np.float64())
        )
        self.filtered_table["N3 anova p-val_1-2"] = -np.log10(
            self.filtered_table["N3 anova p-val_1-2"].to_numpy().astype(np.float64())
        )
        # data filtration
        self.filtered_table = self.filtered_table.loc[
            self.filtered_table[
                (
                    (
                        (self.filtered_table["gini_score_0"] <= self.THRES_CRITERION)
                        & (self.filtered_table["gini_score_1"] <= self.THRES_CRITERION)
                        & (self.filtered_table["gini_score_2"] <= self.THRES_CRITERION)
                    )
                    & (self.filtered_table["model_score"] >= self.THRES_MODEL)
                    & (self.filtered_table["ratio"] <= self.THRES_ZERO_TE_TR)
                )
                & (
                    (self.filtered_table["p-val_1"] >= self.THRES_PVAL)
                    | (self.filtered_table["N2 anova p-val_1-2"] >= self.THRES_PVAL)
                    | (self.filtered_table["N3 anova p-val_1-2"] >= self.THRES_PVAL)
                )
            ].index
        ]
        # data sorting
        if len(sort_by) > 0:
            self.filtered_table.sort_values(
                by=sort_by, ascending=sort_order, inplace=True
            )
        self.candidate = self.filtered_table[self.Y_txt].to_numpy()
        print(
            "\n Done filtering : " + str(len(self.candidate)) + " candidates remains.\n"
        )

    def save_filtered_results(self):
        with open(self.save_dir_path + "filtered_candidates.txt", "w") as file:
            for target_candidates in self.candidate:
                file.write(target_candidates + "\n")
