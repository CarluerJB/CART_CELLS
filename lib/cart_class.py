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
        self.create_out_dir()
        self.load_GE_matrix()
        self.load_tf_list()
        self.analysis_type = analysis_type
        if self.analysis_type == "genes":
            self.Y_txt = "AGI"
            self.X_txt = "TF"
            self.Y = self.ge_matrix.drop(self.tf_list).transpose()
            self.X = self.ge_matrix.loc[self.tf_list].transpose()
        else:
            self.X = self.ge_matrix.drop(self.tf_list).transpose()
            self.Y = self.ge_matrix.loc[self.tf_list].transpose()
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
                    self._perc_cart_tree_interface * 100 / len(self.Y.columns[:100]), 3
                ),
                round(
                    len(self.Y.columns[:100]) * self._time_per_tree
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

    def save_compiled_results(self):
        self.compiled_table.to_csv(
            self.save_dir_path + "compiled_table.csv", header=True, index=False, sep=","
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
        if cond.find(" <=") != -1:
            compiled_row.loc[0, self.X_txt + "1"] = cond[: cond.find(" <=")]
            compiled_row.loc[0, "lim1"] = float(cond[cond.find(" <=") + 4 : -2])
            compiled_row.loc[0, "sign1"] = "<="
            compiled_row.loc[0, "gini_score_0"] = self._CART_scoring.loc[
                0, "gini_score"
            ]
        i = 1
        node_i_list = [1]
        for line in self.tree_rules:
            if re.search("^\|   \|--- ", line):
                pass
            else:
                continue
            cond = line[(line.find("|   |--- ") + len("|   |--- ")) :]
            if cond.find(" <=") != -1:
                compiled_row.loc[0, self.X_txt + str(i + 1)] = cond[: cond.find(" <=")]
                compiled_row.loc[0, "lim" + str(i + 1)] = float(
                    cond[cond.find(" <=") + 4 : -2]
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
                pp_pm = stats.mannwhitneyu(
                    TF_1_sup_TF_2_sup[compiled_row[self.Y_txt]],
                    TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]],
                )
                pp_mp = stats.mannwhitneyu(
                    TF_1_sup_TF_2_sup[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]],
                )
                pp_mm = stats.mannwhitneyu(
                    TF_1_sup_TF_2_sup[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_2_inf[compiled_row[self.Y_txt]],
                )
                pm_mp = stats.mannwhitneyu(
                    TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]],
                )
                pm_mm = stats.mannwhitneyu(
                    TF_1_sup_TF_2_inf[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_2_inf[compiled_row[self.Y_txt]],
                )
                mp_mm = stats.mannwhitneyu(
                    TF_1_inf_TF_2_sup[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_2_inf[compiled_row[self.Y_txt]],
                )

                compiled_row.loc[0, "mean_1+2+"] = TF_1_sup_TF_2_sup.mean().values
                compiled_row.loc[0, "mean_1-2+"] = TF_1_inf_TF_2_sup.mean().values
                compiled_row.loc[0, "mean_1+2-"] = TF_1_sup_TF_2_inf.mean().values
                compiled_row.loc[0, "mean_1-2-"] = TF_1_inf_TF_2_inf.mean().values
                i = 0
                if len(TF_1_sup_TF_2_sup.index) > 0:
                    if len(TF_1_sup_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_++_+-"] = pp_pm.pvalue[0]
                        i = i + 1
                    if len(TF_1_inf_TF_2_sup.index) > 0:
                        compiled_row.loc[0, "N2_p-val_++_-+"] = pp_mp.pvalue[0]
                        i = i + 1
                    if len(TF_1_inf_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_++_--"] = pp_mm.pvalue[0]
                        i = i + 1
                if len(TF_1_sup_TF_2_inf.index) > 0:
                    if len(TF_1_inf_TF_2_sup.index) > 0:
                        compiled_row.loc[0, "N2_p-val_+-_-+"] = pm_mp.pvalue[0]
                        i = i + 1
                    if len(TF_1_inf_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_+-_--"] = pm_mm.pvalue[0]
                        i = i + 1
                if len(TF_1_inf_TF_2_sup.index) > 0:
                    if len(TF_1_inf_TF_2_inf.index) > 0:
                        compiled_row.loc[0, "N2_p-val_-+_--"] = mp_mm.pvalue[0]
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

                # TF_1_sup_TF_3_sup['sign'] = "++"
                # TF_1_sup_TF_3_inf['sign'] = "+-"
                # TF_1_inf_TF_3_sup['sign'] = "-+"
                # TF_1_inf_TF_3_inf['sign'] = "--"
                # tukey_data = TF_1_sup_TF_3_sup.append([TF_1_sup_TF_3_sup, TF_1_sup_TF_3_inf, TF_1_inf_TF_3_sup, TF_1_inf_TF_3_inf])
                # result = pairwise_tukeyhsd(tukey_datcompiled_row[cond[Y_txt]], groups=tukey_datcompiled_row['sign'])
                pp_pm = stats.mannwhitneyu(
                    TF_1_sup_TF_3_sup[compiled_row[self.Y_txt]],
                    TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]],
                )
                pp_mp = stats.mannwhitneyu(
                    TF_1_sup_TF_3_sup[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]],
                )
                pp_mm = stats.mannwhitneyu(
                    TF_1_sup_TF_3_sup[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]],
                )
                pm_mp = stats.mannwhitneyu(
                    TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]],
                )
                pm_mm = stats.mannwhitneyu(
                    TF_1_sup_TF_3_inf[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]],
                )
                mp_mm = stats.mannwhitneyu(
                    TF_1_inf_TF_3_sup[compiled_row[self.Y_txt]],
                    TF_1_inf_TF_3_inf[compiled_row[self.Y_txt]],
                )

                compiled_row.loc[0, "mean_1+3+"] = TF_1_sup_TF_3_sup.mean().values
                compiled_row.loc[0, "mean_1-3+"] = TF_1_inf_TF_3_sup.mean().values
                compiled_row.loc[0, "mean_1+3-"] = TF_1_sup_TF_3_inf.mean().values
                compiled_row.loc[0, "mean_1-3-"] = TF_1_inf_TF_3_inf.mean().values
                i = 0
                if len(TF_1_sup_TF_3_sup.index) > 0:
                    if len(TF_1_sup_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_++_+-"] = pp_pm.pvalue[0]
                        i = i + 1
                    if len(TF_1_inf_TF_3_sup.index) > 0:
                        compiled_row.loc[0, "N3_p-val_++_-+"] = pp_mp.pvalue[0]
                        i = i + 1
                    if len(TF_1_inf_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_++_--"] = pp_mm.pvalue[0]
                        i = i + 1
                if len(TF_1_sup_TF_3_inf.index) > 0:
                    if len(TF_1_inf_TF_3_sup.index) > 0:
                        compiled_row.loc[0, "N3_p-val_+-_-+"] = pm_mp.pvalue[0]
                        i = i + 1
                    if len(TF_1_inf_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_+-_--"] = pm_mm.pvalue[0]
                        i = i + 1
                if len(TF_1_inf_TF_3_sup.index) > 0:
                    if len(TF_1_inf_TF_3_inf.index) > 0:
                        compiled_row.loc[0, "N3_p-val_-+_--"] = mp_mm.pvalue[0]
                        i = i + 1
        self.compiled_table = pd.concat([self.compiled_table, compiled_row])
        self.end = time.time()
        self._time_per_tree = max(self.end - self.start, self._time_per_tree)

    def filter_cart_results(
        self,
        sort_by=[],
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
        # self.compiled_table = self.compiled_table.loc[
        #     self.compiled_table[
        #         (
        #             (
        #                 (self.compiled_table["gini_score_0"] <= self.THRES_CRITERION)
        #                 & (self.compiled_table["gini_score_1"] <= self.THRES_CRITERION)
        #                 & (self.compiled_table["gini_score_2"] <= self.THRES_CRITERION)
        #             )
        #             & (self.compiled_table["model_score"] >= self.THRES_MODEL)
        #             & (self.compiled_table["ratio"] <= self.THRES_ZERO_TE_TR)
        #         )
        #         & (
        #             (self.compiled_table["p-val_1"] >= self.THRES_PVAL)
        #             | (self.compiled_table["N2 anova p-val_1-2"] >= self.THRES_PVAL)
        #             | (self.compiled_table["N3 anova p-val_1-2"] >= self.THRES_PVAL)
        #         )
        #     ].index
        # ]

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
