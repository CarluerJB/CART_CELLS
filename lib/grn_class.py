# ===============================
# AUTHOR     : CARLUER Jean-Baptiste
# CREATE DATE     : 2022-2023
# PURPOSE     : Thesis in BioInformatics
# SPECIAL NOTES:
# ===============================
# Change History:
#
# # =================================

from cmath import log
from codecs import ignore_errors
from logging import raiseExceptions
from tkinter.tix import COLUMN
import networkx as nx
import py4cytoscape as py4
import pandas as pd
import os
from unittest.mock import patch
import sys
from lib.utils_class import NoStdStreams
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan
import seaborn as sns
from umap import UMAP

RANDOM_STATE = 42


class GRN:
    def __init__(self, out_data, TF_AGI_CORR=None):
        self.G = nx.Graph()
        self.node_size = []
        self.node_color = []
        self.node_style = []

        self.save_dir_path = out_data
        self.parameter_file_path = "PARAMETERS/PARAM_GRN_DEFAULT.txt"
        self.load_parameter_file()
        self.path_node_table = (
            self.save_dir_path + "network/" + self.node_table_filename
        )
        self.path_edge_table = (
            self.save_dir_path + "network/" + self.edge_table_filename
        )
        self.path_edge_simplified_table = (
            self.save_dir_path + "network/" + self.edge_simplified_table_filename
        )
        self.path_resume_table = (
            self.save_dir_path + "network/" + self.resume_table_filename
        )
        self.path_resume_target = (
            self.save_dir_path + "network/" + self.tf_target_filename
        )
        self.path_src_target_dir = self.save_dir_path + "network/genes_study/"

        if TF_AGI_CORR == None:
            self.LIST_TF = {  # TEMPORARY FIND A SMART WAY TO REMOVE IT FROM CODE !
                "BEE2": "AT4G36540",
                "CDF1": "AT5G62430",
                "bZIP3": "AT5G15830",
                "COL5": "AT5G57660",
                "CRF4": "AT4G27950",
                "ERF5": "AT5G47230",
                "GATA17": "AT3G16870",
                "HAT22": "AT4G37790",
                "HB6": "AT2G22430",
                "HHO2": "AT1G68670",
                "HHO3": "AT1G25550",
                "LBD37": "AT5G67420",
                "LBD38": "AT3G49940",
                "MYB61": "AT1G09540",
                "NAC4": "AT5G07680",
                "NAP": "AT1G69490",
                "RAV1": "AT1G13260",
                "TCP23": "AT1G35560",
                "TGA1": "AT5G65210",
                "TGA4": "AT5G10030",
                "VRN1": "AT3G18990",
                "WRKY18": "AT4G31800",
                "WRKY54": "AT2G40750",
            }
        else:
            temp_TG_TF = pd.read_table(
                TF_AGI_CORR, sep=",", header=None, names=["gene", "AGI"]
            )
            self.LIST_TF = dict(zip(temp_TG_TF.gene, temp_TG_TF.AGI))
        self.create_out_dir("network")
        self.create_out_dir("network/evaluateNet")
        self.create_out_dir("network/genes_study")
        self.create_out_dir("network/association_study")
        self.show_parameter()

    def delete_GRN(self):
        self.G = None
        self.G = nx.Graph()
        self.node_size = []
        self.node_color = []
        self.node_style = []

    def load_parameter_file(self):
        with open(self.parameter_file_path, "r") as parameter_file:
            for line in parameter_file:
                if ("#" in line) or (line == "\n"):
                    continue
                target_param, param = line.split(" ")
                if target_param == "TF_SHAPE":
                    self.TF_SHAPE = param[:-1]
                if target_param == "GENE_SHAPE":
                    self.GENE_SHAPE = param[:-1]
                if target_param == "TF_INTER_SHAPE":
                    self.TF_INTER_SHAPE = param[:-1]
                if target_param == "TF_INTER_LINE_TYPE":
                    self.TF_INTER_LINE_TYPE = param[:-1]
                if target_param == "DEFAULT_LINE_TYPE":
                    self.DEFAULT_LINE_TYPE = param[:-1]
                if target_param == "TF_NODE_SIZE":
                    self.TF_NODE_SIZE = int(param)
                if target_param == "GENE_NODE_SIZE":
                    self.GENE_NODE_SIZE = int(param)
                if target_param == "TF_INTER_NODE_SIZE":
                    self.TF_INTER_NODE_SIZE = int(param)
                if target_param == "TF_NODE_COLOR":
                    self.TF_NODE_COLOR = param[:-1]
                if target_param == "GENE_NODE_COLOR":
                    self.GENE_NODE_COLOR = param[:-1]
                if target_param == "TF_INTER_NODE_COLOR":
                    self.TF_INTER_NODE_COLOR = param[:-1]
                if target_param == "DIR_EDGE_SIZE":
                    self.DIR_EDGE_SIZE = int(param)
                if target_param == "UNDIR_EDGE_SIZE":
                    self.UNDIR_EDGE_SIZE = int(param)
                if target_param == "DIR_EDGE_COLOR":
                    self.DIR_EDGE_COLOR = param[:-1]
                if target_param == "UNDIR_EDGE_COLOR":
                    self.UNDIR_EDGE_COLOR = param[:-1]
                if target_param == "THRES_CRITERION":
                    self.THRES_CRITERION = float(param)
                if target_param == "PERC_ZERO_TOT":
                    self.PERC_ZERO_TOT = float(param)
                if target_param == "PVAL":
                    self.PVAL = float(param)
                if target_param == "MODEL_SCORE":
                    self.MODEL_SCORE = float(param)
                if target_param == "node_table_filename":
                    self.node_table_filename = param[:-1]
                if target_param == "edge_table_filename":
                    self.edge_table_filename = param[:-1]
                if target_param == "edge_simplified_table_filename":
                    self.edge_simplified_table_filename = param[:-1]
                if target_param == "resume_table_filename":
                    self.resume_table_filename = param[:-1]
                if target_param == "tf_target_filename":
                    self.tf_target_filename = param[:-1]
                if target_param == "tf_all_list_filename":
                    self.tf_all_list_filename = param[:-1]
                if target_param == "gene_all_list_filename":
                    self.gene_all_list_filename = param[:-1]
                if target_param == "eval_score_filename":
                    self.eval_score_filename = param[:-1]
                if target_param == "eval_filename":
                    self.eval_filename = param[:-1]
                if target_param == "eval_score_plot_filename":
                    self.eval_score_plot_filename = param[:-1]
                if target_param == "eval_plot_filename":
                    self.eval_plot_filename = param[:-1]
                if target_param == "found_edges_info_out_path":
                    self.found_edges_info_out_path = param[:-1]

    def show_parameter(self):
        print("\n\tPARAMETERS : ")
        print("\t\tTF_SHAPE : ", self.TF_SHAPE)
        print("\t\tGENE_SHAPE : ", self.GENE_SHAPE)
        print("\t\tTF_INTER_SHAPE : ", self.TF_INTER_SHAPE)
        print("\t\tTF_INTER_LINE_TYPE : ", self.TF_INTER_LINE_TYPE)
        print("\t\tDEFAULT_LINE_TYPE : ", self.DEFAULT_LINE_TYPE)
        print("\t\tTF_NODE_SIZE : ", self.TF_NODE_SIZE)
        print("\t\tGENE_NODE_SIZE : ", self.GENE_NODE_SIZE)
        print("\t\tTF_INTER_NODE_SIZE : ", self.TF_INTER_NODE_SIZE)
        print("\t\tTF_NODE_COLOR : ", self.TF_NODE_COLOR)
        print("\t\tGENE_NODE_COLOR : ", self.GENE_NODE_COLOR)
        print("\t\tTF_INTER_NODE_COLOR : ", self.TF_INTER_NODE_COLOR)
        print("\t\tDIR_EDGE_SIZE : ", self.DIR_EDGE_SIZE)
        print("\t\tUNDIR_EDGE_SIZE : ", self.UNDIR_EDGE_SIZE)
        print("\t\tDIR_EDGE_COLOR : ", self.DIR_EDGE_COLOR)
        print("\t\tUNDIR_EDGE_COLOR : ", self.UNDIR_EDGE_COLOR)
        print("\t\tTHRES_CRITERION : ", self.THRES_CRITERION)
        print("\t\tnode_table_filename : ", self.node_table_filename)
        print("\t\tedge_table_filename : ", self.edge_table_filename)
        print(
            "\t\tedge_simplified_table_filename : ", self.edge_simplified_table_filename
        )
        print("\t\tresume_table_filename : ", self.resume_table_filename)
        print("\t\ttf_target_filename : ", self.tf_target_filename)
        print("\t\ttf_all_list_filename : ", self.tf_all_list_filename)
        print("\t\tgene_all_list_filename : ", self.gene_all_list_filename)
        print("\t\teval_score_filename : ", self.eval_score_filename)
        print("\t\teval_filename : ", self.eval_filename)
        print("\t\teval_score_plot_filename : ", self.eval_score_plot_filename)
        print("\t\teval_plot_filename : ", self.eval_plot_filename)
        print("\n")

    def create_out_dir(self, dir):
        path = Path(self.save_dir_path + dir)
        path.mkdir(parents=True, exist_ok=True)

    def filter_table(
        self,
        filtered_table,
        pval=None,
        perc_zero_tot=None,
        thres_criterion=None,
        model_score=None,
    ):
        if pval == None:
            pval = self.PVAL
        if perc_zero_tot == None:
            perc_zero_tot = self.PERC_ZERO_TOT
        if thres_criterion == None:
            thres_criterion = self.THRES_CRITERION
        if model_score == None:
            model_score = self.MODEL_SCORE
        filtered_table["p-val_1"] = -np.log10(filtered_table["p-val_1"])
        filtered_table["N2_p-val_++_+-"] = -np.log10(filtered_table["N2_p-val_++_+-"])
        filtered_table["N2_p-val_++_-+"] = -np.log10(filtered_table["N2_p-val_++_-+"])
        filtered_table["N2_p-val_++_--"] = -np.log10(filtered_table["N2_p-val_++_--"])
        filtered_table["N2_p-val_+-_-+"] = -np.log10(filtered_table["N2_p-val_+-_-+"])
        filtered_table["N2_p-val_+-_--"] = -np.log10(filtered_table["N2_p-val_+-_--"])
        filtered_table["N2_p-val_-+_--"] = -np.log10(filtered_table["N2_p-val_-+_--"])
        filtered_table["N3_p-val_++_+-"] = -np.log10(filtered_table["N3_p-val_++_+-"])
        filtered_table["N3_p-val_++_-+"] = -np.log10(filtered_table["N3_p-val_++_-+"])
        filtered_table["N3_p-val_++_--"] = -np.log10(filtered_table["N3_p-val_++_--"])
        filtered_table["N3_p-val_+-_-+"] = -np.log10(filtered_table["N3_p-val_+-_-+"])
        filtered_table["N3_p-val_+-_--"] = -np.log10(filtered_table["N3_p-val_+-_--"])
        filtered_table["N3_p-val_-+_--"] = -np.log10(filtered_table["N3_p-val_-+_--"])
        filtered_table = filtered_table[
            (filtered_table["gini_score_0"] < thres_criterion)
            & (filtered_table["model_score"] > model_score)
            & (filtered_table["perc_zero_tot"] < perc_zero_tot)
            & (
                (filtered_table["p-val_1"] > pval)
                | (filtered_table["N2_p-val_++_+-"] > pval)
                | (filtered_table["N2_p-val_++_-+"] > pval)
                | (filtered_table["N2_p-val_++_--"] > pval)
                | (filtered_table["N2_p-val_+-_-+"] > pval)
                | (filtered_table["N2_p-val_+-_--"] > pval)
                | (filtered_table["N2_p-val_-+_--"] > pval)
                | (filtered_table["N3_p-val_++_+-"] > pval)
                | (filtered_table["N3_p-val_++_-+"] > pval)
                | (filtered_table["N3_p-val_++_--"] > pval)
                | (filtered_table["N3_p-val_+-_-+"] > pval)
                | (filtered_table["N3_p-val_+-_--"] > pval)
                | (filtered_table["N3_p-val_-+_--"] > pval)
            )
        ]
        return filtered_table

    def init_candidate_info_file(self, save_path=None):
        if save_path == None:
            save_path = self.save_dir_path + "network/evaluateNet/"
        with open(save_path + self.eval_filename, "w") as file:
            file.write(
                "pval,perc_zero_tot,model_score,thres_criterion,nb_gene,nb_nodes,nb_edges,datapath\n"
            )

    def analyse_eval_PCA(self):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + "network/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )
        evaluateNetScore.dropna(axis=0, inplace=True)
        evaluateNetScore.reset_index(inplace=True)
        # normalize
        evaluateNetScore["pval"] = (
            evaluateNetScore["pval"] - np.mean(evaluateNetScore["pval"])
        ) / np.std(evaluateNetScore["pval"])
        evaluateNetScore["perc_zero_tot"] = (
            evaluateNetScore["perc_zero_tot"]
            - np.mean(evaluateNetScore["perc_zero_tot"])
        ) / np.std(evaluateNetScore["perc_zero_tot"])

        evaluateNetScore["model_score"] = (
            evaluateNetScore["model_score"] - np.mean(evaluateNetScore["model_score"])
        ) / np.std(evaluateNetScore["model_score"])

        evaluateNetScore["thres_criterion"] = (
            evaluateNetScore["thres_criterion"]
            - np.mean(evaluateNetScore["thres_criterion"])
        ) / np.std(evaluateNetScore["thres_criterion"])

        evaluateNetScore_X = np.array(
            [
                evaluateNetScore["pval"],
                evaluateNetScore["perc_zero_tot"],
                evaluateNetScore["model_score"],
                evaluateNetScore["thres_criterion"],
                evaluateNetScore["tpr"],
            ]
        )
        evaluateNetScore_Y = np.array(
            [
                evaluateNetScore["tpr"],
            ]
        )
        pca = PCA(n_components=None)
        principal_component = pca.fit_transform(evaluateNetScore_X.T)
        principal_evalscore = pd.DataFrame(
            data=principal_component,
            columns=["principal component 1", "principal component 2"],
        )
        print(principal_evalscore)
        print(
            "Explained variation per principal component: {}".format(
                pca.explained_variance_ratio_
            )
        )
        plt.figure(figsize=(10, 10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        plt.xlabel("Principal Component - 1", fontsize=20)
        plt.ylabel("Principal Component - 2", fontsize=20)
        plt.title("Principal Component Analysis of evaluation metrics", fontsize=20)
        targets = ["perc_zero_tot", "model_score", "thres_criterion"]
        colors = ["r", "g"]
        targets = ["Good Network", "Bad Network"]
        evaluateNetScore["tpr_eval"] = "Bad Network"
        evaluateNetScore.loc[
            evaluateNetScore["tpr"] >= 0.5, "tpr_eval"
        ] = "Good Network"
        colors = ["r", "g"]
        print(evaluateNetScore)
        for target, color in zip(targets, colors):
            indicesToKeep = evaluateNetScore["tpr_eval"] == target
            print(principal_evalscore)
            plt.scatter(
                principal_evalscore.loc[indicesToKeep, "principal component 1"],
                principal_evalscore.loc[indicesToKeep, "principal component 3"],
                c=color,
                s=50,
            )
            plt.legend(targets)
        plt.show()

    def plot_evaluation_curves(self):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + "network/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )
        dict_max = {
            "pval": max(evaluateNetScore["pval"]),
            "perc_zero_tot": max(evaluateNetScore["perc_zero_tot"]),
            "model_score": max(evaluateNetScore["model_score"]),
            "thres_criterion": max(evaluateNetScore["thres_criterion"]),
            "tp": max(evaluateNetScore["tp"]),
        }
        evaluateNetScore.dropna(axis=0, inplace=True)
        evaluateNetScore["pval"] = evaluateNetScore["pval"] / max(
            evaluateNetScore["pval"]
        )
        evaluateNetScore["perc_zero_tot"] = evaluateNetScore["perc_zero_tot"] / max(
            evaluateNetScore["perc_zero_tot"]
        )
        evaluateNetScore["model_score"] = evaluateNetScore["model_score"] / max(
            evaluateNetScore["model_score"]
        )
        evaluateNetScore["thres_criterion"] = evaluateNetScore["thres_criterion"] / max(
            evaluateNetScore["thres_criterion"]
        )
        evaluateNetScore["tp"] = evaluateNetScore["tp"] / max(evaluateNetScore["tp"])
        evaluateNetScore.sort_values(by="fscore", inplace=True)
        fig = plt.figure(figsize=(12, 10))
        plt.plot(
            range(0, len(evaluateNetScore["fscore"])),
            evaluateNetScore["fscore"],
            label="fscore",
        )
        plt.scatter(
            range(0, len(evaluateNetScore["tp"])),
            evaluateNetScore["pval"],
            label="pval_" + str(dict_max["pval"]),
        )
        plt.scatter(
            range(0, len(evaluateNetScore["tp"])),
            evaluateNetScore["model_score"],
            label="model_score_" + str(dict_max["model_score"]),
        )
        plt.scatter(
            range(0, len(evaluateNetScore["tp"])),
            evaluateNetScore["thres_criterion"],
            label="thres_criterion_" + str(dict_max["thres_criterion"]),
        )
        plt.scatter(
            range(0, len(evaluateNetScore["tp"])),
            evaluateNetScore["perc_zero_tot"],
            label="perc_zero_tot_" + str(dict_max["perc_zero_tot"]),
        )
        plt.legend()
        plt.savefig(
            self.save_dir_path + "network/evaluateNet/" + self.eval_plot_filename
        )

    def plot_precision_curves(self):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + "network/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )
        fig = plt.figure(figsize=(12, 10))
        plt.scatter(evaluateNetScore["precision"], evaluateNetScore["nb_candidate"])
        plt.savefig(self.save_dir_path + "network/evaluateNet/" + "precision_plot.png")

    def plot_fscore_curves(self):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + "network/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )
        fig = plt.figure(figsize=(12, 10))
        plt.scatter(evaluateNetScore["fscore"], evaluateNetScore["nb_candidate"])
        plt.savefig(self.save_dir_path + "network/evaluateNet/" + "fscore_plot.png")

    def plot_recall_curves(
        self, log10=False, arrow=False, metric="recall", net_dir="network"
    ):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + net_dir + "/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )

        best_candidates_1 = evaluateNetScore[
            (evaluateNetScore[metric] >= 0.072)
            & (np.log10(evaluateNetScore["nb_candidate"]) >= 2.0)
        ]
        best_candidates_2 = evaluateNetScore[
            (evaluateNetScore[metric] >= 0.30)
            & (np.log10(evaluateNetScore["nb_candidate"]) >= 0.8)
        ]
        print(best_candidates_1)
        print(best_candidates_2)
        plt.figure(figsize=(12, 8))
        if arrow:
            if log10 == True:

                ax = plt.axes()
                ax.scatter(
                    evaluateNetScore[metric],
                    np.log10(evaluateNetScore["nb_candidate"]),
                    c="g",
                    marker="o",
                    s=2,
                )
                ax.set_title("Determination of GRN size effect on recall")
                ax.set_ylabel("Number of GRN node")
                ax.set_xlabel(metric.capitalize())

                plt.scatter(
                    best_candidates_1[metric],
                    np.log10(best_candidates_1["nb_candidate"]),
                    color="red",
                )
                plt.scatter(
                    best_candidates_2[metric],
                    np.log10(best_candidates_2["nb_candidate"]),
                    color="red",
                )
                displacement_x = 0.00
                displacement_y = 0.00
                for index, points in best_candidates_1.iterrows():
                    label = f"({points['pval']},{points['model_score']},{points['perc_zero_tot']},{points['thres_criterion']})"

                    plt.annotate(
                        label,
                        (points[metric], np.log10(points["nb_candidate"])),
                        xycoords="data",
                        xytext=(
                            0.5 + displacement_y,
                            np.log(2) + displacement_x,
                        ),
                        textcoords="axes fraction",
                        va="top",
                        ha="left",
                        arrowprops=dict(facecolor="black", shrink=0.05),
                    )
                    # plt.annotate(
                    #     label,  # this is the text
                    #     (
                    #         points["recall"] + displacement,
                    #         np.log10(points["nb_candidate"]) + displacement,
                    #     ),
                    #     textcoords="offset points",
                    #     xytext=(0, 10),
                    #     ha="center",
                    #     rotation=60,
                    # )  # horizontal alignment can be left, right or center
                    displacement_x += 0.015
                    displacement_y += 0.01
                for _, points in best_candidates_2.iterrows():
                    label = f"({points['pval']},{points['model_score']},{points['perc_zero_tot']},{points['thres_criterion']})"

                    plt.annotate(
                        label,  # this is the text
                        (
                            points[metric],
                            np.log10(points["nb_candidate"]),
                        ),  # these are the coordinates to position the label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha="center",
                        rotation=60,
                    )  # horizontal alignment can be left, right or center
                plt.savefig(
                    self.save_dir_path
                    + net_dir
                    + "/evaluateNet/"
                    + metric
                    + "log10_plot.png"
                )
            else:
                # ax = plt.axes()
                # print(evaluateNetScore["recall"])
                # ax.scatter(
                #     evaluateNetScore["recall"],
                #     evaluateNetScore["nb_candidate"],
                #     c="g",
                #     marker="o",
                #     s=2,
                # )
                # ax.set_title("Determination of GRN size effect on recall")
                # ax.set_ylabel("Number of GRN node")
                # ax.set_xlabel("recall")
                ax = plt.axes()
                ax.scatter(
                    evaluateNetScore[metric],
                    evaluateNetScore["nb_candidate"],
                    c="g",
                    marker="o",
                    s=2,
                )
                ax.set_title("Determination of GRN size effect on recall")
                ax.set_ylabel("Number of GRN node")
                ax.set_xlabel(metric.capitalize())

                plt.scatter(
                    best_candidates_1[metric],
                    best_candidates_1["nb_candidate"],
                    color="red",
                )
                plt.scatter(
                    best_candidates_2[metric],
                    best_candidates_2["nb_candidate"],
                    color="red",
                )
                displacement_x = 0.00
                displacement_y = 0.00
                for index, points in best_candidates_1.iterrows():
                    label = f"({points['pval']},{points['model_score']},{points['perc_zero_tot']},{points['thres_criterion']})"

                    plt.annotate(
                        label,
                        (points[metric], points["nb_candidate"]),
                        xycoords="data",
                        xytext=(
                            0.5 + displacement_y,
                            np.log(1.2) + displacement_x,
                        ),
                        textcoords="axes fraction",
                        va="top",
                        ha="left",
                        arrowprops=dict(facecolor="black", shrink=0.05),
                    )
                    # plt.annotate(
                    #     label,  # this is the text
                    #     (
                    #         points["recall"] + displacement,
                    #         np.log10(points["nb_candidate"]) + displacement,
                    #     ),
                    #     textcoords="offset points",
                    #     xytext=(0, 10),
                    #     ha="center",
                    #     rotation=60,
                    # )  # horizontal alignment can be left, right or center
                    displacement_x += 0.015
                    displacement_y += 0.01
                for _, points in best_candidates_2.iterrows():
                    label = f"({points['pval']},{points['model_score']},{points['perc_zero_tot']},{points['thres_criterion']})"

                    plt.annotate(
                        label,  # this is the text
                        (
                            points[metric],
                            np.log10(points["nb_candidate"]),
                        ),  # these are the coordinates to position the label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha="center",
                        rotation=60,
                    )  # horizontal alignment can be left, right or center
                plt.savefig(
                    self.save_dir_path
                    + net_dir
                    + "/evaluateNet/"
                    + metric
                    + "_plot.png"
                )
        else:
            ax = plt.axes()
            ax.scatter(
                evaluateNetScore[metric],
                evaluateNetScore["nb_candidate"],
                c="g",
                marker="o",
                s=2,
            )
            ax.set_title("Determination of GRN size effect on recall")
            ax.set_ylabel("Number of GRN node")
            ax.set_xlabel(metric.capitalize())

            plt.scatter(
                best_candidates_1[metric],
                best_candidates_1["nb_candidate"],
                color="red",
            )
            plt.scatter(
                best_candidates_2[metric],
                best_candidates_2["nb_candidate"],
                color="red",
            )
            plt.tight_layout()
            plt.savefig(
                self.save_dir_path
                + net_dir
                + "/evaluateNet/"
                + metric
                + "_plot_no_arrow.png"
            )

    def plot_pval_curves(self):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + "network/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )
        fig = plt.figure(figsize=(12, 10))
        plt.scatter(evaluateNetScore["pval"], evaluateNetScore["nb_candidate"])
        plt.savefig(self.save_dir_path + "network/evaluateNet/" + "pval_plot.png")

    def plot_pczero_curves(self):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + "network/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )
        fig = plt.figure(figsize=(12, 10))
        plt.scatter(evaluateNetScore["perc_zero_tot"], evaluateNetScore["nb_candidate"])
        plt.scatter(
            evaluateNetScore["pval"] / np.max(evaluateNetScore["pval"]),
            evaluateNetScore["nb_candidate"],
        )
        plt.savefig(
            self.save_dir_path + "network/evaluateNet/" + "perc_zero_tot_plot.png"
        )

    def detect_anomaly_and_plot(self):
        evaluateNetScore = pd.read_table(
            self.save_dir_path + "network/evaluateNet/" + self.eval_score_filename,
            sep=",",
            header=0,
        )
        evaluateNetScore = evaluateNetScore.fillna(0)
        # evaluateNetScore.dropna(subset=["recall", "nb_candidate"], inplace=True)
        df = np.array(
            [
                evaluateNetScore["recall"].to_numpy(),
                evaluateNetScore["nb_candidate"].to_numpy(),
                evaluateNetScore["pval"].to_numpy(),
                evaluateNetScore["perc_zero_tot"].to_numpy(),
                evaluateNetScore["thres_criterion"].to_numpy(),
                evaluateNetScore["fscore"].to_numpy(),
            ]
        ).T

        print(df.shape)
        model = UMAP(n_components=2, min_dist=0.3, n_neighbors=30, metric="correlation")
        embedded_data = model.fit_transform(df)
        color1 = "#D4CC47"
        color2 = "#7C4D8B"
        num_points = 5000
        plt.scatter(*embedded_data.T)
        plt.scatter(
            *embedded_data[np.where(evaluateNetScore["recall"].to_numpy() > 0.3)].T,
            color="red",
        )
        plt.show()
        print(embedded_data.shape)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=500).fit(embedded_data)
        # sns.distplot(
        #     clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True
        # )
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.95)
        outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
        print(outliers)
        print(len([*df.T]))
        # plt.scatter(*df.T, s=50, linewidth=0, color="gray", alpha=0.25)
        # plt.scatter(*df[outliers].T, s=50, linewidth=0, color="red", alpha=0.5)
        sns.distplot(
            clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True
        )

        plt.show()

    def save_candidates_info(
        self,
        save_path,
        pval=None,
        perc_zero_tot=None,
        model_score=None,
        thres_criterion=None,
        nb_candidate="0",
        nb_nodes="0",
        nb_edges="0",
        datapath=None,
    ):
        with open(save_path + self.eval_filename, "a") as file:
            file.write(
                pval
                + ","
                + perc_zero_tot
                + ","
                + model_score
                + ","
                + thres_criterion
                + ","
                + nb_candidate
                + ","
                + nb_nodes
                + ","
                + nb_edges
                + ","
                + datapath
                + "\n"
            )

    def create_GRN(
        self,
        filtered_table=None,
        table_default_name="compiled_table.csv",
        pval=None,
        perc_zero_tot=None,
        model_score=None,
        thres_criterion=None,
        TF_indicator="",  # or "TF_"
    ):
        if filtered_table == None:
            filtered_table = pd.read_table(
                self.save_dir_path + table_default_name, sep=","
            )
            filtered_table = self.filter_table(
                filtered_table, pval, perc_zero_tot, thres_criterion, model_score
            )
            print("There is ", len(filtered_table.index), " candidate.")
        self._perc_grn_interface = 0.0
        for _, target in filtered_table.iterrows():
            self._perc_grn_interface += 100 / len(filtered_table.index)
            sys.stdout.write(
                "\r GENERATING GRN : node {0} ({1}%)".format(
                    target["AGI"], round(self._perc_grn_interface, 3)
                )
            )
            sys.stdout.flush()
            self.create_node([target["AGI"]], "GENE")

            # SIMPLE
            self.create_node([TF_indicator + target["TF1"]], "TF")
            self.create_edge(
                target["AGI"],
                TF_indicator + target["TF1"],
                score=target["gini_score_0"],
            )

            # INTER 1
            if target["gini_score_1"] != None:
                if target["gini_score_1"] < self.THRES_CRITERION:
                    TF1_TF2_inter_node = self.create_node(
                        [TF_indicator + target["TF1"], TF_indicator + target["TF2"]],
                        "TF_INTER",
                    )
                    self.create_edge(
                        target["AGI"],
                        TF1_TF2_inter_node,
                        score=target["gini_score_1"],
                    )
            # INTER 2
            if target["gini_score_2"] != None:
                if target["gini_score_2"] < self.THRES_CRITERION:
                    TF1_TF3_inter_node = self.create_node(
                        [TF_indicator + target["TF1"], TF_indicator + target["TF3"]],
                        "TF_INTER",
                    )
                    self.create_edge(
                        target["AGI"],
                        TF1_TF3_inter_node,
                        score=target["gini_score_2"],
                    )
        print(
            "\n Done generating GRN :\n\t- "
            + str(self.get_number_of_nodes())
            + " nodes found\n\t- "
            + str(self.get_number_of_edges())
            + " edges found\n"
        )
        return (
            len(filtered_table.index),
            self.get_number_of_nodes(),
            self.get_number_of_edges(),
        )

    def get_number_of_nodes(self):
        return self.G.number_of_nodes()

    def get_number_of_edges(self):
        return self.G.number_of_edges()

    def create_node(
        self, node_name, node_type, node_color=None, node_style=None, node_size=None
    ):
        if node_type not in ["TF", "TF_INTER", "GENE"]:
            if (node_color == None) or (node_size == None) or (node_style == None):
                raise NotImplementedError(
                    "Unknow node type and no default style are given"
                )
        if len(node_name) < 2:
            node_name = node_name[0]
            if node_name not in list(self.G.nodes):  # node don't exist
                if node_type == "TF_INTER":  # node is an inter
                    if node_name.split("-")[1] + "-" + node_name.split("-")[
                        0
                    ] not in list(
                        self.G.nodes
                    ):  # reverted node don't exist
                        self.G.add_node(node_name)
                        self.node_color.append(
                            self.TF_NODE_COLOR
                            if node_type == "TF"
                            else self.GENE_NODE_COLOR
                            if node_type == "GENE"
                            else self.TF_INTER_NODE_COLOR
                            if node_type == "TF_INTER"
                            else node_color
                        )
                        self.node_style.append(
                            self.TF_SHAPE
                            if node_type == "TF"
                            else self.GENE_SHAPE
                            if node_type == "GENE"
                            else self.TF_INTER_SHAPE
                            if node_type == "TF_INTER"
                            else node_style
                        )
                        self.node_size.append(
                            self.TF_NODE_SIZE
                            if node_type == "TF"
                            else self.GENE_NODE_SIZE
                            if node_type == "GENE"
                            else self.TF_INTER_NODE_SIZE
                            if node_type == "TF_INTER"
                            else node_size
                        )
                        return node_name
                    else:
                        return node_name.split("-")[1] + "-" + node_name.split("-")[0]
                else:
                    self.G.add_node(node_name)
                    self.node_color.append(
                        self.TF_NODE_COLOR
                        if node_type == "TF"
                        else self.GENE_NODE_COLOR
                        if node_type == "GENE"
                        else self.TF_INTER_NODE_COLOR
                        if node_type == "TF_INTER"
                        else node_color
                    )
                    self.node_style.append(
                        self.TF_SHAPE
                        if node_type == "TF"
                        else self.GENE_SHAPE
                        if node_type == "GENE"
                        else self.TF_INTER_SHAPE
                        if node_type == "TF_INTER"
                        else node_style
                    )
                    self.node_size.append(
                        self.TF_NODE_SIZE
                        if node_type == "TF"
                        else self.GENE_NODE_SIZE
                        if node_type == "GENE"
                        else self.TF_INTER_NODE_SIZE
                        if node_type == "TF_INTER"
                        else node_size
                    )
            else:
                return node_name
        else:
            for node in node_name:
                self.create_node([node], "TF")
            node_inter = self.create_node(["-".join(node_name)], "TF_INTER")
            for node in node_name:
                self.create_edge(node, node_inter, None)
            return node_inter

    def create_edge(
        self,
        source_name,
        target_name,
        score,
        edge_type="UNDIRECTED",
        edge_color=None,
        edge_size=None,
    ):
        if edge_type not in ["UNDIRECTED", "DIRECTED"]:
            if (edge_color == None) or (edge_size == None):
                raise NotImplementedError(
                    "Unknow edge type and no default style are given"
                )
        if ((source_name, target_name) not in self.G.edges) or (
            (target_name, source_name) not in self.G.edges
        ):
            self.G.add_edge(
                source_name,
                target_name,
                score=score,
                line_type=self.DEFAULT_LINE_TYPE
                if score != None
                else self.TF_INTER_LINE_TYPE,
                edge_color=self.DIR_EDGE_COLOR
                if edge_type == "DIRECTED"
                else self.UNDIR_EDGE_COLOR
                if edge_type == "UNDIRECTED"
                else edge_color,
            )
        else:
            pass

    def to_tables(self):
        df = nx.to_pandas_edgelist(self.G)
        df["interaction"] = "interacts"
        df.rename(columns={"score": "weight"}, inplace=True)
        full_node = pd.DataFrame(self.G.nodes, columns=["id"])
        full_node["node_size"] = self.node_size
        full_node["node_color"] = self.node_color
        full_node["node_style"] = self.node_style
        node_degree = pd.concat(
            [df["source"].value_counts(), df["target"].value_counts()],
            join="outer",
            axis=1,
        ).reset_index()
        node_degree.fillna(0, inplace=True)
        node_degree.index = node_degree["index"]
        node_degree["source"] = node_degree["source"] + node_degree["target"]
        node_degree.drop(["index", "target"], inplace=True, axis=1)
        return df, full_node, node_degree

    def send_to_cytoscape(self, title="GRN_default", path_save_session="default.cys"):
        df, full_node, _ = self.to_tables()
        dir(py4)
        py4.cytoscape_ping()
        py4.networks.create_network_from_data_frames(full_node, df, title=title)
        py4.set_node_color_mapping(
            table_column="node_color",
            table_column_values=["red", "blue", "green"],
            colors=["#F05B0B", "#899AEE", "#6BF367"],
            mapping_type="discrete",
            network=title,
        )
        py4.save_session(path_save_session)
        py4.delete_network(title)

    def save_graph_to_table(
        self, with_resume=True, save_path=None, inter_behavior="drop", TF_indicator=""
    ):
        if save_path == None:
            path_edge_table = self.path_edge_table
            path_edge_simplified_table = self.path_edge_simplified_table
            path_node_table = self.path_node_table
            path_resume_table = self.path_resume_table
        else:
            path_edge_table = save_path + "/" + self.edge_table_filename
            path_edge_simplified_table = (
                save_path + "/" + self.edge_simplified_table_filename
            )
            path_node_table = save_path + "/" + self.node_table_filename
            path_resume_table = save_path + "/" + self.resume_table_filename

        df, full_node, node_degree = self.to_tables()
        if TF_indicator != "":
            df["source_ori"] = df["source"]
            df["target_ori"] = df["target"]
            # Put TF in from column
            df.loc[
                df[
                    (df["target"].str.match(r"^" + TF_indicator))
                    & (df["source"].str.match(r"^AT"))
                ].index,
                "source",
            ] = df.loc[
                df[
                    (df["target"].str.match(r"^" + TF_indicator))
                    & (df["source"].str.match(r"^AT"))
                ].index
            ][
                "target"
            ]
            # and TG in to column
            df.loc[
                df[
                    (df["target_ori"].str.match(r"^" + TF_indicator))
                    & (df["source_ori"].str.match(r"^AT"))
                ].index,
                "target",
            ] = df.loc[
                df[
                    (df["target_ori"].str.match(r"^" + TF_indicator))
                    & (df["source_ori"].str.match(r"^AT"))
                ].index
            ][
                "source_ori"
            ]
            df.drop(["source_ori", "target_ori"], axis=1, inplace=True)

            df["source"] = df["source"].str.replace(TF_indicator, "")
        df.to_csv(path_edge_table, sep=",", header=True, index=False)
        full_node.to_csv(path_node_table, sep=",", header=True, index=False)
        df.drop(
            columns=["weight", "line_type", "edge_color", "interaction"], inplace=True
        )

        # Prepare data for from-to format (adapted for Oceane Cassan net-evaluation)
        df.columns = ["from", "to"]
        sub_df = df[df["to"].str.contains("-")]
        sub_df.append(df[df["from"].str.contains("-")])
        df.drop(df[df["to"].str.contains("-")].index, axis=0, inplace=True)
        df.drop(df[df["from"].str.contains("ATMG")].index, axis=0, inplace=True)
        df.drop(df[df["to"].str.contains("ATMG")].index, axis=0, inplace=True)
        df.drop(df[df["from"].str.contains("-")].index, axis=0, inplace=True)
        df["_to_comp"] = df["from"] + "-" + df["to"]
        if inter_behavior == "transform":
            for fr, to in zip(sub_df["from"], sub_df["to"]):
                if "-" in fr:
                    fr = fr.split("-")
                else:
                    fr = [fr]
                if "-" in to:
                    to = to.split("-")
                else:
                    to = [to]
                for elem1 in fr:
                    for elem2 in to:
                        if (elem1 + "-" + elem2) in df["_to_comp"].values:
                            continue
                        if (elem2 + "-" + elem1) in df["_to_comp"].values:
                            continue
                        if elem1 == elem2:
                            continue
                        new_row = {
                            "from": elem1,
                            "to": elem2,
                            "_to_comp": elem1 + "-" + elem2,
                        }
                        df = df.append(new_row, ignore_index=True)
        elif inter_behavior == "drop":
            pass
        else:
            raise NotImplementedError("Unknow behavior to adopt for interaction")
        df.drop(columns=["_to_comp"], inplace=True)

        df.replace(self.LIST_TF, inplace=True)

        df.to_csv(path_edge_simplified_table, sep=",", header=True, index=False)
        if with_resume:
            node_degree.to_csv(
                path_resume_table,
                index=True,
                header=["degree"],
                index_label="source",
            )

    def resume_founded_edges_by_val(
        self, save_path, which="max", grn_size_metric="nb_nodes"
    ):
        eval_table = pd.read_table(save_path + self.eval_filename, header=0, sep=",")
        if which == "max":
            tg_filename = eval_table.iloc[eval_table[grn_size_metric].idxmax()][
                "datapath"
            ]
            print(tg_filename)
            found_edges = pd.read_table(
                tg_filename + "/found_edges.csv", header=0, sep=","
            )
            print(found_edges)
            print(
                found_edges["type"].value_counts().index,
                found_edges["type"].value_counts().values,
            )
            fig, ax = plt.subplots()
            ax.bar(
                x=found_edges["type"].value_counts().index,
                height=found_edges["type"].value_counts().values,
            )
            # plot = found_edges["type"].value_counts().plot.barh()
            addlabels(
                found_edges["type"].value_counts().index,
                found_edges["type"].value_counts().values,
            )
            plt.xticks(rotation=0, horizontalalignment="center")
            plt.tight_layout()
            plt.savefig(save_path + "resume_max_net_founded_edge.png")
            print(save_path + "resume_max_net_founded_edge.png")
        else:
            raise NotImplementedError("Unknow behavior to adopt for which option")

    def save_target_genes(
        self, sources, drop_inter=False
    ):  # add_inter_to_direct=True):
        matching_target = []
        for source in sources:
            df, _, _ = self.to_tables()
            # if add_inter_to_direct:
            #     inter = df.loc[df[df["target"].str.contains("-") == True].index].copy(
            #         deep=True
            #     )
            #     for elem in inter["target"].to_list():
            #         TF1, TF2 = elem.split("-")
            #         i_src_TF1 = df[df["source"].str.contains(TF1) == True].index
            #         i_src_TF2 = df[df["source"].str.contains(TF2) == True].index
            #         i_tg_TF1 = df[df["target"].str.contains(TF1) == True].index
            #         i_tg_TF2 = df[df["target"].str.contains(TF2) == True].index
            #         i_common_tf_src = len(list(set(i_src_TF1).intersection(i_tg_TF2)))
            #         i_common_tf_tg = len(list(set(i_src_TF2).intersection(i_tg_TF1)))
            #         if (i_common_tf_src + i_common_tf_tg) == 0:
            #             print("did we go here ?")
            #             missing_inter = pd.DataFrame(
            #                 {
            #                     "source": [TF1],
            #                     "target": [TF2],
            #                     "weight": None,
            #                     "line_type": None,
            #                     "edge_color": None,
            #                     "interaction": None,
            #                 }
            #             )
            #             df.append(missing_inter)
            if drop_inter == True:
                df.drop(
                    df[df["target"].str.contains("-") == True].index,
                    axis=0,
                    inplace=True,
                )
                df.drop(
                    df[df["source"].str.contains("-") == True].index,
                    axis=0,
                    inplace=True,
                )
            l_target = df.loc[df["source"] == source, "target"]
            r_target = df.loc[df["target"] == source, "source"]
            targets = pd.concat([l_target, r_target], ignore_index=True)
            if len(targets.index) > 0:
                for key, val in self.LIST_TF.items():
                    targets = targets.str.replace(key, val)
                targets.to_csv(
                    self.path_src_target_dir + source + ".txt",
                    header=False,
                    index=False,
                )
                matching_target.append(source)
        with open(self.path_resume_target, "w") as file:
            file.write("source\n")
            for elem in matching_target:
                file.write(elem + "\n")

        return matching_target

    # TODO method to update graph based on graph evaluation
    # def update_cytoscape_graph_eval(self, save_path):
    #     list_gene_info = pd.read_table(
    #         self.gene_path + self.eval_score_plot_filename, header=0, sep=","
    #     )
    #     for _, points in list_gene_info.iterrows():
    #         if points["nb_candidate"] < 1:
    #             continue
    #         old_edge_table = points["datapath"] + "/" + self.edge_table_filename
    #         new_edge_table = (
    #             points["datapath"] + "/" + self.edge_table_post_eval_filename
    #         )

    def send_for_evaluation(self, save_path):
        out_save_path = save_path + self.eval_score_filename
        save_path = save_path + self.eval_filename
        import subprocess

        # res = subprocess.call(
        #     "Rscript NetworkEvaluation/runEvaluateNet.R "
        #     + save_path
        #     + " "
        #     + out_save_path
        #     + " "
        #     + self.edge_simplified_table_filename
        #     + " "
        #     + self.save_dir_path
        #     + self.tf_all_list_filename
        #     + " "
        #     + self.save_dir_path
        #     + self.gene_all_list_filename,
        #     shell=True,
        # )
        res = subprocess.call(
            "Rscript NetworkEvaluation/runEvaluateNet.R "
            + save_path
            + " "
            + out_save_path
            + " "
            + self.edge_simplified_table_filename
            + " "
            + self.found_edges_info_out_path,
            shell=True,
        )
        print(res)

    def run_find_enrichment(self, sources, pval=0.05):
        # Copyright (c) 2010-2018, Haibao Tang
        # All rights reserved.

        # Redistribution and use in source and binary forms, with or without modification,
        # are permitted provided that the following conditions are met:

        # Redistributions of source code must retain the above copyright notice, this list
        # of conditions and the following disclaimer.

        # Redistributions in binary form must reproduce the above copyright notice, this
        # list of conditions and the following disclaimer in the documentation and/or
        # other materials provided with the distribution.

        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
        # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
        # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
        # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
        # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
        # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
        # ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        print(" GENERATING GO Term association study, this can take few minutes...")
        with NoStdStreams():
            for source in sources:
                testargs = [
                    "find_enrichment.py",
                    "--pval=" + str(pval),
                    "--indent",
                    self.path_src_target_dir + source + ".txt",
                    "/media/carluerj/Data/These/DATA/gene_regulator_network/population_list.txt",
                    "/media/carluerj/Data/These/DATA/gene_regulator_network/population_GO_term_compressed.txt",
                    "--outfile="
                    + self.save_dir_path
                    + "network/association_study/"
                    + source
                    + ".txt",
                ]
                with patch.object(sys, "argv", testargs):
                    from goatools.cli.find_enrichment import GoeaCliArgs
                    from goatools.cli.find_enrichment import GoeaCliFnc

                    obj = GoeaCliFnc(GoeaCliArgs().args)
                    results_specified = obj.get_results()
                    obj.prt_results(results_specified)


def hex_to_RGB(hex_str):
    """#FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return [
        "#" + "".join([format(int(round(val * 255)), "02x") for val in item])
        for item in rgb_colors
    ]


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] // 2, y[i], ha="center")
