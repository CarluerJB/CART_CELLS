from codecs import ignore_errors
from tkinter.tix import COLUMN
from importlib_metadata import itertools
import networkx as nx
import py4cytoscape as py4
import pandas as pd
import os
from unittest.mock import patch
import sys

RANDOM_STATE = 42


class GRN:
    def __init__(self, out_data):
        self.G = nx.Graph()
        self.node_size = []
        self.node_color = []
        self.node_style = []
        self.save_dir_path = out_data
        self.path_node_table = self.save_dir_path + "network/node_table.csv"
        self.path_edge_table = self.save_dir_path + "network/edge_table.csv"
        self.path_resume_table = self.save_dir_path + "network/resume_table.csv"
        self.path_resume_target = self.save_dir_path + "network/resume_TF_target.csv"
        self.path_src_target_dir = self.save_dir_path + "network/genes_study/"
        self.parameter_file_path = "PARAM_GRN_DEFAULT.txt"
        self.TF_SHAPE = "ELLIPSE"
        self.GENE_SHAPE = "ROUND_RECTANGLE"
        self.TF_INTER_SHAPE = "TRIANGLE"
        self.TF_INTER_LINE_TYPE = "DASHED"
        self.DEFAULT_LINE_TYPE = "FULL"
        self.TF_NODE_SIZE = 10
        self.GENE_NODE_SIZE = 10
        self.TF_INTER_NODE_SIZE = 10
        self.TF_NODE_COLOR = "red"
        self.GENE_NODE_COLOR = "blue"
        self.TF_INTER_NODE_COLOR = "green"
        self.DIR_EDGE_SIZE = 10
        self.UNDIR_EDGE_SIZE = 10
        self.DIR_EDGE_COLOR = "red"
        self.UNDIR_EDGE_COLOR = "blue"
        self.THRES_CRITERION = 1.0
        self.load_parameter_file()
        self.LIST_TF = {
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
        self._perc_grn_interface = 0.0
        self.create_out_dir()

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

    def create_out_dir(self):
        os.makedirs(self.save_dir_path + "network", exist_ok=True)
        os.makedirs(self.save_dir_path + "network/genes_study", exist_ok=True)
        os.makedirs(self.save_dir_path + "network/association_study", exist_ok=True)

    def create_GRN(self, filtered_table):
        for _, target in filtered_table.iterrows():
            self._perc_grn_interface += 100 / len(filtered_table.index)
            sys.stdout.write(
                "\r GENERATING GRN : node {0} ({1}%)".format(
                    target["AGI"], self._perc_grn_interface
                )
            )
            sys.stdout.flush()
            self.create_node([target["AGI"]], "GENE")

            # SIMPLE
            self.create_node([target["TF1"]], "TF")
            self.create_edge(target["AGI"], target["TF1"], score=target["gini_score_0"])

            # INTER 1
            if target["gini_score_1"] != None:
                if target["gini_score_1"] > self.THRES_CRITERION:
                    TF1_TF2_inter_node = self.create_node(
                        [target["TF1"], target["TF2"]], "TF_INTER"
                    )
                    self.create_edge(
                        target["AGI"], TF1_TF2_inter_node, score=target["gini_score_1"]
                    )
            # INTER 2
            if target["gini_score_2"] != None:
                if target["gini_score_2"] > self.THRES_CRITERION:
                    TF1_TF3_inter_node = self.create_node(
                        [target["TF1"], target["TF3"]], "TF_INTER"
                    )
                    self.create_edge(
                        target["AGI"], TF1_TF3_inter_node, score=target["gini_score_2"]
                    )
        print("\n Done generating GRN\n")

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
            if node_name not in list(self.G.nodes):
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
            for node in node_name:
                self.create_node([node], "TF")
            node_inter = self.create_node(["-".join(node_name)], "TF_INTER")
            for node in node_name:
                self.create_edge(node, node_inter, None)
            return "-".join(node_name)

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

    def send_to_cytoscape(self):
        df, full_node, _ = self.to_tables()
        dir(py4)
        py4.cytoscape_ping()
        py4.networks.create_network_from_data_frames(full_node, df, title="GRN_default")

    def save_graph_to_table(self, with_resume=True):
        df, full_node, node_degree = self.to_tables()
        df.to_csv(self.path_edge_table, sep=",", header=True, index=False)
        full_node.to_csv(self.path_node_table, sep=",", header=True, index=False)
        if with_resume:
            node_degree.to_csv(
                self.path_resume_table,
                index=True,
                header=["degree"],
                index_label="source",
            )

    def save_target_genes(self, sources, drop_inter=False):
        matching_target = []
        for source in sources:
            df, _, _ = self.to_tables()
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
