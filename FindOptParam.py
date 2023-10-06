# ===============================
# AUTHOR     : CARLUER Jean-Baptiste
# CREATE DATE     : 2022-2023
# PURPOSE     : Thesis in BioInformatics
# SPECIAL NOTES: This programm is meant to generate GRN for each
#   default filter, to evaluate them and then for the bigger GRN to
#   generate a summary of validated interactions sources. The program
#   GenerateCartTree.py need to be run before running this one.
# ===============================
# Change History:
#
# # =================================

from lib.grn_class import GRN
import sys
import argparse

# PARAMETERS
parser = argparse.ArgumentParser(
    prog="Find Optimum Parameters",
    description="This programm is meant to generate GRN for each default filter, to evaluate them and then for the bigger GRN to generate a summary of validated interactions sources. The program GenerateCartTree.py need to be run before running this one.",
    epilog="Realized by Jean-Baptiste Carluer during his PhD in bioinformatics at the University of Montpellier under the supervision of Gabriel Krouk and André Mas",
)
parser.add_argument(
    "-o",
    "--out_path",
    help="Path to your out directory",
    required=True,
)
args = parser.parse_args()
out_data = args.out_path

# These are filter to test, each combination is tested
pval_list = [50, 40, 30, 20, 15, 10, 8, 5, 3, 2]
model_score_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
perc_zero_tot_list = [0.5, 0.4, 0.3, 0.2, 0.1]
thres_criterion_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

G = GRN(out_data)

# Allow to create the file with header
# Comment this line to keep writing in a file
# without deleting previous work
G.init_candidate_info_file()

for pval in pval_list:
    for model_score in model_score_list:
        for perc_zero_tot in perc_zero_tot_list:
            for thres_criterion in thres_criterion_list:
                nb_candidate, nb_nodes, nb_edges = G.create_GRN(
                    pval=pval,
                    perc_zero_tot=perc_zero_tot,
                    model_score=model_score,
                    thres_criterion=thres_criterion,
                )
                if nb_candidate > 0:
                    save_path = (
                        "network/evaluateNet/"
                        + str(pval)
                        + "_"
                        + str(model_score)
                        + "_"
                        + str(perc_zero_tot)
                        + "_"
                        + str(thres_criterion)
                    )
                    G.create_out_dir(save_path)
                    save_path = G.save_dir_path + save_path
                    G.save_graph_to_table(save_path=save_path, TF_indicator="TF_")
                    G.send_to_cytoscape(
                        title=str(pval)
                        + "_"
                        + str(model_score)
                        + "_"
                        + str(perc_zero_tot)
                        + "_"
                        + str(thres_criterion),
                        path_save_session=save_path + "/network.cys",
                    )
                    G.save_candidates_info(
                        G.save_dir_path + "network/evaluateNet/",
                        pval=str(pval),
                        perc_zero_tot=str(perc_zero_tot),
                        model_score=str(model_score),
                        thres_criterion=str(thres_criterion),
                        nb_candidate=str(nb_candidate),
                        nb_nodes=str(nb_nodes),
                        nb_edges=str(nb_edges),
                        datapath=save_path,
                    )
                else:
                    G.save_candidates_info(
                        G.save_dir_path + "network/evaluateNet/",
                        pval=str(pval),
                        perc_zero_tot=str(perc_zero_tot),
                        model_score=str(model_score),
                        thres_criterion=str(thres_criterion),
                        nb_candidate="0",
                        nb_nodes="0",
                        nb_edges="0",
                        datapath="None",
                    )
                G.delete_GRN()
# Evaluation of the GRN based on previously founded interactions
G.send_for_evaluation(G.save_dir_path + "network/evaluateNet/")
# Show the ratio of each categories (CHIPSeq, DAPSeq, TF Not Found,...)
G.resume_founded_edges_by_val(G.save_dir_path + "network/evaluateNet/")
