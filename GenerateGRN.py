from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys
import argparse

# PARAMETERS
parser = argparse.ArgumentParser(
    prog="Generate GRN",
    description="This programm is meant to generate GRN using default parameters (see PARAMETERS/PARAM_GRN_DEFAULT.txt). The program GenerateCartTree.py need to be run before running this one.",
    epilog="Realized by Jean-Baptiste Carluer during his PhD in bioinformatics at the University of Montpellier under the supervision of Gabriel Krouk and André Mas",
)
parser.add_argument(
    "-o",
    "--out_path",
    help="Path to your out directory",
    required=True,
)

parser.add_argument(
    "-tfcp",
    "--tf_agi_corr",
    help="Path to a table with correspondance between gene name and AGI name",
    required=False,
)
args = parser.parse_args()
out_data = args.out_path
TF_AGI_CORR = args.tf_agi_corr
# out_data = sys.argv[1]
# if len(sys.argv) > 2:
#     TF_AGI_CORR = sys.argv[2]
# else:
#     TF_AGI_CORR = None

G = GRN(out_data, TF_AGI_CORR)
G.create_GRN()
G.save_graph_to_table()
G.send_to_cytoscape(path_save_session=G.save_dir_path + "network/global_GRN.cys")
matching_target_genes = G.save_target_genes(G.LIST_TF.keys(), drop_inter=True)

print(matching_target_genes)
G.run_find_enrichment(matching_target_genes)
