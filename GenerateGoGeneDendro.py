# ===============================
# AUTHOR     : CARLUER Jean-Baptiste
# CREATE DATE     : 2022-2023
# PURPOSE     : Thesis in BioInformatics
# SPECIAL NOTES: This programm is meant to generate GO dendrogram based
#   on GRN genes using default parameters
#   (see PARAMETERS/PARAM_CLUST_DEFAULT.txt).
#   The program GenerateCartTree.py need to be run before running this one.
# ===============================
# Change History:
#
# # =================================

from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys
import argparse

# PARAMETERS
parser = argparse.ArgumentParser(
    prog="Generate Gene Ontology Dendrogram",
    description="This programm is meant to generate GO dendrogram based on GRN genes using default parameters (see PARAMETERS/PARAM_CLUST_DEFAULT.txt). The program GenerateCartTree.py need to be run before running this one.",
    epilog="Realized by Jean-Baptiste Carluer during his PhD in bioinformatics at the University of Montpellier under the supervision of Gabriel Krouk and André Mas",
)
parser.add_argument(
    "-o",
    "--out_path",
    help="Path to your out directory",
    required=True,
)
args = parser.parse_args()
# out_data = sys.argv[1]
out_data = args.out_path

clust = DENDROGRAM_TW(out_data)
clust.load_GO_data()
clust.plot_clustering_GOTW()
