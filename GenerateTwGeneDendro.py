from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys
import argparse

# PARAMETERS
parser = argparse.ArgumentParser(
    prog="Generate Two Way Gene Dendrogram",
    description="This programm is meant to generate a clustering + dendrogram on gene and cells",
    epilog="Realized by Jean-Baptiste Carluer during his PhD in bioinformatics at the University of Montpellier under the supervision of Gabriel Krouk and André Mas",
)
parser.add_argument(
    "-d",
    "--data_path",
    help="Path to your data, data should be either a .txt matrix tab separated or a .h5 file.",
    required=True,
)
parser.add_argument(
    "-o",
    "--out_path",
    help="Path to your out directory, directory will be automatically created",
    required=True,
)
parser.add_argument(
    "-tf",
    "--tf_list_path",
    help="Path to your tf list, tf in the file will be search in the data file and used as tf. Should be one tf/row",
    required=True,
)

args = parser.parse_args()
data = args.data_path
out_data = args.out_path
tf_list_path = args.tf_list_path


# data = sys.argv[1]
# out_data = sys.argv[2]
# tf_list_path = sys.argv[3]

clust = DENDROGRAM_TW(out_data, ge_matrix_path=data, TF_list_path=tf_list_path)
clust.load_GE_matrix()
clust.load_tf_list()
clust.extract_TF_from_GE_matrix()
clust.plot_clustering_TW()
