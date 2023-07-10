from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys

data = sys.argv[1]
out_data = sys.argv[2]
tf_list_path = sys.argv[3]

clust = DENDROGRAM_TW(out_data, ge_matrix_path=data, TF_list_path=tf_list_path)
clust.load_GE_matrix()
print("A")
clust.load_tf_list()
print("B")
clust.extract_TF_from_GE_matrix()
print("C")
clust.plot_clustering_TW()
