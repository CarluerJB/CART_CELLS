from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys

out_data = sys.argv[2]

clust = DENDROGRAM_TW(out_data)
clust.load_GO_data()
clust.plot_clustering_GOTW()
