from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys

out_data = sys.argv[2]

G = GRN(out_data)
G.create_GRN()
G.save_graph_to_table()
G.send_to_cytoscape()
matching_target_genes = G.save_target_genes(G.LIST_TF.keys(), drop_inter=True)

G.run_find_enrichment(matching_target_genes)
