from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys

out_data = sys.argv[1]
if len(sys.argv) > 2:
    TF_AGI_CORR = sys.argv[2]
else:
    TF_AGI_CORR = None

G = GRN(out_data, TF_AGI_CORR)
G.create_GRN()
G.save_graph_to_table()
G.send_to_cytoscape(path_save_session=G.save_dir_path + "network/global_GRN.cys")
matching_target_genes = G.save_target_genes(G.LIST_TF.keys(), drop_inter=True)

print(matching_target_genes)
G.run_find_enrichment(matching_target_genes)
