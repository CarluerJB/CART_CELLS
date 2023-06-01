from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys

data = sys.argv[1]
out_data = sys.argv[2]
tf_list_path = sys.argv[3]
analysis_type = sys.argv[4]

# cart = CART_TREE(data, tf_list_path, out_data, analysis_type)
# cart.load_parameter_file()
# cart.show_parameter()
# cart.create_out_dir()
# cart.load_GE_matrix()
# cart.load_tf_list()
# for target in cart.Y.columns:
#     cart.generate_CART_tree(target)
#     cart.eval_model(target)
#     cart.save_CART_tree(target)
#     cart.compile_cart_results(target)
#     cart.save_cartlu_plot(target)
# cart.save_compiled_results()
# cart.filter_cart_results()
# cart.save_filtered_results()

# G = GRN(out_data)
# G.create_GRN()
# G.save_graph_to_table()
# G.send_to_cytoscape()
# matching_target_genes = G.save_target_genes(G.LIST_TF.keys(), drop_inter=True)

# G.run_find_enrichment(matching_target_genes)

# clust = DENDROGRAM_TW(out_data)
# clust.load_data()
# clust.plot_clustering_TW()

pval_list = [50, 40, 30, 20, 15, 10, 8, 5, 3, 2]
model_score_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
perc_zero_tot_list = [0.5, 0.4, 0.3, 0.2, 0.1]
thres_criterion_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

G = GRN(out_data)
G.init_candidate_info_file()
for pval in pval_list:
    for model_score in model_score_list:
        for perc_zero_tot in perc_zero_tot_list:
            for thres_criterion in thres_criterion_list:
                nb_candidate = G.create_GRN(
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
                    G.save_graph_to_table(save_path=save_path)
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
                        datapath="None",
                    )
G.send_for_evaluation(G.save_dir_path + "network/evaluateNet/")
