from lib.grn_class import GRN
import sys

data = sys.argv[1]
out_data = sys.argv[2]
tf_list_path = sys.argv[3]
analysis_type = sys.argv[4]

pval_list = [50, 40, 30, 20, 15, 10, 8, 5, 3, 2]
model_score_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
perc_zero_tot_list = [0.5, 0.4, 0.3, 0.2, 0.1]
thres_criterion_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pval = 50
perc_zero_tot = 0.5
model_score = 0
thres_criterion = 0.7
G = GRN(out_data)
G.init_candidate_info_file(G.save_dir_path + "network/evaluateNet/")
# for pval in pval_list:
#     for model_score in model_score_list:
#         for perc_zero_tot in perc_zero_tot_list:
#             for thres_criterion in thres_criterion_list:
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