source("NetworkEvaluation/evaluateNetwork.R")

args = commandArgs(trailingOnly=TRUE)
models_info_path = args[1]
models_info_out_path = args[2]
model_edge_filename = args[3]
found_edges_info_out_path = args[4]
# list_all_tf_filename = args[4]
# list_all_gene_filename = args[5]
if(length(args)>4){
    validated_edges_path = args[5]
}else{
    validated_edges_path = "NetworkEvaluation/validated_edges.rda"
}
# list_all_tf = read.table(list_all_tf_filename, header=FALSE)
# list_all_gene = read.table(list_all_gene_filename, header=FALSE)
model_eval = read.table(models_info_path, sep=",", header=TRUE)
model_eval$tp = NaN
model_eval$fp = NaN
model_eval$tpr = NaN
model_eval$fpr = NaN
model_eval$fn = NaN
model_eval$recall = NaN
model_eval$precision = NaN
model_eval$fscore = NaN
for(i in 1:nrow(model_eval)) {
    print(paste0("Working on ", i))
    row <- model_eval[i,]
    if(row$nb_candidate>0){
        net = read.table(paste0(row$datapath, "/", model_edge_filename), sep=",", header=TRUE)    
        load(file=validated_edges_path)
        # results = evaluate_network(net, subset_validated_edges=validated_edges, input_genes=list_all_gene, input_tfs=list_all_tf)
        results = evaluate_network(net, subset_validated_edges=validated_edges)
        print(results)
        model_eval[i,"tp"] = results$tp
        model_eval[i,"fp"] = results$fp
        model_eval[i,"tpr"] = results$tpr
        model_eval[i,"fpr"] = results$fpr
        model_eval[i,"fn"] = results$fn
        model_eval[i,"recall"] = results$recall
        model_eval[i,"precision"] = results$tp / (results$tp + results$fp)
        model_eval[i,"fscore"] = 2*results$tp / (2*results$tp + results$fp + results$fn)
        write.csv(results$edges, paste0(row$datapath, "/", found_edges_info_out_path), row.names=FALSE)
        write.csv(model_eval, paste0(models_info_out_path), row.names=FALSE)
    }   
}
write.csv(model_eval, models_info_out_path, row.names=FALSE)
