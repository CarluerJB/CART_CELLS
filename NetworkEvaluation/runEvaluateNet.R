source("NetworkEvaluation/evaluateNetwork.R")

args = commandArgs(trailingOnly=TRUE)
models_info_path = args[1]
models_info_out_path = args[2]
model_edge_filename = args[3]
if(length(args)>3){
    validated_edges_path = args[4]
}else{
    validated_edges_path = "NetworkEvaluation/validated_edges.rda"
}


model_eval = read.table(models_info_path, sep=",", header=TRUE)
model_eval$tp = NaN
model_eval$fp = NaN
model_eval$tpr = NaN
model_eval$fpr = NaN
model_eval$fn = NaN
model_eval$recall = NaN
for(i in 1:nrow(model_eval)) {
    row <- model_eval[i,]
    if(row$nb_candidate>0){
        net = read.table(paste0(row$datapath, "/", model_edge_filename), sep=",", header=TRUE)    
        load(file=validated_edges_path)
        results = evaluate_network(net, subset_validated_edges=validated_edges)
        model_eval[i,"tp"] = results$tp
        model_eval[i,"fp"] = results$fp
        model_eval[i,"tpr"] = results$tpr
        model_eval[i,"fpr"] = results$fpr
        model_eval[i,"fn"] = results$fn
        model_eval[i,"recall"] = results$recall
    }   
}

write.csv(model_eval, models_info_out_path, row.names=FALSE)