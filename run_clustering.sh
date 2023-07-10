# DATA ON BEN
# data="/media/data/home/carluerjb/Documents/data/data_gene_regulatory_network/Amelie_data/norm_matrix_cleared.txt"
# out_data="/media/data/home/carluerjb/Documents/data/data_gene_regulatory_network/Results_CART_Amelie/full_results/"
# tf_list_path="/media/data/home/carluerjb/Documents/data/data_gene_regulatory_network/Amelie_data/tf_list.txt"
# analysis_type="genes"

# DATA ON LOCAL
# data="/media/carluerj/Data/These/DATA/gene_regulator_network/norm_matrix_cleared.txt"
# out_data="/media/carluerj/Data/These/Results/GRN_inference/Amelies_results/full_results/"
# tf_list_path="/media/carluerj/Data/These/DATA/gene_regulator_network/tf_list.txt"
# analysis_type="genes"


data="/media/data/home/carluerjb/Documents/data/data_gene_regulatory_network/Lauras_data/GSE152766_Root_Atlas_SCT_count.h5"
out_data="/home/carluerjb/results/GRN/Lauras_data/Full_candidate/"
tf_list_path="/media/data/home/carluerjb/Documents/data/data_gene_regulatory_network/Lauras_data/GSE152766_Root_Atlas_TF_list.txt"
nitrate_list_path="/media/data/home/carluerjb/Documents/data/data_gene_regulatory_network/Lauras_data/Nitrate_top_genes.txt"
analysis_type="genes"

# python GenerateCartTree.py $data $out_data $tf_list_path $analysis_type
# $nitrate_list_path
# python GenerateGoGeneDendro.py $out_data
python GenerateTwGeneDendro.py $data $out_data $tf_list_path
# python GenerateGRN.py $out_data
# python FindOptParam.py $out_data