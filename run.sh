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

out_data="/media/carluerj/Data/These/Results/GRN_inference/Lauras_results/Full_candidate/"
tf_agi_corr="/media/carluerj/Data/These/DATA/gene_regulator_network/Laura_data/TF_AGI_corr.txt"


# python GenerateCartTree.py $data $out_data $tf_list_path $analysis_type
# $nitrate_list_path
# python GenerateGoGeneDendro.py $out_data
# python GenerateGRN.py $out_data
python FindOptParam.py $out_data
