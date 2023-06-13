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
out_data="/media/carluerj/Data/These/Results/GRN_inference/Lauras_results/head100/"
tf_list_path="/media/data/home/carluerjb/Documents/data/data_gene_regulatory_network/Lauras_data/GSE152766_Root_Atlas_TF_list.txt"
analysis_type="genes"

python FindOptParam.py $out_data
