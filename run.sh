
## PARAMETERS
# path to expression matrix should be a txt file or a h5 file
# Should be formated as bellow (tab_separated): 
# Row1 : CellID1\tCellID2\tCellID...
# Row2_to_Rown : AGI_name\tCellID1_count\tCellID2_count\tCellID..._count...
# with n the number of AGI to analyse
data=""
# path to output directory for result storage
out_data=""
# path to list of tf to analyse should be a txt file as : 
# TF1
# TF2
# ...
tf_list_path=""
# Kind of analysis, typical is "genes"
analysis_type="genes"
# path to sublist of tf to analyse should be a txt file. 
# This should be used if you want to focus on a subset of TF list.
target_sub_list_path=""

# CART generation
# With subsetlist of TF
#./run_cart_generation.sh $data $out_data $tf_list_path $analysis_type $target_sub_list_path
# With full data
./run_cart_generation.sh $data $out_data $tf_list_path $analysis_type 

# Gene Ontology analysis by TF generation
./run_GO_gene_dendro_generation.sh $out_data

# Two way dendrogram for TF/ cells expression analysis
./run_TW_dendro_generation.sh $out_data $data $tf_list_path

# GRN generation
if [ $(pgrep -a java | grep cytoscape | wc -l) -eq 0 ]
then
    echo "
      /!\ No cytoscape instance detected, please open cytoscape in order to generate GRN
      "
else
    ./run_grn_generation.sh $out_data
    echo "  Results can be found in ${out_data}network/"
fi
