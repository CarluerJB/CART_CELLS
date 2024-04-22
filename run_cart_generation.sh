data=$1
out_data=$2
tf_list_path=$3
analysis_type=$4
if [ $# -gt 4 ]
then
    target_sub_list_path=$5
    python GenerateCartTree.py -d $data -o $out_data -tf $tf_list_path -a $analysis_type -tgl $target_sub_list_path
elif [ $# -gt 4 ]
then
    python GenerateCartTree.py -d $data -o $out_data -tf $tf_list_path -a $analysis_type
else
    python GenerateCartTree.py
fi