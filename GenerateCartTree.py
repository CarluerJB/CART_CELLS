from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys
import argparse

# PARAMETERS
parser = argparse.ArgumentParser(
    prog="GenerateCartTree",
    description="This programm is meant to generate CART TREE for each gene against all TF",
    epilog="Design by Jean-Baptiste CARLUER under the supervision of Gabriel Krouk and André Mas",
)
parser.add_argument(
    "-d",
    "--data_path",
    help="Path to your data, data should be either a .txt matrix tab separated or a .h5 file.",
    required=True,
)
parser.add_argument(
    "-o",
    "--out_path",
    help="Path to your out directory, directory will be automatically created",
    required=True,
)
parser.add_argument(
    "-tf",
    "--tf_list_path",
    help="Path to your tf list, tf in the file will be search in the data file and used as tf. Should be one tf/row",
    required=True,
)
parser.add_argument(
    "-a",
    "--analysis_type",
    help="Indicator of the directionnality of the study, should we search for best tf for each genes or the opposite ?",
    default="genes",
)
parser.add_argument(
    "-tgl",
    "--target_list_path",
    help="List of TG to study instead of the full available TG",
    default=None,
)
args = parser.parse_args()
data = args.data_path
out_data = args.out_path
tf_list_path = args.tf_list_path
analysis_type = args.analysis_type
target_sub_list_path = args.target_list_path


# data = sys.argv[1]
# out_data = sys.argv[2]
# tf_list_path = sys.argv[3]
# analysis_type = sys.argv[4]
# target_sub_list_path = None
# if len(sys.argv) > 5:
#     target_sub_list_path = sys.argv[5]
# else:
#     target_sub_list_path = None

# INIT CART TREE CLASS
cart = CART_TREE(
    data,
    tf_list_path,
    out_data,
    analysis_type,
    target_sub_list_path=target_sub_list_path,
)
cart.load_parameter_file()
cart.show_parameter()
cart.create_out_dir()
cart.load_GE_matrix()
cart.load_tf_list()
sub_target_list = cart.Y_target_list.tolist()
# TO USE TO RUN THE COMPUTATION FROM A CERTAIN POINT
# sub_target_list = cart.Y_target_list.tolist()[(cart.Y_target_list.tolist().index("AT5G63140")+1):]
for target in sub_target_list:
    cart.generate_CART_tree(target)
    cart.eval_model(target)
    cart.save_CART_tree(target)
    cart.compile_cart_results(target, save=True, save_in_mem=False)
    cart.save_cartlu_plot(target)
cart.load_compiled_results(filename="compiled_table_a.csv")
cart.filter_cart_results()
cart.save_filtered_results()
