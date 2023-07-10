from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys

data = sys.argv[1]
out_data = sys.argv[2]
tf_list_path = sys.argv[3]
analysis_type = sys.argv[4]
target_sub_list_path = None
if len(sys.argv) > 5:
    target_sub_list_path = sys.argv[5]
else:
    target_sub_list_path = None
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
# sub_target_list = cart.Y_target_list.tolist()[(cart.Y_target_list.tolist().index("AT3G45530")+1):]
# # for target in cart.Y_target_list:
# for target in sub_target_list:
#     cart.generate_CART_tree(target)
#     cart.eval_model(target)
#     cart.save_CART_tree(target)
#     cart.compile_cart_results(target, save=True, save_in_mem=False)
#     cart.save_cartlu_plot(target)
cart.load_compiled_results(filename = "compiled_table_a.csv")
cart.filter_cart_results()
cart.save_filtered_results()
