from lib.cart_class import CART_TREE
from lib.grn_class import GRN
from lib.dendrogram_tw_class import DENDROGRAM_TW
import sys

data = sys.argv[1]
out_data = sys.argv[2]
tf_list_path = sys.argv[3]
analysis_type = sys.argv[4]

cart = CART_TREE(data, tf_list_path, out_data, analysis_type)
cart.load_parameter_file()
cart.show_parameter()
cart.create_out_dir()
cart.load_GE_matrix()
cart.load_tf_list()
for target in cart.Y.columns:
    cart.generate_CART_tree(target)
    cart.eval_model(target)
    cart.save_CART_tree(target)
    cart.compile_cart_results(target)
    cart.save_cartlu_plot(target)
cart.save_compiled_results()
cart.filter_cart_results()
cart.save_filtered_results()
