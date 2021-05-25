file_color_models = '/var/lib/lookiero/direct_buy_model/color_model/model/'
product_numeric_vars = 12  #

color_decomposition_filename_matplotlib = '/var/lib/lookiero/color_detection/LK_gc_matplotlib_distributions.csv'
color_decomposition_filename_lk = '/var/lib/lookiero/color_detection/LK_gc_lookiero_distributions.csv'
color_decomposition_filename_lk_heuristic = '/var/lib/lookiero/color_detection/LK_gc_lookiero_distributions_custom.csv'
color_info_filename_lk = '/var/lib/lookiero/color_detection/LK_colors_info.csv'
color_info_filename_lk_heuristic = '/var/lib/lookiero/color_detection/LK_colors_info_custom.csv'
color_info_filename_matplotlib = '/var/lib/lookiero/color_detection/matplotlib_colors_info.csv'
color_detected_filename = '/var/lib/lookiero/color_detection/colors_detected.csv'
not_color_decomposition_filename = '/var/lib/lookiero/color_detection/LK_gc_not_found.csv'
comb_tr_filename = '/var/lib/lookiero/direct_buy_model/data/compatible_clothes.csv'
comb_test_filename = '/Users/ivan/Downloads/test_data_color.csv'
group_color_filename = '/var/lib/lookiero/direct_buy_model/color_model/data/group_color_df.obj'
dict_test_filename = '/var/lib/lookiero/direct_buy_model/color_model/data/dict_test_info.obj'
dict_tr_filename = '/var/lib/lookiero/direct_buy_model/color_model/data/dict_tr_info.obj'

# file_compatible_clothes_train = '/var/lib/lookiero/direct_buy_model/color_model/data/comb_train.csv'
# file_compatible_clothes_val = '/var/lib/lookiero/direct_buy_model/color_model/data/comb_val.csv'
# file_compatible_clothes_test = 'test_con_ps'

dict_params_model = {
    'num_epoch': 200,
    'batch_size': 128,
    'shuffle': True,
    'num_positive': -1,
    'num_negative': 64,
    'verbose': 1,
    'initial_lr': 0.001,
    'batch_size_predict': 1024*128
}