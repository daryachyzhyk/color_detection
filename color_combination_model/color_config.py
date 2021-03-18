file_color_models = '/var/lib/lookiero/direct_buy_model/color_model/model/'
product_numeric_vars = 12  #

color_decomposition_filename = '/var/lib/lookiero/color_detection/LK_gc_matplotlib_distributions.csv'
not_color_decomposition_filename = '/var/lib/lookiero/color_detection/LK_gc_not_found_color_distributions.csv'
comb_tr_filename = '/var/lib/lookiero/direct_buy_model/data/compatible_clothes.csv'
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