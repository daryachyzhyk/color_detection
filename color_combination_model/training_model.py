import pickle
import os
import pandas as pd
from color_combination_model import color_config
from data_core import util

logger = util.LoggingClass('color_model')


class ColorModelTraining(object):
    def __init__(self):

        self.dir = color_config.file_color_models

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.comb_train_df = None
        self.comb_val_df = None
        self.comb_test_df = None
        self.list_groupcolors = None

    def generate_data(self, repo_direct_buy, comb_train_df, comb_val_df, comb_test_df):
        logger.log('StyleTraining/generate_data')

        siamese_info = SiameseInfo(families=self.families)
        comb_train_df, comb_val_df, comb_test_df, product_numeric_df, b_info = siamese_info.prepare_data(repo_direct_buy=repo_direct_buy,
                                                                                                         comb_train_df=comb_train_df,
                                                                                                         comb_val_df=comb_val_df,
                                                                                                         comb_test_df=comb_test_df)

        if b_info > 0:
            self.comb_train_df = comb_train_df
            self.comb_val_df = comb_val_df
            self.comb_test_df = comb_test_df
            self.product_numeric_df = product_numeric_df
            self.list_groupcolors = list(product_numeric_df.index)

        return b_info

    def load_and_train(self, repo_direct_buy, comb_train_df, comb_val_df, comb_test_df):
        g_b = self.generate_data(repo_direct_buy=repo_direct_buy, comb_train_df=comb_train_df, comb_val_df=comb_val_df,
                                 comb_test_df=comb_test_df)
        if g_b:
            p_b = self.preprocessing_data()
            if p_b:
                self.save()
                t_b = self.train_dl()
                if t_b:
                    tm_b = self.get_threshold_metrics()
                    return tm_b
                return t_b
            return p_b
        return g_b


if __name__ == '__main__':
    repo_file = '/var/lib/lookiero/repo_direct_buy.obj'
    repo_db = pickle.load(open(repo_file, 'rb'))
    comb_training_df = pd.read_csv(color_config.file_compatible_clothes_train)
    # comb_val_df = pd.read_csv(color_config.file_compatible_clothes_val)
    comb_testing_df = pd.read_csv(color_config.file_compatible_clothes_test)
