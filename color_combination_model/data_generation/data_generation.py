import pickle
import pandas as pd
import tqdm
from color_combination_model import color_config
from data_core import util

logger = util.LoggingClass('color_model')


class ColorInfo(object):

    def __init__(self, color_decomp_df: pd.DataFrame, comb_df: pd.DataFrame, comb_test_df: pd.DataFrame):
        self.color_decomp_df = color_decomp_df
        self.comb_df = comb_df
        self.comb_test_df = comb_test_df

        self.list_group_colors = []
        self.group_color_df = None
        self.dict_comb_test = {}
        self.dict_info_comb_train = {}

    def update_info_products(self, gc_1, gc_2):
        """ Makes sure that the look is in the list of products which we have color decomposition about """
        if gc_1 not in self.color_decomp_df.index or gc_2 not in self.color_decomp_df.index:
            return False
        return True

    def validate_pair(self, gc_1, gc_2, compare_with_pair_test=True):
        """ Is it a valid pair? """
        if self.update_info_products(gc_1=gc_1, gc_2=gc_2):
            if compare_with_pair_test and ((gc_1, gc_2) in self.dict_comb_test or (gc_2, gc_1) in self.dict_comb_test):
                return False
            return True
        return False

    def get_test_info(self) -> bool:
        """ Prepare test to dict for color model """
        try:
            dict_comb_target = {}

            for index, row in tqdm.tqdm(self.comb_test_df.iterrows(), total=self.comb_test_df.shape[0]):
                gc_1 = row['gc_1']
                gc_2 = row['gc_2']
                target = row['compatible']

                if self.validate_pair(gc_1=gc_1, gc_2=gc_2, compare_with_pair_test=False):
                    dict_comb_target[(gc_1, gc_2)] = target
                    dict_comb_target[(gc_2, gc_1)] = target

            self.dict_comb_test = dict_comb_target
            return True

        except Exception as error:
            logger.log('Error in ColorInfo / get_test_info', exc_info=True, level='error')
            logger.log(error)
            return False

    def transform_extraction_notes(self, k=2):
        """ Transforms a data frame containing combinations of k apparels into a df with only pairs """
        if 3 >= k > 1:
            try:
                train_df = pd.DataFrame()
                gc_df = self.comb_df[['group_color_1', 'group_color_2']].dropna()
                train_df = pd.concat([train_df, gc_df], ignore_index=True)
                if k == 3:
                    gc_df = self.comb_df[['group_color_1', 'group_color_2', 'group_color_3']].dropna()
                    temp_df = gc_df[['group_color_1', 'group_color_3']]
                    temp_df.columns = ['group_color_1', 'group_color_2']
                    train_df = pd.concat([train_df, temp_df], ignore_index=True)
                    temp_df = gc_df[['group_color_2', 'group_color_3']]
                    temp_df.columns = ['group_color_1', 'group_color_2']
                    train_df = pd.concat([train_df, temp_df], ignore_index=True)
                self.comb_df = train_df
            except Exception as error:
                logger.log('Error in ColorInfo / transform_extracton_notes')
                return False
        else:
            logger.log("The input garments set for the Training DF is not a valid.", exc_info=True, level='error')
            return False

    def get_train_info(self) -> bool:
        """ Prepares training data from train_comb to dict for color model """
        try:
            dict_gc_comb = {}

            for index, row in tqdm.tqdm(self.comb_df.iterrows(), total=self.comb_df.shape[0]):
                gc_1 = row['group_color_1']
                gc_2 = row['group_color_2']

                if self.validate_pair(gc_1=gc_1, gc_2=gc_2, compare_with_pair_test=True):
                    if gc_1 not in dict_gc_comb:
                        dict_gc_comb[gc_1] = []

                    if gc_2 not in dict_gc_comb:
                        dict_gc_comb[gc_2] = []

                    dict_gc_comb[gc_1].append(gc_2)
                    dict_gc_comb[gc_2].append(gc_1)

            dict_gc_comb = {k: list(set(v)) for k, v in dict_gc_comb.items()}
            dict_gc_comb = {k: v for k, v in dict_gc_comb.items() if len(v) > 0}

            self.dict_info_comb_train = dict_gc_comb
            return True
        except Exception as error:
            logger.log('Error in ColorInfo / get_train_info', exc_info=True, level='error')
            logger.log(error)
            return False

    def prepare_info(self) -> bool:
        """ Prepares the datasets for training and testing of Color model """
        try:
            if 'group_color' in list(self.color_decomp_df.columns):
                # We set the group_colors as indexes
                self.color_decomp_df.set_index('group_color', inplace=True)
            # In case that we use extraction_notes() as
            self.transform_extraction_notes(k=3)
            self.get_test_info()
            self.get_train_info()

            list_group_colors = [x[0] for x in self.dict_comb_test.keys()] + \
                                [x[1] for x in self.dict_comb_test.keys()] + list(self.dict_info_comb_train.keys())

            self.list_group_colors = list(set(list_group_colors))
            self.group_color_df = self.color_decomp_df.loc[self.list_group_colors]

            return True
        except Exception as error:
            logger.log('Error in ColorInfo / prepare_info', exc_info=True, level='error')
            logger.log(error)
            return False

    def save(self) -> bool:
        self.group_color_df.to_pickle(color_config.group_color_filename)
        with open(color_config.dict_test_filename, 'wb') as f:
            pickle.dump(self.dict_comb_test, f)
        with open(color_config.dict_tr_filename, 'wb') as f:
            pickle.dump(self.dict_info_comb_train, f)
        return True


if __name__ == '__main__':

    color_decomposition_df = pd.read_csv(color_config.color_decomposition_filename_matplotlib)

    comb_tr_df = pd.read_csv(color_config.comb_tr_filename)

    comb_test_df = pd.read_csv(color_config.comb_test_filename)

    self = ColorInfo(color_decomp_df=color_decomposition_df, comb_df=comb_tr_df, comb_test_df=comb_test_df)

    self.prepare_info()

    self.save()
