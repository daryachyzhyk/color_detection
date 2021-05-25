import random
import numpy as np
import pandas as pd
import os
import pickle
import time
import tqdm
from joblib import delayed, Parallel
from typing import Any, Dict, List, Tuple, Generator

from data_core import util

from color_combination_model import color_config

logger = util.LoggingClass('color_model')


def create_pairs(dict_info: Dict, n_pos: int, n_neg: int) -> List:
    """ GENERA CANDIDATOS PARA EL ENTRENAMIENTO """
    inicio = time.time()
    logger.log('Creating candidates for training color')

    list_groupcolors = list(dict_info.keys())
    pairs = []

    def gc_info(x):
        positive_list = dict_info[x]
        negative_list = list(set(list_groupcolors) - set(positive_list))
        # todo: filtrar las de igual familia de la lista de negativos?
        #  todo aquello que podría combinar pero jamás irá junto por incompatibilidad de otros motivos
        # todo: updatear filtro de color del are_compatibles con el algo de Jonko si funciona mejor
        #  para los pares del extraction notes
        positive_reco = random.choices(positive_list, k=n_pos)
        negative_reco = random.choices(negative_list, k=n_neg)
        for positive in positive_reco:
            pairs.append([x, positive, 1])
        for negative in negative_reco:
            pairs.append([x, negative, 0])

    for gc in tqdm.tqdm(list_groupcolors):
        gc_info(gc)

    fin = time.time()
    logger.log('Creating candidates for training color OK, elapse time: {}'.format(fin - inicio))
    return pairs


class ColorGenerator(object):
    """ GENERADOR DE COLOR """

    def __init__(self, num_df: pd.DataFrame = None):
        self.num_df = num_df

    def get_batch_df(self, groupcolors: List) -> List:
        """ genera batch de una lista de grupocolores.
        groupcolors: lista de groupcolor = [gc_1, gc_2, ..., gc_n] """
        num_batch_df = self.num_df.loc[groupcolors]
        return [num_batch_df.values]

    def get_batch_df_list(self, list_groupcolors: List) -> List:
        """
        Genera batch de una lista de lista de grupocolores:
        list_groupcolors = [[gc_1, gc_2, ..., gc_n], [gcc_1, gcc_2, gcc_3, ..., gcc_n], [gccc_1, gccc_2, gccc_3, ..., gccc_n]]
        """

        return [self.get_batch_df(groupcolors=x) for x in list_groupcolors]

    @staticmethod
    def get_num_batches(len_info, batch_size):
        if len_info % batch_size == 0:
            num_batches = len_info // batch_size
        else:
            num_batches = len_info // batch_size + 1
        return num_batches

    def create_gen(self, dict_comb_info: Dict, n_pos: int, n_neg: int, batch_size: int, shuffle: bool) -> Generator:
        """ Generator for training """

        while True:
            list_pairs = create_pairs(dict_info=dict_comb_info, n_pos=n_pos, n_neg=n_neg)

            len_pairs = len(list_pairs)

            num_batches = self.get_num_batches(len_info=len_pairs, batch_size=batch_size)

            if shuffle:
                random.shuffle(list_pairs)

            for bid in range(num_batches):
                batch = list_pairs[bid * batch_size: (bid + 1) * batch_size]
                batch_for_df = [[x[0] for x in batch], [x[1] for x in batch]]  # [x[2] for x in batch]

                df_list = self.get_batch_df_list(list_groupcolors=batch_for_df)

                batch_apparel_1 = df_list[0]
                batch_apparel_2 = df_list[1]

                yield [batch_apparel_1, batch_apparel_2], np.array([1] * batch_size)  # todo: ésto del np.array qué es?

    def create_gen_predict(self, list_groupcolors: List, batch_size: int) -> Generator:
        while True:

            len_groupcolors = len(list_groupcolors)

            num_batches = self.get_num_batches(len_info=len_groupcolors, batch_size=batch_size)

            for bid in range(num_batches):
                batch = list_groupcolors[bid * batch_size: (bid + 1) * batch_size]

                df = self.get_batch_df(groupcolors=batch)

                yield df, np.array([1] * batch_size)


if __name__ == '__main__':
    from PIL import Image

    dict_comb_info = pickle.load(open('/var/lib/lookiero/direct_buy_model/data/triplet_approach/dict_tr_info.obj',
                                      'rb'))

    dict_comb_test = pickle.load(open('/var/lib/lookiero/direct_buy_model/data/triplet_approach/dict_test_info.obj',
                                      'rb'))

    cat_df = pd.read_csv(triplet_config.groupcolor_cat_filename).set_index('Unnamed: 0')
    num_df = pd.read_csv(triplet_config.groupcolor_num_filename).set_index('Unnamed: 0')

    self = ColorGenerator(path='/var/lib/lookiero/images_group_color/',
                          shape=(512, 224, 3), preprocess_fun=None, cat_df=cat_df, num_df=num_df,
                          use_frontal=False)

    train_gen = self.create_gen(dict_comb_info=dict_comb_info, n_pos=2, n_neg=4, batch_size=32, augument=False,
                                parallel=True, shuffle=False)
    aa = next(train_gen)

    example = 4
    Image.fromarray(aa[0][0][5][example].astype(np.uint8), 'RGB').show()
    Image.fromarray(aa[0][1][5][example].astype(np.uint8), 'RGB').show()
    Image.fromarray(aa[0][2][5][example].astype(np.uint8), 'RGB').show()
