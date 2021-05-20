import os
import tqdm
import requests
import pickle
import pandas as pd
import numpy as np
import psycopg2
from matplotlib import colors
import random
from data_core.core import Category
from color_combination_model import color_config

folder_path = '/Users/ivan/Downloads/color-test'


def is_broken_url(url):
    r = requests.get(url, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        return 0
    else:
        return 1


class ColorExtractorDB:
    def __init__(self, local=True):
        """ Extract colors from image and map their distribution to Lookiero and to Matplotlib colors"""
        if local:
            db_catalog = {'database': 'buying_back',
                          'user': 'buying_back_ro',
                          'password': 'ShuperShekret',
                          'host': '127.0.0.1',
                          'port': 5433}
        else:
            # db_catalog = {'host': 'db-buying-back-slave.lookiero.tech',
            db_catalog = {'host': '127.0.0.1',
                          # 'port': 5432,
                          'port': 5433,
                          'dbname': 'buying_back',
                          'user': 'buying_back_ro',
                          'password': 'ShuperShekret'}

        self.conn_catalog = psycopg2.connect(**db_catalog)
        self.lk_colors_rgb = None
        self.lk_colors_rgb_idx = None
        self.local = local
        self.dict_LK_colors = {}
        self.data_LK_colors = None
        self.data_matplotlib_colors = None
        self.matplotlib_colors_rgb = None

    def get_rgb_standard(self, color_name):
        """
        Get RGB standard representation of a given color name
        """
        return colors.to_rgb(color_name)

    def standardize_rgb_inverse(self, rgb_color):
        """
        Transform from RGB between [0, 1] to RGB values between [0, 255]
        """
        return int(255*rgb_color[0]), int(255*rgb_color[1]), int(255*rgb_color[2])

    def standardize_rgb(self, rgb_color):
        """
        Transform from RGB between [0, 255] to RGB values between [0, 1]
        """
        return rgb_color[0]/255, rgb_color[1]/255, rgb_color[2]/255

    def get_LK_color(self):
        """
        Method to get LK colors and their RGB representation.
        """
        try:
            query_colors = "select code, hexadecimal from color;"
            data_colors = pd.read_sql_query(query_colors, self.conn_catalog)

            for row in data_colors.values:
                rgb_color = list(np.round(colors.hex2color("".join(["#", row[1].lower()])), 2))
                self.dict_LK_colors[row[0]] = rgb_color

            self.lk_colors_rgb = {k: [self.standardize_rgb_inverse(v)[0], self.standardize_rgb_inverse(v)[1],
                                      self.standardize_rgb_inverse(v)[2]]
                                  for k, v in self.dict_LK_colors.items()}
            self.lk_colors_rgb_idx = {"_".join([str(v[0]), str(v[1]), str(v[2])]): k for k, v in self.lk_colors_rgb.items()}
        except Exception as error:
            print("Error extracting colors form LK db.")
            print(error)

    def get_matplotlib_color_data(self):
        """ Method to extract all the available colors in Matplotlib with their information."""
        try:
            list_colors = []
            self.matplotlib_colors_rgb = dict()
            for color_name in sorted(colors.cnames.keys()):
                dict_matplotlib_color = dict()
                dict_matplotlib_color["Name"] = color_name

                red_st, green_st, blue_st = self.get_rgb_standard(color_name)
                red, green, blue = self.standardize_rgb_inverse([red_st, green_st, blue_st])

                dict_matplotlib_color["Red"] = red
                dict_matplotlib_color["Green"] = green
                dict_matplotlib_color["Blue"] = blue

                list_colors.append(dict_matplotlib_color)
                self.matplotlib_colors_rgb[color_name] = [red, green, blue]

            self.data_matplotlib_colors = pd.DataFrame(list_colors)
        except Exception as error:
            print("Error extracting colors form matplotlib.")
            print(error)


if __name__ == "__main__":
    samples_color_test_df = pd.read_csv('/var/lib/lookiero/color_detection/samples_color_test.csv')
    matplotlib_df = pd.read_csv('/var/lib/lookiero/color_detection/matplotlib_colors_info.csv')
    matplotlib_df.set_index('group_color', inplace=True)
    # gc_to_matplotlib = matplotlib_df.idxmax(axis="columns")
    lookiero_df = pd.read_csv('/var/lib/lookiero/color_detection/LK_colors_info.csv')
    repo_file = '/var/lib/lookiero/repo_direct_buy.obj'
    repo = pickle.load(open(repo_file, 'rb'))
    colorext = ColorExtractorDB(local=True)
    colorext.get_LK_color()
    colorext.get_matplotlib_color_data()

    # Filtro para que no sea demasiado confuso para las PS
    lookiero_df = lookiero_df[lookiero_df['lk_color_4'].isna()]
    list_samples = []

    for idx, row in lookiero_df.iterrows():
        try:
            # Filtro para evaluarlo sobre las prendas target de salida
            if row['group_color'] in repo.products and max(repo.products[row['group_color']].seasons) >= 9:
                group = repo.products[row['group_color']].group
                color = repo.products[row['group_color']].color
                groupcolor = "".join([group, color])
                url = f"https://s3-eu-west-1.amazonaws.com/catalogo.labs/{group}/{groupcolor}.jpg"
                if not is_broken_url(url):
                    color_matplotlib = matplotlib_df.loc[row['group_color']]['matplotlib_color_1']
                    dict_row = {'gc': row['group_color'],
                                # 'url': url,
                                'rgb_detectado': row['img_color_1_rgb'],
                                # 'color_informado': color,
                                # 'rgb_color_informado': "_".join([str(x) for x in colorext.lk_colors_rgb[color]]),
                                'color_match_matplotlib_new': color_matplotlib,
                                'rgb_match_matplotlib_new': "_".join([str(x) for x in colorext.matplotlib_colors_rgb[color_matplotlib]]),
                                'color_match_fashion_labs_new': row['lk_color_1'],
                                'rgb_match_fashion_labs_new': "_".join([str(x) for x in colorext.lk_colors_rgb[row['lk_color_1']]])
                                }
                    list_samples.append(dict_row)
        except Exception as error:
            print(error)
            continue

    color_samples_new_df = pd.DataFrame(list_samples)
    samples_color_test_df = samples_color_test_df.merge(color_samples_new_df, on='gc', how='inner').drop(columns=['rgb_detectado_x']).rename(columns={'rgb_detectado_y': 'rgb_detectado'})
    column_list = [column_name for column_name in list(samples_color_test_df.columns) if column_name not in ['gc', 'url', 'rgb_detectado']]
    samples_color_test_df = samples_color_test_df[['gc', 'url', 'rgb_detectado'] + column_list]
    samples_color_test_df.to_csv('/var/lib/lookiero/color_detection/samples_color_test_new.csv', index=False)
    print("ok")

    """
    repo_file = '/var/lib/lookiero/repo_direct_buy.obj'
    repository_db = pickle.load(open(repo_file, 'rb'))
    repository_file = '/var/lib/lookiero/repo_v2.5.0_data_core.obj'
    repo = pickle.load(open(repository_file, 'rb'))

    notes_df = pd.read_csv(color_config.comb_tr_filename)
    notes_triplets_df = notes_df[['order_id', 'group_color_1', 'group_color_2', 'group_color_3']].dropna()
    notes_triplets_df.insert(len(notes_triplets_df.columns), 'n_accessories_in_triplet',
                             [len([gc for gc in triplet if
                                   repository_db.products[gc].get_category() == Category.accessory])
                              for triplet in
                              notes_triplets_df[['group_color_1', 'group_color_2', 'group_color_3']].values])

    list_accessories_boxes = []
    list_order_id = list(
        np.unique(notes_triplets_df[notes_triplets_df['n_accessories_in_triplet'] == 0]['order_id'].values))
    clients = [x for x in repo.clients.values() if len(x.boxes) > 0]

    for client in tqdm.tqdm(clients):
        client_boxes = client.boxes
        for box in client_boxes:
            try:
                if box.id in list_order_id:
                    # todo: save accessory gc if any in dict
                    list_accessories = ["_".join([product.group, product.color])
                                        for product in box.products_purchased.values()
                                        if product.get_category() == Category.accessory]
                    list_accessories.extend(["_".join([product.group, product.color])
                                             for product in box.products_rejected.values()
                                             if product.get_category() == Category.accessory])
                    if len(list_accessories) > 0:
                        dict_accessories_boxes = dict()
                        dict_accessories_boxes['order_id'] = box.id
                        for i, item in enumerate(list_accessories):
                            dict_accessories_boxes[f'accessory_{i+1}'] = item
                        list_accessories_boxes.append(dict_accessories_boxes)
            except:
                continue
    df_accessories = pd.DataFrame(list_accessories_boxes)
    quatriplets_df = notes_triplets_df.merge(df_accessories, on='order_id', how='left')
    quatriplets_df = quatriplets_df.drop(columns=['n_accessories_in_triplet'])
    quatriplets_df.to_csv('/Users/ivan/Downloads/quatriplets_accessory.csv', index=False)
    print("Mean triplets with accessory:")
    print(quatriplets_df.isna().mean()['accessory_1'])
    print("Amount triplets with accessory:")
    print(quatriplets_df[~quatriplets_df['accessory_1'].isna()].shape[0])
    print("Done")
    """
    """
    countries_list = ['ES', 'PT', 'GB', 'FR', 'IT']
    quatriplets_df = pd.read_csv('/Users/ivan/Downloads/quatriplets_accessory.csv').drop(columns=['accessory_2']).dropna()
    quatriplets_df.insert(len(quatriplets_df.columns), 'country_code', [random.sample(countries_list, 1)[0]
                                                                        for x in quatriplets_df['order_id'].values])
    sample_df = quatriplets_df.groupby('country_code').sample(200)
    sample_df.to_csv('/Users/ivan/Downloads/test_quatriplets_accessory.csv', index=False)
    print("Ok")
    """
