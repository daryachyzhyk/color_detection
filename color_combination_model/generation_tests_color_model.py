import pandas as pd
import requests
import os
import tqdm
import pickle
import random
from data_core.core import Category

path_data = '/var/lib/lookiero/modelo_color/data'

if not os.path.exists(path_data):
    os.makedirs(path_data)


class CombineApparels:
    """
    Base class to recommend apparels based on business rules.
    This can be later applied to pairs of apparels or to recommend accessories.
    """

    def __init__(self, repo):
        self.repo = repo

        warm_weather = 0
        cold_weather = 1
        soft_weather = 2
        soft_cold_weather = 3
        soft_warm_weather = 4

        self.climate_map = {
            cold_weather: 1,
            soft_cold_weather: 2,
            soft_weather: 3,
            soft_warm_weather: 4,
            warm_weather: 5
        }

        # We will populate this with the min and max climate values of the group_colors as we find them
        self.climate_min_maxs = dict()

        # We will populate this with the style values of the group_colors as we find them
        self.styles_prendas = dict()

        self.comb_categories = {
            1: {10, 11, 13, 21, 22, 23, 26},
            2: {5, 6, 10, 11, 13, 20, 21, 22, 23, 26},
            3: {5, 6, 10, 11, 13, 20, 21, 22, 23, 26},
            4: {5, 6, 9, 10, 11, 13, 20, 21, 22, 23, 26},
            5: {2, 3, 4, 8, 9, 10, 11, 13, 21, 22, 23, 26},
            6: {2, 3, 4, 8, 9, 10, 11, 13, 21, 22, 23, 26},
            8: {5, 6, 10, 11, 13, 20, 22, 23, 26},
            9: {4, 5, 6, 10, 11, 13, 20, 22, 23, 26},
            10: {1, 2, 3, 4, 5, 6, 8, 9, 12, 20},
            11: {1, 2, 3, 4, 5, 6, 8, 9, 12, 20},
            12: {9, 10, 11, 13, 21, 22, 23, 26},
            13: {1, 2, 3, 4, 5, 6, 8, 9, 12, 20},
            # 14: {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 26},
            # 16: {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 26},
            20: {2, 3, 4, 8, 9, 10, 11, 13, 21, 22, 23, 26},
            21: {1, 2, 3, 4, 5, 6, 12, 20},
            22: {1, 2, 3, 4, 5, 6, 8, 9, 12, 20},
            23: {1, 2, 3, 4, 5, 6, 8, 9, 12, 20},
            26: {1, 2, 3, 4, 5, 6, 8, 9, 12, 20}
        }

    def has_print(self, group_color):
        '''
        Method to know if the given apparel has print or not
        '''
        try:
            return not self.repo.products[group_color].characteristics['estampado_liso']
        except Exception as e:
            return False

    def has_largo(self, group_color):
        '''
        Method to know the length of a given apparel (in cm)
        '''
        try:
            return int(self.repo.products[group_color].characteristics["largoCM"])
        except Exception as e:
            return 0

    def is_loose(self, group_color):
        '''
        Method to know if the given apparel is loose or not
        '''
        try:
            return self.repo.products[group_color].characteristics["fit"].lower() == "holgado"
        except:
            return False

    def is_entallado(self, group_color):
        '''
        Method to know if the given apparel is entallado or not
        '''
        try:
            return self.repo.products[group_color].characteristics["fit"].lower() == "entallado"
        except:
            return False

    def has_special_detail(self, group_color):
        """
        Method to know if the given apparel has an special detail
        """
        try:
            return len([d for d in self.repo.products[group_color].json_item["detalle"] if
                        "especial" in d["text"].lower()]) > 0

        except:
            return False

    def has_acabado_special(self, group_color):
        """
        Method to know if the given apparel has an "acabado especial""
        """
        try:
            if "acabado" in self.repo.products[group_color].json_item.keys():
                return len([d for d in self.repo.products[group_color].json_item["detalle"] if
                            "especial" in d["text"].lower()]) > 0
            else:
                return False
        except:
            return False

    def list_of_fabrics(self, group_color):
        try:
            return [d['id'].lower() for d in self.repo.products[group_color].json_item.get("tejido", [])
                    if d['id'] is not None]
        except Exception as e:
            # logger.log(e)
            return []


class CombinePairsWithoutColor(CombineApparels):
    def __init__(self, repo):
        super().__init__(repo)

    def is_pair_compatible(self, group_color_1, group_color_2):
        """
        Checks if a pair of apparels combine
        :param group_color_1: Group color of the apparel number 1
        :param group_color_2: Group color of the apparel number 2
        :return: True or False. Whether they match
        """
        try:
            return self.is_family_compatible(group_color_1, group_color_2) and \
                   self.is_print_compatible(group_color_1, group_color_2) and \
                   self.is_detail_compatible(group_color_1, group_color_2) and \
                   self.is_acabado_compatible(group_color_1, group_color_2) and \
                   self.is_fit_compatible(group_color_1, group_color_2) and \
                   self.is_climate_compatible(group_color_1, group_color_2)
        except Exception as error:
            # logger.log(error)
            return False

    def is_family_compatible(self, group_color_1, group_color_2):
        """
        Method that returns if the given families are compatible or not.
        """
        try:
            fam_1 = int(self.repo.products[group_color_1].family)
            fam_2 = int(self.repo.products[group_color_2].family)

            return fam_1 in self.comb_categories[fam_2]
        except:
            return False

    def is_print_compatible(self, group_color_1, group_color_2):
        """
        Method to verify if the apparels combine by print
        """
        try:
            return not (self.has_print(group_color_1) and self.has_print(group_color_2))
        except Exception as e:
            return False

    def is_detail_compatible(self, group_color_1, group_color_2):
        """
        Method to verify if two apparels' combine by detail
        """
        try:
            return not (self.has_special_detail(group_color_1) and self.has_special_detail(group_color_2))
        except Exception as e:
            return False

    def is_acabado_compatible(self, group_color_1, group_color_2):
        """
        Method to verify if two apparels' combine by acabado
        """
        try:
            return not (self.has_acabado_special(group_color_1) and self.has_acabado_special(group_color_2))
        except Exception as e:
            return False

    def is_fit_compatible(self, group_color_1, group_color_2):
        """
        Method to verify if two apparels combine by fit
        """
        try:
            return not (self.is_loose(group_color_1) and self.is_loose(group_color_2))
        except:
            return False

    def is_climate_compatible(self, group_color_1, group_color_2):
        """
        Info: Method to compute if two apparels are compatible by climate.
        Output: True/False (bool)
        """
        try:
            max_prenda_1, min_prenda_1 = self.climate_min_maxs.get(group_color_1, (-1, -1))
            if max_prenda_1 == -1:
                climates_1 = [self.climate_map[x] for x in self.repo.products[group_color_1].climate]
                max_prenda_1 = max(climates_1)
                min_prenda_1 = min(climates_1)
                self.climate_min_maxs[group_color_1] = (max_prenda_1, min_prenda_1)

            max_prenda_2, min_prenda_2 = self.climate_min_maxs.get(group_color_2, (-1, -1))
            if max_prenda_2 == -1:
                climates_2 = [self.climate_map[x] for x in self.repo.products[group_color_2].climate]
                max_prenda_2 = max(climates_2)
                min_prenda_2 = min(climates_2)
                self.climate_min_maxs[group_color_2] = (max_prenda_2, min_prenda_2)

            if abs(max_prenda_2 - min_prenda_1) > 2 or abs(min_prenda_2 - max_prenda_1) > 2:
                return False  # incompatible climate
            return True
        except Exception as error:
            return False


if __name__ == "__main__":
    ## Parameters ##
    prendas_to_recommend = 20000  # n desired datapoints per country
    batch_size = 1000  # n prendas/test
    n_batches = int(prendas_to_recommend / batch_size)  # n batches that we will have
    n_groups = 5  # number of basic_score groups to divide into
    n_samples_group_batch = int(batch_size / n_groups)  # how many samples from each basic_score group per batch
    n_samples_cat_group_batch_10_pct = int(n_samples_group_batch / 10)  # it'll be used to distribute the samples
    # from each category inside each basic_group for a batch

    repo_file = '/var/lib/lookiero/repo_direct_buy.obj'
    repository = pickle.load(open(repo_file, 'rb'))
    basico_scores_df = pd.read_csv("/Users/ivan/Downloads/basico_scores.csv")

    combinador = CombinePairsWithoutColor(repository)

    # This is the dataframe that contains the information for each apparel
    prendas = ["_".join([v.group, v.color]) for k, v in repository.products.items() if v.seasons[-1] >= 7]
    prendas_df = pd.DataFrame({'group_color': prendas})
    prendas_df = pd.merge(prendas_df, basico_scores_df, how="left", left_on="group_color", right_on="groupcolor")
    prendas_df = prendas_df.drop(columns=['groupcolor']).dropna()
    prendas_df.insert(len(prendas_df.columns), 'group_score_basico',
                      [int(x * n_groups) + 1 for x in prendas_df['score'].values])
    prendas_df.insert(len(prendas_df.columns), 'cat_prenda',
                      [repository.products[gc].get_category() for gc in prendas_df['group_color'].values])

    # List of countries that will be used in the test
    list_countries = ['PT', 'FR', 'ES', 'GB', 'IT']
    n_countries = len(list_countries)

    list_categories = [Category.dress, Category.top, Category.outer, 'bottom_apparels']

    # We create a dictionary where we store all the apparels by basic_score_group
    prendas_by_score = dict()

    # Dictionary that will be used for recommending as first apparel
    # this is organized to ensure that in each batch we keep a balance of categories and basic_scores in the 1st apparel
    prendas_1_dict = dict()

    for basic_group in range(1, n_groups + 1):
        # We select the apparels that belong to this group of basic
        prendas_basic_group = prendas_df[prendas_df['group_score_basico'] == basic_group]
        list_prendas_basic_group = list(prendas_basic_group['group_color'].values)
        random.shuffle(list_prendas_basic_group)
        prendas_by_score[basic_group] = list_prendas_basic_group

        # Prenda 1 will have also the layer of abstraction by category, in order to offer some variety
        prendas_1_dict[basic_group] = dict()
        for cat in list_categories:
            if cat == 'bottom_apparels':
                list_apparels = list(prendas_basic_group[(prendas_basic_group['cat_prenda'] == Category.skirt) |
                                                         (prendas_basic_group['cat_prenda'] == Category.trousers)]
                                     ['group_color'].values)
                random.shuffle(list_apparels)
                prendas_1_dict[basic_group][cat] = list_apparels
            else:
                list_apparels = list(
                    prendas_basic_group[prendas_basic_group['cat_prenda'] == cat]['group_color'].values)
                random.shuffle(list_apparels)
                prendas_1_dict[basic_group][cat] = list_apparels

    # Ratio of recommendations for each category (out of 10)
    recs_by_cat_group_batch = {Category.dress: int(1 * n_samples_cat_group_batch_10_pct),
                               Category.top: int(4 * n_samples_cat_group_batch_10_pct),
                               Category.outer: int(2 * n_samples_cat_group_batch_10_pct),
                               'bottom_apparels': int(3 * n_samples_cat_group_batch_10_pct)
                               }
    # Recommendations for the test
    # Note: I know the following looks crazy (it probably is!)
    # This is done only to ensure that in each batch there is a balance of categories, basic_scores for each category
    # and balance of basic_scores for each apparel recommended for each of those balanced first apparels,
    # for each country, and hopefully not repeating pairs of apparels.
    # Try not to be like me when programming something like this, please
    list_recommendations = []
    idx_country = 0
    idx_group_basic = 1
    for batch in tqdm.tqdm(range(1, n_batches + 2)):
        # +2 in order to have enough, in case that there is any repeated pair
        for basic_group in range(1, n_groups + 1):
            print(f"Calculating for basic_group {basic_group}...\n\n")
            for cat in list_categories:
                # We shuffle the list of apparels for this basic_group and category
                random.shuffle(prendas_1_dict[basic_group][cat])
                list_apparels_app_1 = prendas_1_dict[basic_group][cat][:recs_by_cat_group_batch[cat]]
                for prenda1 in list_apparels_app_1:
                    # We can do this here, since we recommend one apparel for each garment
                    if idx_country >= n_countries:
                        # We assign each second apparel (from different basic_groups) to each country randomly
                        idx_country = 0
                        random.shuffle(list_countries)

                    if idx_group_basic > n_groups:
                        idx_group_basic = 1

                    # For each apparel, we recommend one apparel from a basic_group
                    random.shuffle(prendas_by_score[idx_group_basic])

                    for prenda2 in prendas_by_score[idx_group_basic]:
                        comp = combinador.is_pair_compatible(prenda1, prenda2)

                        if comp == 1:
                            # It has already been checked that the apparels have an online image
                            list_recommendations.append({'country_code': list_countries[idx_country],
                                                         'gc_1': prenda1,
                                                         'gc_2': prenda2})
                            idx_country += 1
                            idx_group_basic += 1
                            # Only one recommendation per apparel and basic_group
                            break

    recommendations_test_df = pd.DataFrame(list_recommendations).drop_duplicates(subset=['gc_1', 'gc_2'])

    print("\nSplitting into batches...")
    for batch in range(1, n_batches + 1):
        recommendations_test_df_batch = recommendations_test_df.iloc[((batch-1)*batch_size):(batch*batch_size)]
        recommendations_test_df_batch.to_csv(os.path.join(path_data, f"test_color_{batch}.csv"), index=False)
    recommendations_test_df.to_csv(os.path.join(path_data, f"test_color_all.csv"), index=False)
    print("Done.")
