import pandas as pd
import requests
import pickle
import random


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
            return self.is_print_compatible(group_color_1, group_color_2) and \
                   self.is_fit_compatible(group_color_1, group_color_2) and \
                   self.is_climate_compatible(group_color_1, group_color_2)
        except Exception as error:
            # logger.log(error)
            return False

    def is_print_compatible(self, group_color_1, group_color_2):
        """
        Method to verify if the apparels combine by print
        """
        try:
            return not (self.has_print(group_color_1) and self.has_print(group_color_2))
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


def is_broken_url(url):
    r = requests.get(url, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        return 0
    else:
        return 1


if __name__ == "__main__":
    repo_file = '/var/lib/lookiero/repo_direct_buy.obj'
    repository = pickle.load(open(repo_file, 'rb'))
    basico_scores_df = pd.read_csv("/Users/ivan/Downloads/basico_scores.csv")
    recos_df = pd.read_csv("/Users/ivan/PycharmProjects/data_accesorios/recos_approach1_2_without_stock_is_valid.csv")

    combinador = CombinePairsWithoutColor(repository)
    bolsos = ["_".join([v.group, v.color]) for k, v in repository.products.items() if
              v.family == 14 and v.seasons[-1] >= 6]
    random.shuffle(bolsos)

    recos_df['compatible'] = [int(combinador.is_pair_compatible(x, y)) for x, y in recos_df[['gc_1', 'gc_2']].values]

    # todo: mirar que ofrezcan un look completo, pero que combinen menos por color, pero que haya balance de basicos
    recos_df = pd.merge(recos_df, basico_scores_df, how="left", left_on="gc_1", right_on="groupcolor")
    recos_df = pd.merge(recos_df, basico_scores_df, how="left", left_on="gc_2", right_on="groupcolor")
    recos_df = recos_df.rename(columns={'score_x': 'score_gc_1', 'score_y': 'score_gc_2'})
    recos_df = recos_df.drop(columns=['groupcolor_x', 'groupcolor_y'])
    recos_color_df = recos_df.loc[(recos_df['compatible'] == 1) & (recos_df['valid_look_acc'] == 1)].copy().dropna()
    recos_color_df['mean_score'] = [(x + y) / 2 for x, y in recos_color_df[['score_gc_1', 'score_gc_2']].values]
    recos_color_df['group_score_gc_1'] = [int(x / 0.2) + 1 for x in recos_color_df['score_gc_1'].values]
    recos_color_df['group_score_gc_2'] = [int(x / 0.2) + 1 for x in recos_color_df['score_gc_2'].values]
    recos_color_df['group_mean_score'] = [int(x / 0.2) + 1 for x in recos_color_df['mean_score'].values]

    recos_color_df = pd.concat([recos_color_df.copy().groupby('group_score_gc_1').sample(1000),
                                recos_color_df.copy().groupby('group_score_gc_2').sample(1000),
                                recos_color_df.copy().groupby('group_mean_score').sample(1000)]).drop_duplicates()
    recos_color_df.insert(len(recos_color_df.columns), 'country_code', ["es"]*recos_color_df.shape[0])
    recos_color_df = recos_color_df[['gc_1', 'gc_2', 'country_code']]
    recos_color_df.to_csv("/Users/ivan/Downloads/test_modelo_color.csv", index=False)
    print("Finalizado")
