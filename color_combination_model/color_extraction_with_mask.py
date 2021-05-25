import os
import PIL
import cv2
import extcolors
import re
import io
import tqdm
import numpy as np
import pandas as pd
import requests
import urllib.request as url_request
from matplotlib import colors
from skimage.color import rgb2lab, deltaE_ciede2000  # deltaE_cie76
from data_core import util

import config as cfg
from color_combination_model import color_config
from color_detection_extcolors import ColorExtraction
from image_segmentation_apparels.main import find_mask

logger = util.LoggingClass('color_detection')

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 2500)


class ColorExtractionwithMask(ColorExtraction):
    def __init__(self, local=True):
        """ Extract colors from image and map their distribution to Lookiero and to Matplotlib colors"""
        super().__init__(local=local)
        self.lk_colors_rgb = None
        self.lk_colors_rgb_idx = None
        self.local = local
        self.dict_LK_colors = {}
        self.data_LK_colors = None
        self.data_matplotlib_colors = None

        self.dict_heuristics = {
                "C7": {
                    "values": ["C12", "C15", "C20", "C35"],
                    "exception": False,
                    "dist_exception": 0.0
                },
                "C35": {
                    "values": ["C12", "C15", "C20", "C33", "C41"],
                    "exception": False,
                    "dist_exception": 0.0
                },
                "C30": {
                    "values": ["C8", "C15", "C31", "C41", "C43"],
                    "exception": True,
                    "dist_exception": 4.8
                },
                "C15": {
                    "values": ["C20", "C34"],
                    "exception": True,
                    "dist_exception": 4.8
                },
                "C22": {
                    "values": ["C2", "C26"],
                    "exception": False,
                    "dist_exception": 0.0
                }
            }

    def get_image_from_s3(self, group, color, save=True, path='/var/lib/lookiero/images'):
        """
        Return the main image from S3 for a given group and color.
        """
        group_color = "".join([group, color])
        img_url = 'https://s3-eu-west-1.amazonaws.com/catalogo.labs/{}/{}.jpg'.format(group, group_color)

        if save:
            try:
                img = np.array(PIL.Image.open(path + '/' + group_color + '.jpg'))
            except:
                try:
                    url_request.urlretrieve(img_url, path + '/' + group_color + '.jpg')
                    img = np.array(PIL.Image.open(path + '/' + group_color + '.jpg'))
                except:
                    logger.log(f'Error loading get_image_from_s3, groupc:{group}, color: {color}', exc_info=True,
                               level='error')
                    img = None
            return img
        try:
            return np.array(PIL.Image.open(requests.get(img_url, stream=True).raw))
        except:
            logger.log(f'Error loading get_image_from_s3, groupc:{group}, color: {color}', exc_info=True, level='error')
            return None

    def get_LK_images_info(self, filtered=None):
        """
        Info: Method to extract all the different group_colors existing in the LK catalog
        :param filtered Refers to the earliest date from which you would like to get clothes from.
        """
        try:
            color_decomp_df_mat = pd.DataFrame()
            color_decomp_df_lk = pd.DataFrame()
            color_decomp_df_lk_heuristic = pd.DataFrame()
            color_info_lk_df = pd.DataFrame()
            color_info_lk_heuristic_df = pd.DataFrame()
            color_info_matplotlib_df = pd.DataFrame()
            color_detected_df = pd.DataFrame()
            not_color_decomp_df = pd.DataFrame()
            list_computed_gc = []
            list_not_found_gc = []

            # Get the rgb decomposition of Lookiero colors
            self.get_LK_color()
            # Get the yuv, standardize data of Lookiero colors
            self.get_LK_color_data()
            # Get the yuv, standardize data of Matplotlib colors
            self.get_matplotlib_color_data()

            if os.path.exists(color_config.color_info_filename_lk):
                color_info_lk_df = pd.read_csv(color_config.color_info_filename_lk)
                color_info_lk_df.set_index('group_color', inplace=True)
                list_computed_gc = list(color_info_lk_df.index)
            if os.path.exists(color_config.color_info_filename_lk_heuristic):
                color_info_lk_heuristic_df = pd.read_csv(color_config.color_info_filename_lk_heuristic)
                color_info_lk_heuristic_df.set_index('group_color', inplace=True)
            if os.path.exists(color_config.color_info_filename_matplotlib):
                color_info_matplotlib_df = pd.read_csv(color_config.color_info_filename_matplotlib)
                color_info_matplotlib_df.set_index('group_color', inplace=True)
            if os.path.exists(color_config.color_decomposition_filename_lk):
                color_decomp_df_lk = pd.read_csv(color_config.color_decomposition_filename_lk)
                color_decomp_df_lk.set_index('group_color', inplace=True)
            if os.path.exists(color_config.color_decomposition_filename_lk_heuristic):
                color_decomp_df_lk_heuristic = pd.read_csv(color_config.color_decomposition_filename_lk_heuristic)
                color_decomp_df_lk_heuristic.set_index('group_color', inplace=True)
            if os.path.exists(color_config.color_decomposition_filename_matplotlib):
                color_decomp_df_mat = pd.read_csv(color_config.color_decomposition_filename_matplotlib)
                color_decomp_df_mat.set_index('group_color', inplace=True)
            if os.path.exists(color_config.color_detected_filename):
                color_detected_df = pd.read_csv(color_config.color_detected_filename)
                color_detected_df.set_index('group_color', inplace=True)
            if os.path.exists(color_config.not_color_decomposition_filename):
                not_color_decomp_df = pd.read_csv(color_config.not_color_decomposition_filename)
                list_not_found_gc = list(set(not_color_decomp_df['group_color'].values))

            list_gc_dict_lk = []
            list_gc_dict_lk_heuristic = []
            list_gc_dict_matplotlib = []
            list_gc_info_lk = []
            list_gc_info_lk_heuristic = []
            list_gc_info_matplotlib = []
            list_gc_image_color_detected = []
            list_not_found_groupcolors = []

            # Creamos un diccionario de colores que suele consistentemente matchear a un color no deseado
            # En este caso nos apoyamos de lo que informa Fashion Labs
            # P. ej.
            # Cuando el col. inf. es C7, los colores que matchean a C12,C15,C20 o C35 suelen estar mal y deberían ser C7
            # Estos colores podríamos sustituirlos por el C7, que sería el deseado en ese caso.
            # En algunos casos, el color detectado está tan cerca de uno existente, que merece la pena quedarse con él.
            # Para ello existe la clave 'exception' y la 'dist_exception' máxima con el matcheo para fiarnos del match



            if self.local:
                suffix_query = f" where date_created >= {filtered}" if filtered else ""
                query = f"select distinct `_group_id` as 'group', color from variations{suffix_query};"

                data_group = pd.read_sql_query(query, self.conn_mysql)

                for group, color in tqdm.tqdm(data_group.values):
                    group_color = "_".join([group, color])
                    if group_color not in list_computed_gc and group_color not in list_not_found_gc:
                        try:
                            image = self.get_image_from_s3(group, color)
                            dict_colors = self.extract_colors_from_image(image, group_color, tolerance=12)
                            color_distribution_lk = {color_name: 0. for color_name in sorted(self.dict_LK_colors.keys())}
                            color_distribution_lk_heuristic = {color_name: 0. for color_name in sorted(self.dict_LK_colors.keys())}
                            color_distribution_matplotlib = {color_name: 0. for color_name in colors.cnames.keys()}

                            dict_image = dict()
                            dict_image["group_color"] = group_color
                            i = 1
                            # Match each detected color to a similar one in the database
                            if color in self.dict_heuristics:
                                # Las heurísticas pueden aplicar
                                for rgb_str, pct in dict_colors.items():
                                    dict_image["img_color_{}_rgb".format(i)] = rgb_str
                                    dict_image["pct_color_{}".format(i)] = pct
                                    matcolor = self.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                                    lkcolor, dist_lk = self.get_most_similar_LK_color([int(x) for x in rgb_str.split("_")])

                                    # Comprobamos si se cumplen las heuristicas
                                    if lkcolor in self.dict_heuristics[color]["values"]:
                                        if self.dict_heuristics[color]["exception"] and \
                                                dist_lk <= self.dict_heuristics[color]["dist_exception"]:
                                            lkcolor_heuristic = lkcolor
                                        else:
                                            lkcolor_heuristic = color
                                    else:
                                        lkcolor_heuristic = lkcolor

                                    if lkcolor == "C1" and dist_lk < 0.5 and pct > 0.35 and color != "C1":
                                        lkcolor_heuristic = color

                                    # We assign the percentage of the color to the dictionary
                                    color_distribution_matplotlib[matcolor] += pct
                                    color_distribution_lk[lkcolor] += pct
                                    color_distribution_lk_heuristic[lkcolor_heuristic] += pct
                                    i += 1
                            else:
                                # No afectan las heurísticas
                                for rgb_str, pct in dict_colors.items():
                                    dict_image["img_color_{}_rgb".format(i)] = rgb_str
                                    dict_image["pct_color_{}".format(i)] = pct
                                    matcolor = self.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                                    lkcolor, dist_lk = self.get_most_similar_LK_color([int(x) for x in rgb_str.split("_")])
                                    lkcolor_heuristic = lkcolor

                                    if lkcolor == "C1" and dist_lk < 0.5 and pct > 0.35 and color != "C1":
                                        lkcolor_heuristic = color

                                    # We assign the percentage of the color to the dictionary
                                    color_distribution_matplotlib[matcolor] += pct
                                    color_distribution_lk[lkcolor] += pct
                                    color_distribution_lk_heuristic[lkcolor_heuristic] += pct
                                    i += 1

                            # Ordenamos los colores por porcentaje
                            dict_info_lk = dict()
                            dict_info_lk["group_color"] = group_color
                            dict_info_lk_heuristic = dict()
                            dict_info_lk_heuristic["group_color"] = group_color
                            dict_info_mat = dict()
                            dict_info_mat["group_color"] = group_color

                            i = 1
                            for clr, pct in sorted(color_distribution_lk.items(), key=lambda x: x[1],
                                                   reverse=True):
                                if pct > 0.:
                                    dict_info_lk["lk_color_{}".format(i)] = clr
                                    dict_info_lk["pct_color_{}".format(i)] = pct
                                    i += 1
                                else:
                                    break
                            i = 1
                            for clr, pct in sorted(color_distribution_lk_heuristic.items(), key=lambda x: x[1],
                                                   reverse=True):
                                if pct > 0.:
                                    dict_info_lk_heuristic["lk_color_{}".format(i)] = clr
                                    dict_info_lk_heuristic["pct_color_{}".format(i)] = pct
                                    i += 1
                                else:
                                    break
                            i = 1
                            for clr, pct in sorted(color_distribution_matplotlib.items(), key=lambda x: x[1],
                                                   reverse=True):
                                if pct > 0.:
                                    dict_info_mat["lk_color_{}".format(i)] = clr
                                    dict_info_mat["pct_color_{}".format(i)] = pct
                                    i += 1
                                else:
                                    break
                            color_distribution_lk['group_color'] = group_color
                            color_distribution_lk_heuristic['group_color'] = group_color
                            color_distribution_matplotlib['group_color'] = group_color
                            list_gc_info_lk.append(dict_info_lk)
                            list_gc_info_lk_heuristic.append(dict_info_lk_heuristic)
                            list_gc_info_matplotlib.append(dict_info_mat)
                            list_gc_dict_lk.append(color_distribution_lk)
                            list_gc_dict_lk_heuristic.append(color_distribution_lk_heuristic)
                            list_gc_dict_matplotlib.append(color_distribution_matplotlib)
                            list_gc_image_color_detected.append(dict_image)
                        except Exception as error:
                            logger.log(f'Error in computing {group_color}', exc_info=True, level='error')
                            logger.log(error, exc_info=True, level='error')
                            list_not_found_groupcolors.append({'group_color': group_color})
                            continue
            else:
                path_images = '/var/lib/lookiero/images'
                regex = '(\D+\d+)(C\d+)(?:.jpg)'
                n_images_comp = 0
                tot_images = len(os.listdir(path_images))
                for file in os.listdir(path_images):
                    if file.endswith('.jpg'):
                        try:
                            group_color = "_".join(re.findall(regex, file)[0])
                            group, color = group_color.split("_")
                            try:
                                if n_images_comp % 1500 == 0:
                                    pct = n_images_comp*100 // tot_images  # percentage of the computation done
                                    logger.log(f"Percentage of the images computed: {int(pct)} %")
                            except:
                                pass

                            if group_color not in list_computed_gc and group_color not in list_not_found_gc:
                                try:
                                    image = cv2.imread(os.path.join(path_images, file))
                                    dict_colors = self.extract_colors_from_image(image, group_color, tolerance=12)
                                    color_distribution_lk = {color_name: 0. for color_name in
                                                             sorted(self.dict_LK_colors.keys())}
                                    color_distribution_lk_heuristic = {color_name: 0. for color_name in
                                                             sorted(self.dict_LK_colors.keys())}
                                    color_distribution_matplotlib = {color_name: 0. for color_name in
                                                                     sorted(colors.cnames.keys())}

                                    dict_image = dict()
                                    dict_image["group_color"] = group_color

                                    # Find the most similar colors to the ones extracted
                                    i = 1
                                    # Match each detected color to a similar one in the database
                                    if color in self.dict_heuristics:
                                        # Las heurísticas pueden aplicar
                                        for rgb_str, pct in dict_colors.items():
                                            dict_image["img_color_{}_rgb".format(i)] = rgb_str
                                            dict_image["pct_color_{}".format(i)] = pct
                                            matcolor = self.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                                            lkcolor, dist_lk = self.get_most_similar_LK_color(
                                                [int(x) for x in rgb_str.split("_")])

                                            # Comprobamos si se cumplen las heuristicas
                                            if lkcolor in self.dict_heuristics[color]["values"]:
                                                if self.dict_heuristics[color]["exception"] and \
                                                        dist_lk <= self.dict_heuristics[color]["dist_exception"]:
                                                    lkcolor_heuristic = lkcolor
                                                else:
                                                    lkcolor_heuristic = color
                                            else:
                                                lkcolor_heuristic = lkcolor

                                            if lkcolor == "C1" and dist_lk < 0.5 and pct > 0.35 and color != "C1":
                                                lkcolor_heuristic = color

                                            # We assign the percentage of the color to the dictionary
                                            color_distribution_matplotlib[matcolor] += pct
                                            color_distribution_lk[lkcolor] += pct
                                            color_distribution_lk_heuristic[lkcolor_heuristic] += pct
                                            i += 1
                                    else:
                                        # No afectan las heurísticas
                                        for rgb_str, pct in dict_colors.items():
                                            dict_image["img_color_{}_rgb".format(i)] = rgb_str
                                            dict_image["pct_color_{}".format(i)] = pct
                                            matcolor = self.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                                            lkcolor, dist_lk = self.get_most_similar_LK_color(
                                                [int(x) for x in rgb_str.split("_")])
                                            lkcolor_heuristic = lkcolor

                                            if lkcolor == "C1" and dist_lk < 0.5 and pct > 0.35 and color != "C1":
                                                lkcolor_heuristic = color

                                            # We assign the percentage of the color to the dictionary
                                            color_distribution_matplotlib[matcolor] += pct
                                            color_distribution_lk[lkcolor] += pct
                                            color_distribution_lk_heuristic[lkcolor_heuristic] += pct
                                            i += 1

                                    # Ordenamos los colores por porcentaje
                                    dict_info_lk = dict()
                                    dict_info_lk["group_color"] = group_color
                                    dict_info_lk_heuristic = dict()
                                    dict_info_lk_heuristic["group_color"] = group_color
                                    dict_info_mat = dict()
                                    dict_info_mat["group_color"] = group_color

                                    i = 1
                                    for clr, pct in sorted(color_distribution_lk.items(), key=lambda x: x[1],
                                                           reverse=True):
                                        if pct > 0.:
                                            dict_info_lk["lk_color_{}".format(i)] = clr
                                            dict_info_lk["pct_color_{}".format(i)] = pct
                                            i += 1
                                        else:
                                            break
                                    i = 1
                                    for clr, pct in sorted(color_distribution_lk_heuristic.items(),
                                                           key=lambda x: x[1],
                                                           reverse=True):
                                        if pct > 0.:
                                            dict_info_lk_heuristic["lk_color_{}".format(i)] = clr
                                            dict_info_lk_heuristic["pct_color_{}".format(i)] = pct
                                            i += 1
                                        else:
                                            break
                                    i = 1
                                    for clr, pct in sorted(color_distribution_matplotlib.items(),
                                                           key=lambda x: x[1],
                                                           reverse=True):
                                        if pct > 0.:
                                            dict_info_mat["lk_color_{}".format(i)] = clr
                                            dict_info_mat["pct_color_{}".format(i)] = pct
                                            i += 1
                                        else:
                                            break
                                    color_distribution_lk['group_color'] = group_color
                                    color_distribution_lk_heuristic['group_color'] = group_color
                                    color_distribution_matplotlib['group_color'] = group_color
                                    list_gc_info_lk.append(dict_info_lk)
                                    list_gc_info_lk_heuristic.append(dict_info_lk_heuristic)
                                    list_gc_info_matplotlib.append(dict_info_mat)
                                    list_gc_dict_lk.append(color_distribution_lk)
                                    list_gc_dict_lk_heuristic.append(color_distribution_lk_heuristic)
                                    list_gc_dict_matplotlib.append(color_distribution_matplotlib)
                                    list_gc_image_color_detected.append(dict_image)
                                except Exception as error:
                                    logger.log(f'Error computing {group_color}', exc_info=True, level='error')
                                    logger.log(error, exc_info=True, level='error')
                                    list_not_found_groupcolors.append({'group_color': group_color})
                            n_images_comp += 1
                        except Exception as error:
                            logger.log(f'Error computing file {file}', exc_info=True, level='error')
                            logger.log(error, exc_info=True, level='error')
                            n_images_comp += 1
                            continue
                logger.log(f"Percentage of the images computed: 100 %")

            logger.log("Saving data...")
            if list_gc_dict_matplotlib:
                new_gc_dist_df_mat = pd.DataFrame(list_gc_dict_matplotlib)
                new_gc_dist_df_mat.set_index('group_color', inplace=True)
                color_decomp_df_mat = pd.concat([color_decomp_df_mat, new_gc_dist_df_mat])
                color_decomp_df_mat.to_csv(color_config.color_decomposition_filename_matplotlib)
            if list_gc_dict_lk:
                new_gc_dist_df_lk = pd.DataFrame(list_gc_dict_lk)
                new_gc_dist_df_lk.set_index('group_color', inplace=True)
                color_decomp_df_lk = pd.concat([color_decomp_df_lk, new_gc_dist_df_lk])
                color_decomp_df_lk.to_csv(color_config.color_decomposition_filename_lk)
            if list_gc_dict_lk_heuristic:
                new_gc_dist_df_lk_heuristic = pd.DataFrame(list_gc_dict_lk_heuristic)
                new_gc_dist_df_lk_heuristic.set_index('group_color', inplace=True)
                color_decomp_df_lk_heuristic = pd.concat([color_decomp_df_lk_heuristic, new_gc_dist_df_lk_heuristic])
                color_decomp_df_lk_heuristic.to_csv(color_config.color_decomposition_filename_lk_heuristic)
            if list_gc_info_lk:
                new_gc_info_lk_df = pd.DataFrame(list_gc_info_lk)
                new_gc_info_lk_df.set_index('group_color', inplace=True)
                color_info_lk_df = pd.concat([color_info_lk_df, new_gc_info_lk_df])
                color_info_lk_df.to_csv(color_config.color_info_filename_lk)
            if list_gc_info_lk_heuristic:
                new_gc_info_lk_heuristic_df = pd.DataFrame(list_gc_info_lk_heuristic)
                new_gc_info_lk_heuristic_df.set_index('group_color', inplace=True)
                color_info_lk_heuristic_df = pd.concat([color_info_lk_heuristic_df, new_gc_info_lk_heuristic_df])
                color_info_lk_heuristic_df.to_csv(color_config.color_info_filename_lk_heuristic)
            if list_gc_info_matplotlib:
                new_gc_info_matplotlib_df = pd.DataFrame(list_gc_info_matplotlib)
                new_gc_info_matplotlib_df.set_index('group_color', inplace=True)
                color_info_matplotlib_df = pd.concat([color_info_matplotlib_df, new_gc_info_matplotlib_df])
                color_info_matplotlib_df.to_csv(color_config.color_info_filename_matplotlib)
            if list_gc_image_color_detected:
                new_gc_image_color_detected_df = pd.DataFrame(list_gc_image_color_detected)
                new_gc_image_color_detected_df.set_index('group_color', inplace=True)
                color_detected_df = pd.concat([color_detected_df, new_gc_image_color_detected_df])
                color_detected_df.to_csv(color_config.color_detected_filename)
            if list_not_found_groupcolors:
                not_found_gc_df = pd.concat([not_color_decomp_df, pd.DataFrame(list_not_found_groupcolors)])
                not_found_gc_df.to_csv(color_config.not_color_decomposition_filename, index=False)

            logger.log("Done")

            return True

        except Exception as error:
            logger.log('Error in LK_images_info', exc_info=True, level='error')
            logger.log(error, exc_info=True, level='error')
            return False

    def extract_colors_from_image(self, image, group_color, tolerance=10):
        """
        Extracts the colors and their percentage in a given image.
        :return: dictionary where the keys are the color RGB representation and the values are their percentage in the image.
        """
        try:
            # We find the mask that will allow us to identify the pixels related to the apparel
            mask = find_mask(image)

            # We make sure to only keep those pixels that belong to the apparel and not the background
            masked_image = image[np.where(mask > 10)]

            # The masked image is not a rectangular image anymore, so we reshape it to a one row image
            masked_image = masked_image.reshape(masked_image.shape[0], 1, 3)

            # CV2 works in BGR, but PIL expects RGB, so we convert the image to RGB format
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

            # extcolors expects a PIL image, so we transform the masked image to PIL
            masked_image_pil = PIL.Image.fromarray(np.uint8(masked_image))

            # We extract the colors belonging to the garment
            colors_img, pixel_count = extcolors.extract_from_image(masked_image_pil, tolerance=tolerance)

            # The colors are in RGB format and the second element in the tuple is the num. of pixels of that color
            if len(colors_img) > 0:
                dict_image_colors = {"_".join([str(c) for c in k]): np.round(v / pixel_count, 2) for k, v in colors_img
                                     if np.round(v / pixel_count, 2) >= cfg.threshold_min_pct}
                sum_pct = np.round(sum(dict_image_colors.values()), 2)
                if sum_pct < 0.99:
                    # We normalize the percentages of the colors extracted
                    dict_image_colors = {k: np.round(v / sum_pct, 2) for k, v in dict_image_colors.items()}
            else:
                return None

            sum_pct = np.round(sum(dict_image_colors.values()), 2)
            dict_image_colors = {k: v for k, v in sorted(dict_image_colors.items(), key=lambda x: x[1], reverse=True)}
            if sum_pct == 0.99:
                dict_image_colors[list(dict_image_colors.keys())[0]] += 0.01  # add 0.01 to make the 100 pct
            return dict_image_colors
        except Exception as error:
            logger.log(f'Error extracting the colors of {group_color}', exc_info=True, level='error')
            logger.log(error, exc_info=True, level='error')
            return None

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
            logger.log('Error in get_LK_color', exc_info=True, level='error')
            logger.log(error, exc_info=True, level='error')

    def get_LK_color_data(self):
        """ Method to extract all the available colors in LK with their numeric information."""
        try:
            list_colors = []
            for color_name in sorted(self.lk_colors_rgb.keys()):
                dict_colors = {"Name": color_name}

                red_st, green_st, blue_st = self.dict_LK_colors[color_name]
                red, green, blue = self.lk_colors_rgb[color_name]
                l, a, b = rgb2lab([[[red_st, green_st, blue_st]]])[0][0]
                y, u, v = self.transform_rgb_to_yuv([red_st, green_st, blue_st])

                dict_colors["Red"] = red
                dict_colors["Green"] = green
                dict_colors["Blue"] = blue

                dict_colors["Red_st"] = red_st
                dict_colors["Green_st"] = green_st
                dict_colors["Blue_st"] = blue_st

                dict_colors["L"] = l
                dict_colors["A"] = a
                dict_colors["B"] = b

                dict_colors["Y"] = y
                dict_colors["U"] = u
                dict_colors["V"] = v

                list_colors.append(dict_colors)

            self.data_LK_colors = pd.DataFrame(list_colors)
        except Exception as error:
            logger.log('Error in get_LK_color_data', exc_info=True, level='error')
            logger.log(error, exc_info=True, level='error')

    def get_matplotlib_color_data(self):
        """ Method to extract all the available colors in Matplotlib with their information."""
        try:
            list_colors = []
            for color_name in sorted(colors.cnames.keys()):
                dict_matplotlib_color = dict()
                dict_matplotlib_color["Name"] = color_name

                red_st, green_st, blue_st = self.get_rgb_standard(color_name)
                red, green, blue = self.standardize_rgb_inverse([red_st, green_st, blue_st])
                l, a, b = rgb2lab([[[red_st, green_st, blue_st]]])[0][0]
                y, u, v = self.transform_rgb_to_yuv([red_st, green_st, blue_st])

                dict_matplotlib_color["Red"] = red
                dict_matplotlib_color["Green"] = green
                dict_matplotlib_color["Blue"] = blue

                dict_matplotlib_color["Red_st"] = colors.to_rgb(color_name)[0]
                dict_matplotlib_color["Green_st"] = colors.to_rgb(color_name)[1]
                dict_matplotlib_color["Blue_st"] = colors.to_rgb(color_name)[2]

                dict_matplotlib_color["L"] = l
                dict_matplotlib_color["A"] = a
                dict_matplotlib_color["B"] = b

                dict_matplotlib_color["Y"] = y
                dict_matplotlib_color["U"] = u
                dict_matplotlib_color["V"] = v

                list_colors.append(dict_matplotlib_color)

            self.data_matplotlib_colors = pd.DataFrame(list_colors)
        except Exception as error:
            logger.log('Error in get_matplotlib_color_data', exc_info=True, level='error')
            logger.log(error, exc_info=True, level='error')

    def get_most_similar_color(self, query_color: list):
        """ Method to obtain the most similar Matplotlib color from a rgb list """
        red_st, green_st, blue_st = self.standardize_rgb(query_color)
        l, a, b = rgb2lab([[[red_st, green_st, blue_st]]])[0][0]
        # qué es argumento kL? se pueden extraer los colores directamente en LAB?
        distances = deltaE_ciede2000([l, a, b], self.data_matplotlib_colors[["L", "A", "B"]].to_numpy())

        idx_min = np.argmin(distances)
        similar_color = self.data_matplotlib_colors.iloc[idx_min].values
        return similar_color[0]

    def get_most_similar_LK_color(self, query_color: list):
        """ Method to obtain the most similar LK color and its distance from a rgb list """
        red_st, green_st, blue_st = self.standardize_rgb(query_color)
        l, a, b = rgb2lab([[[red_st, green_st, blue_st]]])[0][0]
        # qué es argumento kL? se pueden extraer los colores directamente en LAB?
        distances = deltaE_ciede2000([l, a, b], self.data_LK_colors[["L", "A", "B"]].to_numpy())
        # distances = deltaE_cie76([red_st, green_st, blue_st],
        #                          self.data_LK_colors[["Red_st", "Green_st", "Blue_st"]].to_numpy())

        idx_min = np.argmin(distances)
        # options = [(self.data_LK_colors.iloc[i[0]].values[0], i[1]) for i in sorted(enumerate(distances),
        # key=lambda x: x[1])][:5]
        similar_color = self.data_LK_colors.iloc[idx_min].values
        return similar_color[0], distances[idx_min]


if __name__ == "__main__":
    cem = ColorExtractionwithMask(local=True)
    # df_gc_color_distributions = cem.get_LK_images_info('2020-10-01')
    # Get the rgb decomposition of Lookiero colors
    cem.get_LK_color()
    # Get the yuv, standardize data of Lookiero colors
    cem.get_LK_color_data()
    cem.get_matplotlib_color_data()
    list_group_colors = ['P1103_C22', 'K866_C15', 'Y169_C22', 'K653_C7', 'K915_C30', 'J628_C24']
    for group_color in list_group_colors:
        group, color = group_color.split("_")
        groupcolor = "".join([group, color])
        url = f"https://s3-eu-west-1.amazonaws.com/catalogo.labs/{group}/{groupcolor}.jpg"
        image = cem.get_image_from_s3(group, color)
        dict_colors = cem.extract_colors_from_image(image, group_color, tolerance=12)
        color_distribution_lk = {color_name: 0. for color_name in sorted(cem.dict_LK_colors.keys())}
        color_distribution_lk_heuristic = {color_name: 0. for color_name in sorted(cem.dict_LK_colors.keys())}
        color_distribution_matplotlib = {color_name: 0. for color_name in colors.cnames.keys()}
        dict_image = dict()
        dict_image["group_color"] = group_color

        # Find the most similar colors to the ones extracted
        j = 1
        # Match each detected color to a similar one in the database
        if color in cem.dict_heuristics:
            # Las heurísticas pueden aplicar
            for rgb_str, pct in dict_colors.items():
                dict_image["img_color_{}_rgb".format(j)] = rgb_str
                dict_image["pct_color_{}".format(j)] = pct
                matcolor = cem.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                lkcolor, dist_lk = cem.get_most_similar_LK_color(
                    [int(x) for x in rgb_str.split("_")])

                # Comprobamos si se cumplen las heuristicas
                if lkcolor in cem.dict_heuristics[color]["values"]:
                    if cem.dict_heuristics[color]["exception"] and \
                            dist_lk <= cem.dict_heuristics[color]["dist_exception"]:
                        lkcolor_heuristic = lkcolor
                    else:
                        lkcolor_heuristic = color
                else:
                    lkcolor_heuristic = lkcolor

                if lkcolor == "C1" and dist_lk < 0.5 and pct > 0.35 and color != "C1":
                    lkcolor_heuristic = color

                # We assign the percentage of the color to the dictionary
                color_distribution_matplotlib[matcolor] += pct
                color_distribution_lk[lkcolor] += pct
                color_distribution_lk_heuristic[lkcolor_heuristic] += pct
                j += 1
        else:
            # No afectan las heurísticas
            for rgb_str, pct in dict_colors.items():
                dict_image["img_color_{}_rgb".format(j)] = rgb_str
                dict_image["pct_color_{}".format(j)] = pct
                matcolor = cem.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                lkcolor, dist_lk = cem.get_most_similar_LK_color(
                    [int(x) for x in rgb_str.split("_")])
                lkcolor_heuristic = lkcolor

                if lkcolor == "C1" and dist_lk < 0.5 and pct > 0.35 and color != "C1":
                    lkcolor_heuristic = color

                # We assign the percentage of the color to the dictionary
                color_distribution_matplotlib[matcolor] += pct
                color_distribution_lk[lkcolor] += pct
                color_distribution_lk_heuristic[lkcolor_heuristic] += pct
                j += 1

        # Ordenamos los colores por porcentaje
        dict_info_lk = dict()
        dict_info_lk["group_color"] = group_color
        dict_info_lk_heuristic = dict()
        dict_info_lk_heuristic["group_color"] = group_color
        dict_info_mat = dict()
        dict_info_mat["group_color"] = group_color

        j = 1
        for clr, pct in sorted(color_distribution_lk.items(), key=lambda x: x[1],
                               reverse=True):
            if pct > 0.:
                dict_info_lk["lk_color_{}".format(j)] = clr
                dict_info_lk["pct_color_{}".format(j)] = pct
                j += 1
            else:
                break
        j = 1
        for clr, pct in sorted(color_distribution_lk_heuristic.items(),
                               key=lambda x: x[1],
                               reverse=True):
            if pct > 0.:
                dict_info_lk_heuristic["lk_color_{}".format(j)] = clr
                dict_info_lk_heuristic["pct_color_{}".format(j)] = pct
                j += 1
            else:
                break
        j = 1
        for clr, pct in sorted(color_distribution_matplotlib.items(),
                               key=lambda x: x[1],
                               reverse=True):
            if pct > 0.:
                dict_info_mat["lk_color_{}".format(j)] = clr
                dict_info_mat["pct_color_{}".format(j)] = pct
                j += 1
            else:
                break
        color_distribution_lk['group_color'] = group_color
        color_distribution_lk_heuristic['group_color'] = group_color
        color_distribution_matplotlib['group_color'] = group_color
        print(f"{group_color} done.")
    print("Done")
