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
from matplotlib import colors
from skimage.color import deltaE_cie76
from data_core import util

import config as cfg
from color_detection_extcolors import ColorExtraction
from image_segmentation_apparels.main import find_mask

logger = util.LoggingClass('color_detection')

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 2500)


class ColorExtractionwithMask(ColorExtraction):
    def __init__(self, local=True):
        super().__init__(local=local)
        self.local = local
        self.data_matplotlib_colors = None
        self.get_matplotlib_color_data()
        self.list_not_found_groupcolors = []

    def get_image_from_s3(self, group, color):
        """
        Return the main image from S3 for a given group and color.
        """
        group_color = "".join([group, color])
        img_url = 'https://s3-eu-west-1.amazonaws.com/catalogo.labs/{}/{}.jpg'.format(group, group_color)

        try:
            response = requests.get(img_url)
            image_bytes = io.BytesIO(response.content)
            file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # image = PIL.Image.open(image_bytes)
            return img
        except:
            return None

    def get_LK_images_info(self, filtered=None):
        """
        Info: Method to extract all the different group_colors existing in the LK catalog
        :param filtered Refers to the earliest date from which you would like to get clothes from.
        """
        try:
            list_gc_dict = []
            if self.local:
                suffix_query = f" where date_created >= {filtered}" if filtered else ""
                query = f"select distinct `_group_id` as 'group', color from variations{suffix_query};"

                data_group = pd.read_sql_query(query, self.conn_mysql)

                for group, color in tqdm.tqdm(data_group.values):
                    try:
                        group_color = "_".join([group, color])
                        image = self.get_image_from_s3(group, color)
                        dict_colors = self.get_colors_from_image(image)
                        color_distribution = {color_name: 0. for color_name in colors.cnames.keys()}
                        color_distribution['group_color'] = group_color
                        for rgb_str, pct in dict_colors.items():
                            matcolor = self.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                            color_distribution[matcolor] = pct
                        list_gc_dict.append(color_distribution)
                    except:
                        self.list_not_found_groupcolors.append({'group_color': "_".join([group, color])})
                        continue
                return pd.DataFrame(list_gc_dict)
            else:
                path_images = '/var/lib/lookiero/images'
                regex = '(\D+\d+)(C\d+)(?:.jpg)'
                n_images_comp = 0
                pct = 0  # percentage of the computation done
                tot_images = len(os.listdir(path_images))
                for file in os.listdir(path_images):
                    try:
                        if file.endswith('.jpg'):
                            if np.round(n_images_comp / tot_images, 2) != pct:
                                pct = np.round(n_images_comp / tot_images, 2)
                                logger.log(f"Percentage of the images computed: {int(pct*100)} %")

                            group_color = "_".join(re.findall(regex, file)[0])
                            try:
                                image = cv2.imread(os.path.join(path_images, file))
                                dict_colors = self.get_colors_from_image(image)
                                color_distribution = {color_name: 0. for color_name in colors.cnames.keys()}
                                color_distribution['group_color'] = group_color
                                for rgb_str, pct in dict_colors.items():
                                    matcolor = self.get_most_similar_color([int(x) for x in rgb_str.split("_")])
                                    color_distribution[matcolor] = pct
                                list_gc_dict.append(color_distribution)
                            except Exception:
                                self.list_not_found_groupcolors.append({'group_color': group_color})
                                continue
                            n_images_comp += 1
                    except Exception:
                        continue
                return pd.DataFrame(list_gc_dict)
        except Exception:
            return None

    def get_colors_from_image(self, image):
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
            colors_img, pixel_count = extcolors.extract_from_image(masked_image_pil)

            # The colors are in RGB format and the second element in the tuple is the num. of pixels of that color
            if len(colors_img) > 0:
                dict_image_colors = {"_".join([str(c) for c in k]): np.round(v / pixel_count, 2) for k, v in colors_img
                                     if np.round(v / pixel_count, 2) >= cfg.threshold_min_pct}
                sum_pct = np.round(sum(dict_image_colors.values()), 2)
                if sum_pct < 0.99:
                    # We normalize the percentages of the colors extracted
                    dict_image_colors = {k: np.round(v / sum_pct, 2) for k, v in dict_image_colors.items()}
            else:
                dict_image_colors = {"255_255_255": 1}

            return {k: v for k, v in sorted(dict_image_colors.items(), key=lambda x: x[1], reverse=True)}
        except Exception as error:
            # print(error)
            # print("There has been an error removing the background.")
            return None

    def get_matplotlib_color_data(self):
        """ Method to extract all the available colors in Matplotlib with their information."""
        list_colors = []
        for color_name in colors.cnames.keys():
            dict_matplotlib_color = dict()
            dict_matplotlib_color["Name"] = color_name

            red_st, green_st, blue_st = self.get_rgb_standard(color_name)
            red, green, blue = self.standardize_rgb_inverse([red_st, green_st, blue_st])
            y, u, v = self.transform_rgb_to_yuv([red_st, green_st, blue_st])

            dict_matplotlib_color["Red"] = red
            dict_matplotlib_color["Green"] = green
            dict_matplotlib_color["Blue"] = blue

            dict_matplotlib_color["Red_st"] = colors.to_rgb(color_name)[0]
            dict_matplotlib_color["Green_st"] = colors.to_rgb(color_name)[1]
            dict_matplotlib_color["Blue_st"] = colors.to_rgb(color_name)[2]

            dict_matplotlib_color["Y"] = y
            dict_matplotlib_color["U"] = u
            dict_matplotlib_color["V"] = v

            list_colors.append(dict_matplotlib_color)

        self.data_matplotlib_colors = pd.DataFrame(list_colors)

    def get_most_similar_color(self, query_color: list):
        """ Method to obtain the most similar Matplotlib color from a rgb list """
        red_st, green_st, blue_st = self.standardize_rgb(query_color)
        distances = deltaE_cie76([red_st, green_st, blue_st],
                                 self.data_matplotlib_colors[["Red_st", "Green_st", "Blue_st"]].to_numpy())

        idx_min = np.argmin(distances)
        similar_color = self.data_matplotlib_colors.iloc[idx_min].values
        return similar_color[0]  # "_".join([str(x) for x in similar_color[1:4]])


if __name__ == "__main__":
    cem = ColorExtractionwithMask(local=False)
    df_gc_color_distributions = cem.get_LK_images_info('2020-10-01')
    df_gc_color_distributions.to_csv(os.path.join(cfg.path_data, 'LK_gc_matplotlib_distributions.csv'), index=False)
    if cem.list_not_found_groupcolors:
        df_not_found_gc = pd.DataFrame(cem.list_not_found_groupcolors)
        df_not_found_gc.to_csv(os.path.join(cfg.path_data, 'LK_gc_not_found_color_distributions.csv'), index=False)
    print("Done")
