import extcolors
import PIL
import pandas as pd
import numpy as np
import MySQLdb
# import mysql.connector as MySQLdb
import psycopg2
# from colors import rgb, hex
from matplotlib import colors
from skimage.color import deltaE_cie76
import requests, io
import os
import config as cfg
from data_core import util

logger = util.LoggingClass('color_detection')

desired_width = 320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 2500)


class ColorExtraction:

    def __init__(self, local=True):
        if local:
            db_mysql = {'database': 'lookiero',
                        'user': 'awsuser',
                        'password': 'awspassword',
                        'host': '127.0.0.1',
                        'port': 3307}
            db_catalog = {'database': 'buying_back',
                        'user': 'buying_back_ro',
                        'password': 'ShuperShekret',
                        'host': '127.0.0.1',
                        'port': 5433}
        else:
            db_mysql = {'database': 'lookiero',
                        'user': 'awsuser',
                        'password': 'awspassword',
                        'host': 'db-data-lake.lookiero.tech',
                        'port': 3306}
            db_catalog = {'host': 'db-buying-back-slave.lookiero.tech',
            # db_catalog = {'host': '127.0.0.1',
                          'port': 5432,
                          # 'port': 5433,
                          'dbname': 'buying_back',
                          'user': 'buying_back_ro',
                          'password': 'ShuperShekret'}

        self.conn_mysql = MySQLdb.connect(**db_mysql)
        self.conn_catalog = psycopg2.connect(**db_catalog)

        if not os.path.exists(cfg.path_data):
            os.makedirs(cfg.path_data)

    def load_extracted_colors(self):
        """
        Method to load data of colors extracted from clothes.
        """
        try:
            return pd.read_csv(os.path.join(cfg.path_data, "LK_colors_info.csv"))
            # return np.array(["_".join([x]) for x in data[["group", "color"]].values])
        except:
            return pd.DataFrame()

    def get_LK_color(self):
        """
        Method to get LK colors and their RGB representation.
        """

        query_colors = "select code, hexadecimal from color;"
        data_colors = pd.read_sql_query(query_colors, self.conn_catalog)

        self.dict_colors = {}
        for row in data_colors.values:
            rgb_color = list(np.round(colors.hex2color("".join(["#", row[1].lower()])), 2))
            self.dict_colors[row[0]] = rgb_color

        self.lk_colors_rgb = {k: [self.standardize_rgb_inverse(v)[0], self.standardize_rgb_inverse(v)[1], self.standardize_rgb_inverse(v)[2]]
                              for k, v in self.dict_colors.items()}
        self.lk_colors_rgb_idx = {"_".join([str(v[0]), str(v[1]), str(v[2])]): k for k, v in self.lk_colors_rgb.items()}

    def get_LK_images_info(self, filter=True):

        query = "select distinct i.`_group_id` as 'group', v.color from variations v, items i where v._group_id=i._group_id" \
                "{};".format(" and v.date_created > '2020-10-01'" if filter else "")
                # " and i.family not in (14, 15, 16, 17, 18, 19, 24, 27, 28, 29, 30, 31){};".format(" and v.date_created > '2020-10-01'" if filter else "")
        data_group = pd.read_sql_query(query, self.conn_mysql)
        self.get_LK_color_data()
        df_extracted_group_colors = self.load_extracted_colors()
        if not df_extracted_group_colors.empty:
            list_extracted_group_colors = np.array(["_".join(x) for x in df_extracted_group_colors[["group", "color"]].values])
        else:
            list_extracted_group_colors = []

        list_images = []
        i = 1
        for group, color in data_group.values:
            group_color = "_".join([group, color])
            if not group_color in list_extracted_group_colors:
                try:
                    logger.log("Processing image {} {}_{}".format(i, group, color))
                    image = self.get_image_from_s3(group, color)
                    if image:
                        dict_image = self.get_image_color_info(image)
                        if dict_image:
                            list_images.append({**{"group": group, "color": color}, **dict_image})
                    else:
                        logger.log("Image {}_{} couldn't have been loaded.".format(group, color))
                    i += 1
                except:
                    logger.log("There has been an error with image {}_{}.".format(group, color))
                    pass

        data = pd.DataFrame(list_images)
        if not df_extracted_group_colors.empty:
            data = pd.concat([df_extracted_group_colors, data])
        data.to_csv(os.path.join(cfg.path_data, "LK_colors_info.csv"), index=False)
        return data

    def get_image_from_s3(self, group, color):
        """
        Return the main image from S3 for a given group and color.
        """

        group_color = "".join([group, color])
        img_url = 'https://s3-eu-west-1.amazonaws.com/catalogo.labs/{}/{}.jpg'.format(group, group_color)

        try:
            response = requests.get(img_url)
            image_bytes = io.BytesIO(response.content)
            image = PIL.Image.open(image_bytes)
            return image
        except:
            return None

    def get_image_color_info(self, image):
        """
        It returns a dict with the information of the given image. It mainly returns the colors detected and their LK representation.
        """

        dict_similar_colors, dict_similar_colors_pct = self.get_image_representation(image)
        if dict_similar_colors:
            dict_image = {}
            i = 1
            for color_image, color_lk in dict_similar_colors.items():
                dict_image["img_color_{}_rgb".format(i)] = color_image
                dict_image["lk_color_{}_rgb".format(i)] = color_lk
                dict_image["lk_color_{}".format(i)] = self.lk_colors_rgb_idx[color_lk]
                dict_image["pct_color_{}".format(i)] = dict_similar_colors_pct[color_lk]
                i += 1

            return dict_image
        else:
            return None

    def get_colors_from_image(self, image):
        """
        Extracts the colors and their percentage in a given image. The white color (255, 255, 255) is not taken into acount.
        :return: dictionary where the keys are the color RGB representation and the values are thei percentage in the image.
        """
        try:
            colors_img, pixel_count = extcolors.extract_from_image(image)
            if len(colors_img) > 1:
                if colors_img[0][0] == (255, 255, 255) and colors_img[0][1]/pixel_count > cfg.white_threshold:
                    total_not_white_pixels = int(pixel_count*(1 - cfg.white_threshold))
                else:
                    white_pixels = [x[1] for x in colors_img if x[0] == (255, 255, 255)][0]
                    total_not_white_pixels = pixel_count - white_pixels
                dict_image_colors = {"_".join([str(c) for c in k]): np.round(v / total_not_white_pixels, 2)
                                     for k, v in colors_img
                                     if (k != (255, 255, 255) and np.round(v / total_not_white_pixels, 2) >= cfg.threshold_min_pct)}
                sum_pct = sum(dict_image_colors.values())
                if sum_pct < 1:
                    dict_image_colors["255_255_255"] = np.round(1 - sum_pct, 2)
            else:
                dict_image_colors = {"255_255_255": 1}

            return {k: v for k, v in sorted(dict_image_colors.items(), key=lambda x: x[1], reverse=True)}
        except:
            logger.log("There has been an error removing the background.")
            return None

    def transform_rgb_to_yuv(self, rgb_std):
        """
        Transform a color from a given RGB standard representation to YUV representation
        :return: YUV representation of the color
        """
        yuv_matrix = np.array([[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.1]])
        return np.matmul(yuv_matrix, [rgb_std[0], rgb_std[1], rgb_std[2]])

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

    def get_color_data(self):
        """ Method to extract all the available colors in Matplotlib with their information."""

        list_colors = []
        for color_name in colors.cnames.keys():
            if color_name not in ["darkgray", "magenta"]:
                dict_colors = {}
                dict_colors["Name"] = color_name

                red_st, green_st, blue_st = self.get_rgb_standard(color_name)
                red, green, blue = self.standardize_rgb_inverse([red_st, green_st, blue_st])
                y, u, v = self.transform_rgb_to_yuv([red_st, green_st, blue_st])

                dict_colors["Red"] = red
                dict_colors["Green"] = green
                dict_colors["Blue"] = blue

                dict_colors["Red_st"] = colors.to_rgb(color_name)[0]
                dict_colors["Green_st"] = colors.to_rgb(color_name)[1]
                dict_colors["Blue_st"] = colors.to_rgb(color_name)[2]

                dict_colors["Y"] = y
                dict_colors["U"] = u
                dict_colors["V"] = v

                list_colors.append(dict_colors)

        self.data_colors = pd.DataFrame(list_colors)

    def get_LK_color_data(self):
        """ Method to extract all the available colors in LK with their numeric information."""

        list_colors = []
        for color_name in self.lk_colors_rgb.keys():
            dict_colors = {}
            dict_colors["Name"] = color_name

            red_st, green_st, blue_st = self.dict_colors[color_name]
            red, green, blue = self.lk_colors_rgb[color_name]
            y, u, v = self.transform_rgb_to_yuv([red_st, green_st, blue_st])

            dict_colors["Red"] = red
            dict_colors["Green"] = green
            dict_colors["Blue"] = blue

            dict_colors["Red_st"] = red_st
            dict_colors["Green_st"] = green_st
            dict_colors["Blue_st"] = blue_st

            dict_colors["Y"] = y
            dict_colors["U"] = u
            dict_colors["V"] = v

            list_colors.append(dict_colors)

        self.data_colors = pd.DataFrame(list_colors)

    def get_most_similar_color(self, query_color):

        red_st, green_st, blue_st = self.standardize_rgb(query_color)
        # y_query, u_query, v_query = self.transform_rgb_to_yuv([red_st, green_st, blue_st])
        # distances = np.sum(([y_query, u_query, v_query] - self.data_colors[["Y", "U", "V"]].to_numpy())**2, axis=1)
        distances = deltaE_cie76([red_st, green_st, blue_st], self.data_colors[["Red_st", "Green_st", "Blue_st"]].to_numpy())


        idx_min = np.argmin(distances)
        similar_color = self.data_colors.iloc[idx_min].values
        return "_".join([str(x) for x in similar_color[1:4]])


    def get_image_representation(self, image):
        """
        Returns a representation of the given image by using the most similar colors of matplotlib.
        """

        dict_colors = self.get_colors_from_image(image)
        if dict_colors:
            dict_similar_colors = {k: self.get_most_similar_color([int(x) for x in k.split("_")]) for k in dict_colors.keys()}
            dict_similar_colors_pct = {self.get_most_similar_color([int(x) for x in k.split("_")]): v for k, v in dict_colors.items()}
            return dict_similar_colors, dict_similar_colors_pct
        else:
            return None, None


if __name__ == "__main__":
    # group_color = "T1236_C2"
    # path_image = "/var/lib/lookiero/images_group_color_1/{}/{}.jpg".format(group_color, "".join(group_color.split("_")))
    # image = PIL.Image.open(path_image)

    ce = ColorExtraction()
    image = ce.get_image_from_s3("S3097", "C1")
    ce.get_LK_color()
    ce.get_LK_color_data()
    # ce.get_LK_images_info()
    dict_similar_colors, dict_similar_colors_pct = ce.get_image_representation(image)
    # ce.get_LK_color_data()
