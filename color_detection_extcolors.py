import extcolors
import PIL
import pandas as pd
import numpy as np
import MySQLdb
import psycopg2
# from colors import rgb, hex
from matplotlib import colors
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

        query = "select distinct `_group_id` as 'group', color from variations v {} limit 20;".format("where date_created > '2020-10-01'" if filter else "")
        data_group = pd.read_sql_query(query, self.conn_mysql)
        self.get_LK_color_data()

        list_images = []
        for group, color in data_group.values:
            logger.log("Processing image {}_{}".format(group, color))
            image = self.get_image_from_s3(group, color)
            if image:
                dict_image = self.get_image_color_info(image)
                list_images.append({**{"group": group, "color": color}, **dict_image})
            else:
                logger.log("Image {}_{} couldn't have been loaded.".format(group,color))

        data = pd.DataFrame(list_images)
        data.to_csv(os.path.join(cfg.path_data, "LK_colors_info.csv"), index=False)
        return data

    def get_image_from_s3(self, group, color):
        """
        Return the main image from S3 for a given group and color.
        """

        group_color = "".join([group, color])
        img_url = 'https://s3-eu-west-1.amazonaws.com/catalogo.labs/{}/{}.jpg'.format(group, group_color)

        # http = urllib3.PoolManager()
        # r = http.request('GET', img_url, preload_content=True)
        try:
            response = requests.get(img_url)
            image_bytes = io.BytesIO(response.content)
            image = PIL.Image.open(image_bytes)
            return image
        except:
            return None
        # if r.status != 404:
        #     # with open("/home/jkobe/Downloads/prieba_image.jpg", 'wb') as out:
        #     image = r.read(100)
        #     response = requests.get(img_url)
        #     image_bytes = io.BytesIO(response.content)
        #     img = PIL.Image.open(image_bytes)
        #     if not image:
        #         return None
        #     out.write(image)
        #
        # else:
        #     return None
        #
        # return image

    def get_image_color_info(self, image):
        """
        It returns a dict with the information of the given image. It mainly returns the colors detected and their LK representation.
        """

        dict_similar_colors, dict_similar_colors_pct = self.get_image_representation(image)
        dict_image = {}
        i = 1
        logger.log(dict_similar_colors)
        for color_image, color_lk in dict_similar_colors.items():
            dict_image["img_color_{}_rgb".format(i)] = color_image
            dict_image["lk_color_{}_rgb".format(i)] = color_lk
            dict_image["lk_color_{}".format(i)] = self.lk_colors_rgb_idx[color_lk]
            dict_image["pct_color_{}".format(i)] = dict_similar_colors_pct[color_lk]
            i += 1

        return dict_image

    def get_colors_from_image(self, image, threshold=0.01):
        """
        Extracts the colors and their percentage in a given image. The white color (255, 255, 255) is not taken into acount.
        :return: dictionary where the keys are the color RGB representation and the values are thei percentage in the image.
        """
        colors_img, pixel_count = extcolors.extract_from_image(image)
        white_pixels = [x[1] for x in colors_img if x[0] == (255, 255, 255)][0]
        total_not_white_pixels = pixel_count - white_pixels
        dict_image_colors = {"_".join([str(c) for c in k]): np.round(v / total_not_white_pixels, 2) for k, v in colors_img if k != (255, 255, 255)
                             and np.round(v / total_not_white_pixels, 2) >= threshold}

        return dict_image_colors

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
        y_query, u_query, v_query = self.transform_rgb_to_yuv([red_st, green_st, blue_st])
        distances = np.sum(([y_query, u_query, v_query] - self.data_colors[["Y", "U", "V"]].values)**2, axis=1)

        idx_min = np.argmin(distances)
        similar_color = self.data_colors.iloc[idx_min].values
        return "_".join([str(x) for x in similar_color[1:4]])


    def get_image_representation(self, image):
        """
        Returns a representation of the given image by using the most similar colors of matplotlib.
        """

        dict_colors = self.get_colors_from_image(image)
        dict_similar_colors = {k: self.get_most_similar_color([int(x) for x in k.split("_")]) for k in dict_colors.keys()}
        dict_similar_colors_pct = {self.get_most_similar_color([int(x) for x in k.split("_")]): v for k, v in dict_colors.items()}
        return dict_similar_colors, dict_similar_colors_pct

if __name__ == "__main__":
    group_color = "M117_C32"
    # path_image = "/var/lib/lookiero/images_group_color_1/{}/{}.jpg".format(group_color, "".join(group_color.split("_")))
    # image = PIL.Image.open(path_image)

    ce = ColorExtraction()
    ce.get_LK_color()
    ce.get_LK_images_info()
    # # ce.get_color_data()
    # ce.get_LK_color_data()
