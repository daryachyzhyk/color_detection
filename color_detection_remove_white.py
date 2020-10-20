from PIL import Image


import os
import colorgram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np









folder_item = '/home/darya/Documents/stuart/color/items'
# folder_item = '/home/darya/Documents/stuart/color/items_zoom'

folder_colors = '/home/darya/Documents/stuart/color/items_colors'
folder_wo_white = '/home/darya/Documents/stuart/color/items_wo_white'

image_list = []
id_image = []
for root, dirs, files in os.walk(folder_item):
    for file in files:
        image_list.append(os.path.join(root, file))
        print(os.path.join(root, file))
        print(file)
        id_image.append(os.path.splitext(file)[0])

# id_img = id_image[0]
#
# img = image_list[0]
n_colors = 15
threshold_pct = 0.05

for id_img, file in zip(id_image, image_list):


    id_img = id_image[9]
    file = image_list[9]
    img_open = Image.open(file)
    img = img_open.convert("RGBA")

    data = img.getdata()

    new_data = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((np.nan, np.nan, np.nan, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(os.path.join(folder_wo_white, id_img + '_wo_white_.png'), "PNG")

    img_wo_white = os.path.join(folder_wo_white, id_img + '_wo_white_.png')

    colors_image = colorgram.extract(img_wo_white, n_colors)

    # aa = colors_image.sort(key=lambda c: c.rgb.h)


    color_theme = []
    proportions = []

    for color in colors_image:
        rgb = color.rgb
        rgb = tuple(rgb)
        color_theme.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255))

        proportions.append(color.proportion)

    image_colors_pct = [x for x in proportions if x > threshold_pct]
    number_colors_real = len(image_colors_pct)
    image_colors = color_theme[:number_colors_real]

    s = pd.Series(1, index=np.arange(number_colors_real))
    s.plot(kind='bar', color=image_colors)

    plt.savefig(os.path.join(folder_colors, id_img + '_colors_' + str(threshold_pct) + '.png'))

