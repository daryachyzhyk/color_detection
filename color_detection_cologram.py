
import os
import colorgram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from matplotlib import colors




# Extract n colors from an image.
path_folder = '/home/darya/Documents/stuart/color'

# image_file_1 = 'D534C1.jpg' # dress
# image_file_1 = 'test.png'

# image_file_1 = 'S931C24.jpg' # blusa coral

image_file_1 = 'J1133C5.jpg' # rallas

# image_file_1 = 'T820C2.jpg' # blusa flores

n_colors = 15

image_path = os.path.join(path_folder, image_file_1)
colors_image = colorgram.extract(image_path, n_colors)


color_theme = []
proportions = []

for color in colors_image:
    rgb = color.rgb
    rgb = tuple(rgb)
    color_theme.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255))

    proportions.append(color.proportion)


# df = pd.DataFrame([[1], [2], [3]])
#
# df.plot.bar()
# barlist[0].set_color(color_theme[0])

s = pd.Series(1, index=np.arange(n_colors))
s.plot(kind='bar', color=color_theme)




#
#
#
# for i, color in enumerate(colors_image):
#     rgb = color.rgb
#     # rgb = colors.colorConverter.to_rgb(color)
#     # rgb_new = make_rgb_transparent(rgb, (1, 1, 1), alpha)
#     print(color, rgb)
#     rgb = tuple(rgb)
#     # rgb = colors.colorConverter.to_rgb(rgb) self.rgb0 = self.rgb[0] / 255, self.rgb[1] / 255, self.rgb[2] / 255
#     # aa = rgb[0] / 255, rgb[1] / 255, rgb[2] / 255
#     color_theme.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255))
#     print('tuple')
#
# # df = pd.DataFrame([[1], [2], [3]])
# #
# # df.plot.bar()
# # barlist[0].set_color(color_theme[0])
#
# s = pd.Series([1, 2, 3])
# s.plot(kind='bar', color=color_theme)
#
#
# color_theme = []
#
#
#
#
#
#
#
#
# # colorgram.extract returns Color objects, which let you access
# # RGB, HSL, and what proportion of the image was that color.
# first_color = colors_image[0]
# rgb = first_color.rgb # e.g. (255, 151, 210)
# hsl = first_color.hsl # e.g. (230, 255, 203)
# proportion  = first_color.proportion # e.g. 0.34
#
# # RGB and HSL are named tuples, so values can be accessed as properties.
# # These all work just as well:
# red = rgb[0]
# red = rgb.r
# saturation = hsl[1]
# saturation = hsl.s
#
# color_theme = []
#
#
# for i, color in enumerate(colors_image):
#     rgb = color.rgb
#     # rgb = colors.colorConverter.to_rgb(color)
#     # rgb_new = make_rgb_transparent(rgb, (1, 1, 1), alpha)
#     print(color, rgb)
#     rgb = tuple(rgb)
#     # rgb = colors.colorConverter.to_rgb(rgb) self.rgb0 = self.rgb[0] / 255, self.rgb[1] / 255, self.rgb[2] / 255
#     # aa = rgb[0] / 255, rgb[1] / 255, rgb[2] / 255
#     color_theme.append((rgb[0] / 255, rgb[1] / 255, rgb[2] / 255))
#     print('tuple')
#
# # df = pd.DataFrame([[1], [2], [3]])
# #
# # df.plot.bar()
# # barlist[0].set_color(color_theme[0])
#
# s = pd.Series([1, 2, 3])
# s.plot(kind='bar', color=color_theme)
#
#
#
# df.plot.bar(color=color_theme)
#
#
#
#     # plt.scatter([i], [0], c=rgb)
#     #
#     # plt.scatter([i], c=rgb)
#
#     # plt.scatter([i], [1], color=color, alpha=alpha, **kwargs)
#     # plt.scatter([i], [2], color=rgb_new, **kwargs)
# plt.scatter([i], [0], c=tuple(rgb))
#
# for i in colors:
#     first_color = colors[i]
#     rgb = first_color.rgb
#
#
#
#
#
#
# plt.imshow([[rgb]])
#
#
# plt.imshow([[(252, 0, 250)]])
#
#
# rgb_red = colors.colorConverter.to_rgb('red')
#
#
#
# # from matplotlib import colors
# import matplotlib.pyplot as plt
#
# alpha = 0.5
#
# kwargs = dict(edgecolors='none', s=3900, marker='s')
# for i, color in enumerate(['red', 'blue', 'green']):
#     rgb = colors.colorConverter.to_rgb(color)
#
#
#
# df = pd.DataFrame([[1], [2], [3]])
# color_theme=[(52/235, 235/235, 86/235), (52/235, 70/235, 235/235),
# (165/235, 52/235, 235/235)]
# df.plot.bar(color=color_theme)
#
# matplotlib.colors.ColorConverter.to_rgb("#ff0000")