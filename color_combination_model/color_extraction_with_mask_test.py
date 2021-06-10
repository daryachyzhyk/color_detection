from color_extraction_with_mask import ColorExtractionwithMask
from matplotlib import colors


def color_extraction_with_mask_cron(local=True):
    cem = ColorExtractionwithMask(local=local)
    # df_gc_color_distributions = cem.get_LK_images_info(filtered_season=9)

    # Get the rgb decomposition of Lookiero colors
    cem.get_LK_color()
    # Get the yuv, standardize data of Lookiero colors
    cem.get_LK_color_data()
    cem.get_matplotlib_color_data()
    list_group_colors = ['Q500_C16', 'Q522_C50']
    for group_color in list_group_colors:
        group, color = group_color.split("_")
        image = cem.get_image_from_s3(group, color)
        if image is None:
            # Image not found
            print(f"Error descargando la imagen {group_color}. Not found or wrong url.")
            continue
        dict_colors = cem.extract_colors_from_image(image, group_color, tolerance=16)
        color_distribution_lk = {color_name: 0. for color_name in sorted(cem.dict_LK_colors.keys())}
        color_distribution_lk_heuristic = {color_name: 0. for color_name in sorted(cem.dict_LK_colors.keys())}
        color_distribution_matplotlib = {color_name: 0. for color_name in colors.cnames.keys()}

        if dict_colors is not None:
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
            # Aplicamos la heurística de C20->C1
            for clr, pct in sorted(color_distribution_lk_heuristic.items(),
                                   key=lambda x: x[1],
                                   reverse=True)[:1]:
                if clr == "C20" and color != "C20":
                    color_distribution_lk_heuristic["C1"] += pct
                    color_distribution_lk_heuristic["C20"] = 0.
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
                    dict_info_mat["matplotlib_color_{}".format(j)] = clr
                    dict_info_mat["pct_color_{}".format(j)] = pct
                    j += 1
                else:
                    break
            dict_info_lk["successful_extraction"] = 1
            dict_info_lk_heuristic["successful_extraction"] = 1
            color_distribution_lk['group_color'] = group_color
            color_distribution_lk_heuristic['group_color'] = group_color
            color_distribution_matplotlib['group_color'] = group_color
            print(f"{group_color}:")
            for color in dict_colors.keys():
                print(color)
            print("done\n")
        else:
            # Extraction failed
            dict_info_lk = dict()
            dict_info_lk["group_color"] = group_color
            dict_info_lk["lk_color_1"] = color
            dict_info_lk["pct_color_1"] = 1.0
            dict_info_lk["lk_color_2"] = color
            dict_info_lk["pct_color_2"] = 1.0
            dict_info_lk["successful_extraction"] = 0

            dict_info_lk_heuristic = dict()
            dict_info_lk_heuristic["group_color"] = group_color
            dict_info_lk_heuristic["lk_color_1"] = color
            dict_info_lk_heuristic["pct_color_1"] = 1.0
            dict_info_lk_heuristic["lk_color_2"] = color
            dict_info_lk_heuristic["pct_color_2"] = 1.0
            dict_info_lk_heuristic["successful_extraction"] = 0

    print("Done")


if __name__ == "__main__":
    color_extraction_with_mask_cron(local=False)