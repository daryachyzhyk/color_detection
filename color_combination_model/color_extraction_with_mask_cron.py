from color_extraction_with_mask import ColorExtractionwithMask


def color_extraction_with_mask_cron(local=True):
    cem = ColorExtractionwithMask(local=local)
    _ = cem.get_LK_images_info(filtered_season=9)


if __name__ == "__main__":
    color_extraction_with_mask_cron(local=False)
    print("Done")
