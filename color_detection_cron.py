
from color_detection_extcolors import ColorExtraction


def color_detection_cron():
    ce = ColorExtraction()
    ce.get_LK_color()
    ce.get_LK_images_info()

if __name__ == "__main__":
    color_detection_cron()