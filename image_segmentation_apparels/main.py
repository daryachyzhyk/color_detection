import skimage.color as skcolor
import numpy as np
from image_segmentation_apparels import config as cfg
import os
from sklearn.cluster import KMeans
import timeit
import cv2
from data_core import util

logger = util.LoggingClass('image_segmentation')


def find_mask(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_r = gray.reshape(gray.shape[0] * gray.shape[1]) / 255
        gray_apparel = gray_r[gray_r <= 0.975]
        mean_white_apparel = gray_apparel.mean()

        treated_by_region = False
        mask2 = None

        if mean_white_apparel >= cfg.white_threshold:
            # The apparel is white-ish
            mask = canny_edge_detector(gray)
            mask = np.uint8(mask)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), 'uint8'))
            mask = cv2.erode(mask, None, iterations=cfg.MASK_ERODE_ITER)
            mask = cv2.dilate(mask, None, iterations=cfg.MASK_DILATE_ITER)
            mask = mask * 255
            for i in range(150):
                for kernel in cfg.list_filter_kernels:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.GaussianBlur(mask, (cfg.BLUR, cfg.BLUR), 0)
            # for i in range(3):
            #     for kernel in cfg.list_alternative_filters:
            #         mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        elif mean_white_apparel >= cfg.white_second_threshold:
            # The apparel has a darker color
            mask_region = region_based_segmentation(gray, 0.2)
            mask_canny = canny_edge_detector(gray)
            a = mask_region.flatten().sum()
            b = mask_canny.flatten().sum()
            if 1.1 * a >= b:
                treated_by_region = True
                mask = mask_region
                mask = np.uint8(mask)
                # mask2 = mask.copy()  # we will have a second version of the masking
                # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel=np.ones((37, 1), 'uint8'))
                # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 3), 'uint8'))
                mask = mask * 255
                # mask2 = mask2 * 255
                for i in range(150):
                    for kernel in cfg.list_filter_kernels:
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
                mask = cv2.erode(mask, None, iterations=4)
                mask = cv2.dilate(mask, None, iterations=1)
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                # mask2 = cv2.erode(mask2, None, iterations=4)
                # mask2 = cv2.dilate(mask2, None, iterations=1)
                # mask2 = cv2.GaussianBlur(mask2, (3, 3), 0)
            else:
                mask = mask_canny
                mask = np.uint8(mask)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), 'uint8'))
                mask = cv2.dilate(mask, None, iterations=cfg.MASK_DILATE_ITER)
                mask = cv2.erode(mask, None, iterations=cfg.MASK_ERODE_ITER)
                mask = mask * 255
                for i in range(150):
                    for kernel in cfg.list_filter_kernels:
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.GaussianBlur(mask, (cfg.BLUR, cfg.BLUR), 0)
                # for i in range(3):
                #     for kernel in cfg.list_alternative_filters:
                #         mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        else:
            # The apparel has an even darker color
            treated_by_region = True
            mask = region_based_segmentation(gray, 0.78)
            mask = np.uint8(mask)
            mask2 = mask.copy()  # we will have a second version of the masking
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel=np.ones((42, 1), 'uint8'))
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 3), 'uint8'))
            mask = mask * 255
            mask2 = mask2 * 255
            for i in range(150):
                for kernel in cfg.list_filter_kernels:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
            mask = cv2.erode(mask, None, iterations=4)
            mask = cv2.dilate(mask, None, iterations=1)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            mask2 = cv2.erode(mask2, None, iterations=4)
            mask2 = cv2.dilate(mask2, None, iterations=1)
            mask2 = cv2.GaussianBlur(mask2, (3, 3), 0)

        return mask
    except Exception as error:
        # logger.log(error)
        print(error)
        return None


def mask_apparel(image_file):
    """
    Function: Masks the passed image separating the apparel from the background.
    :param image_file: Local path to the image to mask.
    :return: Nothing. The image is saved as a file in disk.
    """
    img = cv2.imread(image_file)
    # TODO: Revisist as I modified and moved part of the code to find_mask() to use it in a different project
    """
    img_r = img.reshape(img.shape[0] * img.shape[1], img.shape[2]) / 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_r_apparel = img_r[np.where(np.all(img_r != 1., axis=1))[0]]
    # np.array([row for row in img_r if not np.array_equal(row, np.array([1.]*img.shape[2]))])
    mean_img_apparel = img_r_apparel.mean(axis=0)

    if np.all(mean_img_apparel >= cfg.white_threshold):
        # The apparel is white-ish
        mask = canny_edge_detector(gray)
    else:
        # The apparel has a darker color
        mask = region_based_segmentation(gray)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = find_mask(img)
    try:
        # -- Smooth mask, then blur it --------------------------------------------------------
        # mask = cv2.dilate(mask, None, iterations=cfg.MASK_DILATE_ITER)
        # mask = cv2.erode(mask, None, iterations=cfg.MASK_ERODE_ITER)
        # mask = cv2.GaussianBlur(mask, (cfg.BLUR, cfg.BLUR), 0)
        # kernel = np.ones((3, 3), 'uint8')

        # for i in range(38):
        #     for kernel in cfg.list_filter_kernels:
        #         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # mask = cv2.dilate(mask, kernel=kernel, iterations=cfg.MASK_DILATE_ITER)
        # mask = cv2.erode(mask, kernel=kernel, iterations=cfg.MASK_ERODE_ITER)
        # mask = cv2.GaussianBlur(mask, (cfg.BLUR, cfg.BLUR), 0)

        # -- Blend masked img into MASK_COLOR background --------------------------------------
        mask = mask / 255
        mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask
        img_red_mask = img.copy().astype('float32') / 255.0  # for easy blending

        img_red_mask = (mask_stack * img_red_mask) + ((1 - mask_stack) * cfg.MASK_COLOR)  # Blend
        img_red_mask = (img_red_mask * 255).astype('uint8')  # Convert back to 8-bit

        # Calculate the proportions for the cropping
        # height, width = np.where(mask > 0)
        height, width = np.where(gray < 255*0.975)
        min_width = np.min(width)
        max_width = np.max(width)
        min_height = np.min(height)
        max_height = np.max(height)

        # Add alpha layer with OpenCV
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # Set alpha layer semi-transparent with Numpy indexing, B=0, G=1, R=2, A=3
        bgra[..., 3] = mask * 255

        img_cropped_trans_bg = bgra[min_height:max_height + 1, min_width:max_width + 1, :]
        img_cropped_white_bg = img[min_height:max_height + 1, min_width:max_width + 1, :]

        directories = image_file.split("/")
        dir_path = "/".join([x for x in directories[:-1]])
        filename = directories[-1]
        filename_red_bg = filename.replace(".jpg", "--masked.jpg")
        filename_trans_bg = filename.replace(".jpg", "--cropped.png")
        filename_white_bg = filename.replace(".jpg", "--cropped-white-bg.jpg")
        file_path_red_bg = os.path.join(dir_path, filename_red_bg)
        file_path_trans_bg = os.path.join(dir_path, filename_trans_bg)
        file_path_white_bg = os.path.join(dir_path, filename_white_bg)
        cv2.imwrite(file_path_red_bg, img_red_mask)  # Save
        cv2.imwrite(file_path_trans_bg, img_cropped_trans_bg)  # Save
        cv2.imwrite(file_path_white_bg, img_cropped_white_bg)  # Save

        if treated_by_region and mask2 is not None:
            mask2 = mask2 / 255
            mask_stack2 = np.dstack([mask2] * 3)  # Create 3-channel alpha mask
            img_red_mask2 = img.copy().astype('float32') / 255.0  # for easy blending

            img_red_mask2 = (mask_stack2 * img_red_mask2) + ((1 - mask_stack2) * cfg.MASK_COLOR)  # Blend
            img_red_mask2 = (img_red_mask2 * 255).astype('uint8')  # Convert back to 8-bit

            # Add alpha layer with OpenCV
            bgra2 = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            # Set alpha layer semi-transparent with Numpy indexing, B=0, G=1, R=2, A=3
            bgra2[..., 3] = mask2 * 255

            img_cropped_trans_bg2 = bgra2[min_height:max_height + 1, min_width:max_width + 1, :]

            filename_red_bg2 = filename.replace(".jpg", "-v2--masked.jpg")
            filename_trans_bg2 = filename.replace(".jpg", "-v2--cropped.png")
            file_path_red_bg2 = os.path.join(dir_path, filename_red_bg2)
            file_path_trans_bg2 = os.path.join(dir_path, filename_trans_bg2)
            cv2.imwrite(file_path_red_bg2, img_red_mask2)  # Save
            cv2.imwrite(file_path_trans_bg2, img_cropped_trans_bg2)  # Save
    except Exception as error:
        # logger.log(error)
        # logger.log(f"The mask for apparel {image_file.split('/')[1]} could not be created.")
        print(f"The mask for apparel {image_file.split('/')[1]} could not be created.")


def kmeans_clustering_segmentation(image):
    """
    Approach: to detect regions based on clusters of colors.
    In this case, we assume three clusters: background, primary and secondary colors of the apparel (if any).
    :param image: Image read with plt.imread()
    :return: Image with the the detected groups painted as the centroids of their cluster
    """
    image_n = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(image_n)
    cluster_img = kmeans.cluster_centers_[kmeans.labels_]
    return cluster_img.reshape(image.shape[0], image.shape[1], image.shape[2])


def pure_white_removal(image):
    """
    Approach: to detect the background assuming it is the only region with pure white color.
    :param image: Image read with plt.imread()
    :return: Image with the detected mask painted with bright luminosity, apparel as dark.
    """
    gray = skcolor.rgb2gray(image)
    gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] == 1.:
            gray_r[i] = 1
        else:
            gray_r[i] = 0

    return gray_r.reshape(gray.shape[0], gray.shape[1])


def region_based_segmentation(gray_img, modificator):
    """
    Approach: to detect regions based on thresholds of luminosity (so far, only two: apparel and background).
    :param gray_img: Gray Image read from cv2.imread().
    :param modificator: Modificator of the threshold.
    :return: Detected mask for the image (0 for background, 1 for apparel).
    """
    gray = gray_img.copy() / 255
    gray_r = gray.reshape(gray_img.shape[0] * gray_img.shape[1])
    mean_img = gray_r.mean()
    threshold = mean_img + modificator * (1. - mean_img)  # threshold of luminosity to separate both regions
    mask = gray < threshold
    mask = mask.astype('int')

    return mask


def canny_edge_detector(gray_img):
    """
    Approach: to detect the background using the canny edge detector.
    :param gray_img: Gray Image read from cv2.imread().
    :return: Detected mask for the image (0 for background, 1 for apparel).
    """
    # == Processing =======================================================================
    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray_img.copy(), cfg.CANNY_THRESH_1, cfg.CANNY_THRESH_2)
    edges = cv2.dilate(edges, None) # todo: to revise
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.drawContours(mask, max_contour, 0, 1, -1)
    # for contour in contour_info:
    #     cv2.fillConvexPoly(mask, contour[0], (255))
    #     cv2.fillConvexPoly(mask, max_contour[0], (255))

    # # -- Smooth mask, then blur it --------------------------------------------------------
    # mask = cv2.dilate(mask, None, iterations=cfg.MASK_DILATE_ITER)
    # mask = cv2.erode(mask, None, iterations=cfg.MASK_ERODE_ITER)
    # mask = cv2.GaussianBlur(mask, (cfg.BLUR, cfg.BLUR), 0)
    # mask = mask.astype('float32') / 255.0  # Use float matrices,

    return mask


if __name__ == '__main__':
    dirs = os.listdir('images')
    n_apparels = 0
    start = timeit.default_timer()
    # mask_apparel(f'images/D418C9.jpg')

    for category in dirs:
        if not category.startswith('.'):
            cat_filename = os.path.join('images', category)
            if os.path.isfile(cat_filename):
                if not cat_filename.endswith('--masked.jpg') and not cat_filename.endswith('--cropped.png') and \
                   not cat_filename.endswith('--cropped-white-bg.jpg'):
                    mask_apparel(f'images/{category}')
                    n_apparels += 1
            if os.path.isdir(cat_filename):
                for cluster in os.listdir(cat_filename):
                    if not cluster.startswith('.'):
                        cluster_filename = os.path.join(cat_filename, cluster)
                        if os.path.isdir(cluster_filename):
                            print(f"Segmenting apparels -- Cat. {category} -- Cluster {cluster}")
                            for image_file in os.listdir(cluster_filename):
                                if not image_file.startswith('.') and not image_file.endswith('--masked.jpg') and \
                                        not image_file.endswith('--cropped.png') and \
                                        not image_file.endswith('--cropped-white-bg.jpg'):
                                    mask_apparel(os.path.join(cluster_filename, image_file))
                                    n_apparels += 1
    stop = timeit.default_timer()
    print("Total running time: ", (stop - start))
    print('Time per apparel: ', (stop - start) / n_apparels)
