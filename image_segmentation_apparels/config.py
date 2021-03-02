import numpy as np
white_threshold = 0.7
white_second_threshold = 0.3

# == Parameters =======================================================================
BLUR = 21  # 81 or 91: this has to be an odd number
CANNY_THRESH_1 = 1  # 1 or 10
CANNY_THRESH_2 = 3  # 12 or 32
MASK_DILATE_ITER = 16  # 32
MASK_ERODE_ITER = 21  # 38 or 30
MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format


kernel_right = np.array([[0, 1, 1],
                         [0, 1, 1],
                         [0, 1, 1]], dtype=np.uint8)

kernel_left = np.array([[1, 1, 0],
                        [1, 1, 0],
                        [1, 1, 0]], dtype=np.uint8)

kernel_up = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [0, 0, 0]], dtype=np.uint8)

kernel_down = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.uint8)

kernel_up_right = np.array([[1, 1, 1],
                            [0, 1, 1],
                            [0, 0, 1]], dtype=np.uint8)

kernel_up_left = np.array([[1, 1, 1],
                           [1, 1, 0],
                           [1, 0, 0]], dtype=np.uint8)

kernel_down_right = np.array([[0, 0, 1],
                              [0, 1, 1],
                              [1, 1, 1]], dtype=np.uint8)

kernel_down_left = np.array([[1, 0, 0],
                             [1, 1, 0],
                             [1, 1, 1]], dtype=np.uint8)

kernel_down_left_2 = np.array([[1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 1]], dtype=np.uint8)
kernel_down_right_2 = np.array([[0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 1],
                                [1, 1, 1, 1, 1]], dtype=np.uint8)
kernel_up_left_2 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0]], dtype=np.uint8)
kernel_up_right_2 = np.array([[1, 1, 1, 1, 1],
                              [0, 0, 0, 1, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1]], dtype=np.uint8)

list_filter_kernels = [kernel_right, kernel_left, kernel_up, kernel_down, kernel_up_left,
                       kernel_down_left, kernel_up_right, kernel_down_right]

# list_alternative_filters = [kernel_up_left_2, kernel_down_left_2, kernel_up_right_2, kernel_down_right_2]
