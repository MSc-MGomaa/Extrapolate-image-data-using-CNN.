"""
<ex4.py>
Author:<MOHAMMED GOMAA SABER MAHMOUD>
Matr.Nr: <11739013>
"""
import numpy as np


def ex4(image_array, crop_size, crop_center):
    if not isinstance(image_array,np.ndarray):
        raise ValueError('The input is not 2D Array')
    # check if the crop size tuple consists of two objects:
    if len(crop_size) != 2 or len(crop_center) != 2:
        raise ValueError('There must be only two objects here')
    # check if the crop size elements are odd numbers.
    if crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
        raise ValueError('the values in the crop size must be odd numbers')
    # use the crop size tuple():
    value_1 = int(crop_size[0] / 2)
    value_2 = int(crop_size[1] / 2)
    # to detect the border of our crop:
    crop_start = (crop_center[0] - value_1, crop_center[1] - value_2)
    crop_end = (crop_center[0] + value_1, crop_center[1] + value_2)
    X, Y = image_array.shape
    if (crop_end[0]+20 > X-1) or (crop_end[1]+20 > Y-1):
        raise ValueError('distance between the rectangle and the border of image_array is less than 20 pixels')
    if (crop_start[0]-20 < 0) or (crop_start[1]-20 < 0):
        raise ValueError('distance between the rectangle and the border of image_array is less than 20 pixels')
    # 1: the pixels in the cropped-out rectangle are set to 0
    # 2: the crop array:
    crop_array = np.zeros(image_array.shape, image_array.dtype)
    target_list = []
    for y in range(crop_start[0], crop_end[0] + 1):
        for x in range(crop_start[1], crop_end[1] + 1):
            target_list.append(image_array[y][x])
            image_array[y][x] = 0
            crop_array[y][x] = 1
    as_array = np.array(target_list)
    target_array = as_array.reshape(crop_size)

    return image_array, crop_array, target_array
