from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from operator import itemgetter
from PIL import Image
import numpy as np
import torch
import glob
import tqdm
import os


def ex4(image_array, crop_size, crop_center):
    if not isinstance(image_array, np.ndarray):
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
    if (crop_end[0] + 20 > X - 1) or (crop_end[1] + 20 > Y - 1):
        raise ValueError('distance between the rectangle and the border of image_array is less than 20 pixels')
    if (crop_start[0] - 20 < 0) or (crop_start[1] - 20 < 0):
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


def collate_fn(batch_as_list: list):
    # 1: Get the maximum shape within the current mini-batch.
    image_shapes = np.stack([np.array(sample[0].shape) for sample in batch_as_list], axis=-1)
    # instead, we can also use: itemgetter
    # image_shapes = [sample[0].shape for sample in batch_as_list]
    # max_x = max(image_shapes, key=itemgetter(1))[0]
    # max_y = max(image_shapes, key=itemgetter(1))[1]
    max_image_shape = image_shapes.max(axis=-1)

    # 2: initialize a tensor with 0 values and can hold all stacked sequences.
    stacked_images = torch.zeros(size=(len(batch_as_list), *max_image_shape), dtype=torch.float32)
    # 3: write the images into the tensor.
    for i in range(len(batch_as_list)):
        stacked_images[i, :, :batch_as_list[i][0].shape[-2], :batch_as_list[i][0].shape[-1]] \
            = torch.from_numpy(batch_as_list[i][0])

    stacked_targets = [torch.from_numpy(sample[1]) for sample in batch_as_list]
    stacked_filenames = [sample[2] for sample in batch_as_list]
    stacked_means = [sample[3] for sample in batch_as_list]
    stacked_stds = [sample[4] for sample in batch_as_list]
    return stacked_images, stacked_targets, stacked_filenames, stacked_means, stacked_stds


def cropping_info(image_as_array: np.ndarray, rnd_gen: np.random.RandomState):
    image_shape = np.array(image_as_array.shape)
    # The width and height of the cropped-out rectangle have to be odd integers
    # of at least 5 and at most 21 pixels, returns a tuple of size 2.
    crop_size = rnd_gen.choice(np.arange(5, 22, step=2), size=2)
    # The border between the cropped-out rectangle and the borders of the image will be at least 20 pixels,
    # as in exercise 4.
    min_border = 20
    # to detect the border of our crop:
    crop_start = np.asarray(crop_size / 2, dtype=np.int) + min_border
    crop_end = image_shape - np.asarray(crop_size / 2, dtype=np.int) - min_border

    crop_center = (rnd_gen.randint(low=crop_start[0], high=crop_end[0]),
                   rnd_gen.randint(low=crop_start[1], high=crop_end[1]))
    return tuple(crop_size), crop_center


def Normalization(image_array: np.ndarray):
    # from uint8 to float32
    image_array = image_array.astype(np.float32)
    mean = image_array.mean()
    std = image_array.std()
    image_array[:] -= mean
    image_array[:] /= std
    return image_array, mean, std


def De_Normalization(output, target_shape, mean, std, uint8: bool = False):
    output *= std
    output += mean
    if uint8:
        output = torch.clamp(output, min=0, max=255).round()
    return output.reshape(target_shape)


def resize(filepath):
    image = Image.open(filepath)
    # resize image using the interpolation method PIL.Image.LANCZOS.
    image = TF.resize(img=image, size=70, interpolation=Image.LANCZOS)
    # convert the tensor to an array..
    image_array = np.array(image)
    image_array = image_array.astype(np.uint8)
    # at most 100 pixels
    image_array = image_array[:100, :100]
    return image_array


class Dataset_Reader(Dataset):
    def __init__(self, path):
        # a list of the absolute path for every .jpg file in the input path.
        self.image_files = sorted(glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True))
        # next, we need to rescale the images so that, they will have a width and height of
        # at least 70 and at most 100 pixels.
        self.image_arrays = [resize(image_file)
                             for image_file in tqdm.tqdm(self.image_files, desc="Reading images")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, id):
        file_name = self.image_files[id]
        image_array = self.image_arrays[id]
        # print(image_array.shape)
        return file_name, image_array


# here we will apply the augmentation and the cropping to the uploaded images...
class Main_Dataset(Dataset):
    def __init__(self, dataset: Dataset_Reader, fixed: bool = False):
        self.dataset = dataset
        self.fixed = fixed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # from the Dataset_Reader
        file_name, image_array = self.dataset.__getitem__(index)
        if self.fixed:
            rnd_gen = np.random.RandomState(index)
        else:
            # but here, the crop values will be random.
            rnd_gen = np.random.RandomState()
        # get the crop info:
        crop_size, crop_center = cropping_info(image_as_array=image_array, rnd_gen=rnd_gen)

        image_array, crop_array, target_array = \
            ex4(image_array=image_array, crop_size=crop_size, crop_center=crop_center)
        # the next step is to normalize the inputs:
        image_array, mean, std = Normalization(image_array)

        # stack the input-image-array and the cropped array, for feeding the input into NN, we should concatenate
        # the image array and the crop array, resulting an input array of shape (2, X, Y).
        crop_array = crop_array.astype(np.float32)
        inputs = np.stack([image_array, crop_array], axis=0)
        target_array = target_array.astype(np.float32)

        return inputs, target_array, file_name, mean, std
