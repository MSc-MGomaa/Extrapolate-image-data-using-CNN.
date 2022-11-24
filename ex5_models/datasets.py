# -*- coding: utf-8 -*-
"""architectures.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2019

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Data readers: compared models for exercise 5

"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import functional as TF
import tqdm


def ex4(image_array: np.ndarray, crop_size: tuple, crop_center: tuple):
    """See assignment sheet for usage description"""
    if len(crop_size) != 2 or len(crop_center) != 2:
        raise ValueError(f"crop_size and crop_center must have length 2 but have length {len(crop_size)} "
                         f"and {len(crop_center)}")
    if crop_size[0] % 2 == 0 or crop_size[1] % 2 == 0:
        raise ValueError(f"crop_size must only include odd integer values")
    
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise ValueError(f"image_data must be numpy array of shape (H, W)")
    
    crop_x_start = crop_center[0] - int(crop_size[0] / 2)
    crop_x_end = crop_center[0] + int(crop_size[0] / 2)
    n_border_pixels_x = [crop_x_start, image_array.shape[0] - 1 - crop_x_end]
    
    crop_y_start = crop_center[1] - int(crop_size[1] / 2)
    crop_y_end = crop_center[1] + int(crop_size[1] / 2)
    n_border_pixels_y = [crop_y_start, image_array.shape[1] - 1 - crop_y_end]
    
    if min(n_border_pixels_x) < 20 or min(n_border_pixels_y) < 20:
        raise ValueError(f"border for cropped out rectangle should be >= 20 but is {n_border_pixels_x} "
                         f"and {n_border_pixels_y}")
    
    crop_x_end += 1
    crop_y_end += 1
    crop_array = np.zeros_like(image_array)
    crop_array[crop_x_start:crop_x_end, crop_y_start:crop_y_end] = 1
    
    target_array = np.copy(image_array[crop_x_start:crop_x_end, crop_y_start:crop_y_end])
    
    image_array[crop_x_start:crop_x_end, crop_y_start:crop_y_end] = 0
    
    return image_array, crop_array, target_array


def image_to_small_array(filepath):
    """Downscale images"""
    image = Image.open(filepath)
    image = TF.resize(img=image, size=70, interpolation=Image.LANCZOS)
    image_array = np.array(image, dtype=np.uint8)
    image_array = image_array[:100, :100]
    return image_array


def make_crops(image_array: np.ndarray, rnd_gen: np.random.RandomState):
    min_border = 20
    image_shape = np.array(image_array.shape)
    crop_size = rnd_gen.choice(np.arange(5, 21+1, step=2), size=2)
    crop_ind_min = np.asarray(crop_size/2, dtype=np.int) + min_border
    crop_ind_max = image_shape - np.asarray(crop_size/2, dtype=np.int) - min_border
    crop_ind = (rnd_gen.randint(low=crop_ind_min[0], high=crop_ind_max[0]),
                rnd_gen.randint(low=crop_ind_min[1], high=crop_ind_max[1]))
    return tuple(crop_size), crop_ind


def normalize_input(input_array: np.ndarray):
    input_array = np.asarray(input_array, dtype=np.float32)
    mean = input_array.mean()
    std = input_array.std()
    input_array[:] -= mean
    input_array[:] /= std
    return input_array, mean, std


def make_minibatch_collate_fn(batch_as_list: list):
    """Function to be passed to torch.utils.data.DataLoader as collate_fn"""
    image_shapes = np.stack([np.array(sample[0].shape) for sample in batch_as_list], axis=-1)
    max_image_shape = image_shapes.max(axis=-1)
    stacked_images = torch.zeros(size=(len(batch_as_list), *max_image_shape), dtype=torch.float32)
    for i in range(len(batch_as_list)):
        stacked_images[i, :, :batch_as_list[i][0].shape[-2], :batch_as_list[i][0].shape[-1]] \
            = torch.from_numpy(batch_as_list[i][0])
    
    stacked_targets = [torch.from_numpy(sample[1]) for sample in batch_as_list]
    stacked_filenames = [sample[2] for sample in batch_as_list]
    stacked_means = [sample[3] for sample in batch_as_list]
    stacked_stds = [sample[4] for sample in batch_as_list]
    return stacked_images, stacked_targets, stacked_filenames, stacked_means, stacked_stds


class ImageDataset(Dataset):
    def __init__(self, data_folder: str):
        """Dataset for reading images"""
        self.filenames = glob.glob(os.path.join(data_folder, '**', '*.jpg'), recursive=True)
        self.filenames.sort()
        self.image_arrays = [image_to_small_array(image_file)
                             for image_file in tqdm.tqdm(self.filenames, desc="Reading images")]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        image_array = self.image_arrays[idx]
        # image_array = image_to_small_array(file_name)  # Could be done on-the-fly if not enough RAM
        return file_name, image_array


class MissingPixels(Dataset):
    def __init__(self, dataset: ImageDataset, deterministic: bool = False):
        """Dataset for wrapping ImageDataset and applying augmentations and cropping to the images"""
        self.dataset = dataset
        self.deterministic = deterministic

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file_name, image_array = self.dataset.__getitem__(idx)
        if self.deterministic:
            rnd_gen = np.random.RandomState(idx)
        else:
            rnd_gen = np.random.RandomState()
        crop_size, crop_ind = make_crops(image_array=image_array, rnd_gen=rnd_gen)
        input_array, crop_array, target_array = ex4(image_array=image_array, crop_size=crop_size, crop_center=crop_ind)
    
        # Normalization of input
        input_array, mean, std = normalize_input(input_array)
    
        input_array = np.stack([input_array, np.asarray(crop_array, dtype=np.float32)], axis=0)
    
        return input_array, np.asarray(target_array, dtype=np.float32), file_name, mean, std
