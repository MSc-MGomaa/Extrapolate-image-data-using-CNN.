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

Evaluation functions: compared models for exercise 5

"""

import os
import dill as pkl
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from datasets import normalize_input, ex4


def finalize_output(flat_output_tensor: torch.Tensor, target_shape, mean, std, pretend_uint8: bool = False):
    """De-normalize output and round/convert to uint8"""
    flat_output_tensor *= std
    flat_output_tensor += mean
    if pretend_uint8:
        flat_output_tensor = torch.clamp(flat_output_tensor, min=0, max=255).round()
    return flat_output_tensor.reshape(target_shape)


def evaluate_model(model, dataloader, device):
    """Evaluate `model` on dataset given by `dataloader`"""
    mse = torch.nn.MSELoss()
    loss = 0.
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            inputs, targets, file_names, means, stds = data
            inputs = inputs.to(device)
            targets = [target.to(device) for target in targets]
            
            # Get outputs for network
            outputs = model(inputs)
            outputs = [finalize_output(output, target_shape=target.shape, mean=mean, std=std, pretend_uint8=True)
                       for output, target, mean, std in zip(outputs, targets, means, stds)]
            
            # Calculate loss, do backward pass, and update
            loss += (torch.stack([mse(output, target) for output, target in zip(outputs, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def make_predictions(model, inputsfilepath: str, predictionsfilepath: str, plotpath: str, device: torch.device):
    """Create predictions for file `inputsfilepath` using `model`; Save output to `predictionsfilepath` and plot it;"""
    with open(inputsfilepath, "rb") as tfh:
        testset = pkl.load(tfh)
    input_arrays = testset['images']
    crop_sizes = testset['crop_sizes']
    crop_centers = testset['crop_centers']
    predictions = []
    crop_arrays = []
    
    with torch.no_grad():
        for sample_i in tqdm.tqdm(range(len(input_arrays)), desc="testest predictions", position=0):
            input_array, crop_array, target_array = ex4(image_array=input_arrays[sample_i],
                                                        crop_size=crop_sizes[sample_i],
                                                        crop_center=crop_centers[sample_i])
            # Normalization of input
            input_array, mean, std = normalize_input(input_array)
            input_array = np.stack([input_array, np.asarray(crop_array, dtype=np.float32)], axis=0)
            input_array = torch.tensor(input_array, device=device)[None]
            
            # Get outputs for network
            output = model(input_array)[0]
            output = finalize_output(output, target_shape=crop_sizes[sample_i], mean=mean, std=std, pretend_uint8=True)
            prediction = np.asarray(output.cpu().numpy(), dtype=np.uint8)
            predictions.append(prediction)
            crop_arrays.append(np.asarray(crop_array, dtype=np.bool))
    
    with open(predictionsfilepath, "wb") as pfh:
        pkl.dump(predictions, file=pfh)
    
    plot(input_arrays, predictions, crop_arrays,
         os.path.join(plotpath, 'testset'))


def plot(input_arrays, predictions, crop_arrays, path):
    """Plotting the testset inputs and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 1)
    for i in tqdm.tqdm(range(len(input_arrays)), desc="plotting testset"):
        imputed = input_arrays[i].copy()
        imputed[crop_arrays[i]] = predictions[i].reshape(-1)
        ax[0].clear()
        ax[0].set_title('input')
        ax[0].imshow(input_arrays[i], cmap=plt.cm.gray, interpolation='none')
        ax[0].set_axis_off()
        ax[1].clear()
        ax[1].set_title('imputed')
        ax[1].imshow(imputed, cmap=plt.cm.gray, interpolation='none')
        ax[1].set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"{i:07d}.png"), dpi=1000)
    del fig
