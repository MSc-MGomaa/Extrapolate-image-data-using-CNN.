import torch
import tqdm
import numpy as np
from datasets import De_Normalization, Normalization, ex4
import dill as pkl


def Evaluation(model, dataloader, device):
    mse = torch.nn.MSELoss()
    loss = 0
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            inputs, targets, file_names, means, stds = data
            inputs = inputs.to(device)
            targets = [target.to(device) for target in targets]
            outputs = model(inputs)
            outputs = [De_Normalization(output, target_shape=target.shape, mean=mean, std=std, uint8=True)
                       for output, target, mean, std in zip(outputs, targets, means, stds)]
            loss += (torch.stack([mse(output, target) for output, target in zip(outputs, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def make_predictions(model, inputsfilepath: str, predictionsfilepath: str, device: torch.device):
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
            input_array, mean, std = Normalization(input_array)
            input_array = np.stack([input_array, np.asarray(crop_array, dtype=np.float32)], axis=0)
            input_array = torch.tensor(input_array, device=device)[None]

            # Get outputs for network
            output = model(input_array)[0]
            output = De_Normalization(output, target_shape=crop_sizes[sample_i], mean=mean, std=std, uint8=True)
            prediction = np.asarray(output.cpu().numpy(), dtype=np.uint8)
            predictions.append(prediction)
            crop_arrays.append(np.asarray(crop_array, dtype=np.bool))

    with open(predictionsfilepath, "wb") as pfh:
        pkl.dump(predictions, file=pfh)