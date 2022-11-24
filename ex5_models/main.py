# -*- coding: utf-8 -*-
"""main.py

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

Main file: compared models for exercise 5
python version: 3.6
torch version: 1.6.0
torchvision version: 0.7.0

losses example testset:
ModelA: 3051.9613390933964
ModelB 5e4 updates: 738.311957346821
ModelB 1e5 updates: 688.4573640997343

ModelB 5e4 updates:
test loss: 841.1346435546875
validation loss: 866.6614990234375
training loss: 843.42919921875


ModelB 1e5 updates (4.5h@4 cpu threads):
test loss: 798.9235229492188
validation loss: 829.1820068359375
training loss: 803.7723999023438
"""

import os
import numpy as np
import torch
import torch.utils.data
from datasets import ImageDataset, MissingPixels, make_minibatch_collate_fn
from architectures import ModelA, ModelB
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import tqdm
from evaluation import finalize_output, evaluate_model, make_predictions
from utils import plot


def main(results_path: str, architecture: str, network_config: dict, traininset_path: str, testset_path: str,
         n_updates: int = int(1e5), print_stats_at: int = 1e2, plot_at: int = 1e4, validate_at: int = 5e3,
         learningrate: float = 1e-3, weight_decay: float = 1e-5, num_workers: int = 8, batch_size: int = 4,
         device: str = "cpu"):
    
    device = torch.device(device)
    
    # Prepare a path to plot to
    plotpath = os.path.join(results_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)
    
    image_dataset = ImageDataset(data_folder=traininset_path)

    # Split dataset into training, validation, and test set randomly
    trainingset = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset)*(3/5))))
    validationset = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset)*(3/5)),
                                                                             int(len(image_dataset)*(4/5))))
    testset = torch.utils.data.Subset(image_dataset, indices=np.arange(int(len(image_dataset)*(4/5)),
                                                                       len(image_dataset)))
    
    trainset_nondeterministic = MissingPixels(trainingset, deterministic=False)
    trainset_deterministic = MissingPixels(trainingset, deterministic=True)
    valset_deterministic = MissingPixels(validationset, deterministic=True)
    testset_deterministic = MissingPixels(testset, deterministic=True)
    trainloader_nondeterministic = torch.utils.data.DataLoader(trainset_nondeterministic, batch_size=batch_size,
                                                               shuffle=True, num_workers=num_workers,
                                                               collate_fn=make_minibatch_collate_fn)
    trainloader_deterministic = torch.utils.data.DataLoader(trainset_deterministic, batch_size=batch_size,
                                                            shuffle=False, num_workers=num_workers,
                                                            collate_fn=make_minibatch_collate_fn)
    valloader_deterministic = torch.utils.data.DataLoader(valset_deterministic, batch_size=batch_size,
                                                          shuffle=False, num_workers=num_workers,
                                                          collate_fn=make_minibatch_collate_fn)
    testloader_deterministic = torch.utils.data.DataLoader(testset_deterministic, batch_size=batch_size,
                                                           shuffle=False, num_workers=num_workers,
                                                           collate_fn=make_minibatch_collate_fn)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))
    
    # Create Network
    if architecture == "ModelA":
        net = ModelA(**network_config)
    elif architecture == "ModelB":
        net = ModelB(**network_config)
    else:
        raise NotImplementedError(f"Architecture {architecture} unknown")
    net.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer
    if len(list(net.parameters())):
        optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    update = 0  # current update counter
    best_validation_loss = val_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    
    # Train until n_updates update have been reached
    while update < n_updates:
        for data in trainloader_nondeterministic:
            inputs, targets, file_names, means, stds = data
            inputs = inputs.to(device)
            targets = [target.to(device) for target in targets]
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Get outputs for network
            outputs = net(inputs)
            outputs = [finalize_output(output, target_shape=target.shape, mean=mean, std=std)
                       for output, target, mean, std in zip(outputs, targets, means, stds)]
            
            # Calculate loss, do backward pass, and update weights
            loss = torch.stack([mse(output, target) for output, target in zip(outputs, targets)]).mean()
            loss.backward()
            optimizer.step()
            
            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)
            
            # Plot output
            if update % plot_at == 0:
                plot(inputs.detach().cpu().numpy(),
                     [(target.detach().cpu().numpy() - mean) / std for target, std, mean in zip(targets, stds, means)],
                     [(output.detach().cpu().numpy() - mean) / std for output, std, mean in zip(outputs, stds, means)],
                     plotpath, update)
            
            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valloader_deterministic, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))
            
            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()
            
            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break
    
    # Save best model for early stopping
    if best_validation_loss > val_loss:
        torch.save(net, os.path.join(results_path, 'best_model.pt'))
    
    update_progess_bar.close()
    print('Finished Training!')
    
    # Load best model and compute score on training, validation, and test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=testloader_deterministic, device=device)
    val_loss = evaluate_model(net, dataloader=valloader_deterministic, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader_deterministic, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as rfh:
        print(f"Scores:", file=rfh)
        print(f"test loss: {test_loss}", file=rfh)
        print(f"validation loss: {val_loss}", file=rfh)
        print(f"training loss: {train_loss}", file=rfh)
        
    # Make predictions for challenge test set
    make_predictions(net, inputsfilepath=testset_path,
                     predictionsfilepath=os.path.join(results_path, "challenge_predictions.pkl"),
                     plotpath=plotpath, device=device)


if __name__ == '__main__':
    torch.set_num_threads(4)
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file
    
    with open(config_file, 'r') as cfh:
        config = json.load(cfh)
    main(**config)
