from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from Architecture import CNN_Model
from datasets import Dataset_Reader, Main_Dataset, collate_fn, De_Normalization
import numpy as np
from evaluation import Evaluation, make_predictions
import os
import tqdm

########################################################################
# Parameters:
batch_size = 4
num_workers = 0
device = 'cpu'
results_path = 'C:\\Users\\moham\\PycharmProjects\\python2'
update = 0
learning_rate = 1e-4
weight_decay = 1e-5
n_updates = 1e3
print_stats_at = 1e2
validate_at = 1e3

########################################################################
# Load the data from the drive:
input_path = 'Training Data'

My_dataset = Dataset_Reader(input_path)

# split our dataset into training, validation and test set:
# 60% for the training
# [ordered like from 0 to 59 if we have a 100 samples]
training_set = torch.utils.data.Subset(My_dataset, indices=np.arange(int(len(My_dataset) * (3 / 5))))
# 20% validation
validation_set = torch.utils.data.Subset(My_dataset, indices=np.arange(int(len(My_dataset) * (3 / 5)),
                                                                       int(len(My_dataset) * (4 / 5))))
# 20% test.
test_set = torch.utils.data.Subset(My_dataset, indices=np.arange(int(len(My_dataset) * (4 / 5)),
                                                                 len(My_dataset)))

# unit 4: While creating the samples randomly, we use the index as random seed to get (the same sample for the same ID),
# here we will use it to get the same crop values for the same id, and a different crop only for the training samples.
# ex5: if you want to drastically increase your dataset size using
# data augmentation, you can also use random cropping followed by resizing to create more input images:

# Now, from the images, we need to generate the inputs and the outputs.

trainset_augmented = Main_Dataset(training_set, fixed=False)
trainset_fixed = Main_Dataset(training_set, fixed=True)

validation_set_fixed = Main_Dataset(validation_set, fixed=True)
testset_fixed = Main_Dataset(test_set, fixed=True)


trainset_augmented_loader = torch.utils.data.DataLoader(trainset_augmented, batch_size=batch_size, shuffle=True,
                                                        num_workers=num_workers, collate_fn=collate_fn)

trainset_fixed_loader = torch.utils.data.DataLoader(trainset_fixed, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_workers, collate_fn=collate_fn)

validation_set_fixed_loader = torch.utils.data.DataLoader(validation_set_fixed, batch_size=batch_size, shuffle=False,
                                                          num_workers=num_workers, collate_fn=collate_fn)

testset_fixed_loader = torch.utils.data.DataLoader(testset_fixed, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers, collate_fn=collate_fn)

#######################################################################################################################

# Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

# Network:
Cnn = CNN_Model()
Cnn.to(device)

# mean square error as a loss function
mse = torch.nn.MSELoss()

# define adam optimizer
if len(list(Cnn.parameters())):
    optimizer = torch.optim.Adam(Cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_validation_loss = val_loss = np.inf  # initialize with infinity
update_progress_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

torch.save(Cnn, os.path.join(results_path, 'best_model.pt'))

# Train until n_updates = 0
while update < n_updates:
    for data in trainset_augmented_loader:
        # stacked batch
        inputs, targets, file_names, means, stds = data
        inputs = inputs.to(device)
        targets = [target.to(device) for target in targets]

        # Reset gradients
        optimizer.zero_grad()

        # Get outputs for network
        outputs = Cnn(inputs)
        # we need to de_normalize the output and convert it back to the original d-type:
        outputs = [De_Normalization(output, target_shape=target.shape, mean=mean, std=std)
                   for output, target, mean, std in zip(outputs, targets, means, stds)]

        # calculate the mean square error between the output and the real target values.
        # then the final loss will be the mean over the all outputs.
        # then apply the backward and finally update the weights.
        loss = torch.stack([mse(output, target) for output, target in zip(outputs, targets)]).mean()
        loss.backward()
        optimizer.step()

    ####################################################################################################################
        # Print current status and score
        if update % print_stats_at == 0 and update > 0:
            writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

        # Evaluate model on validation set
        if update % validate_at == 0 and update > 0:
            val_loss = Evaluation(Cnn, dataloader=validation_set_fixed_loader, device=device)
            writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
            # Add weights as arrays to tensorboard
            for i, param in enumerate(Cnn.parameters()):
                writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                     global_step=update)
            # Add gradients as arrays to tensorboard
            for i, param in enumerate(Cnn.parameters()):
                writer.add_histogram(tag=f'validation/gradients_{i}',
                                     values=param.grad.cpu(),
                                     global_step=update)
            # Save best model for early stopping, 'best so far!'
            if best_validation_loss > val_loss:
                best_validation_loss = val_loss
                torch.save(Cnn, os.path.join(results_path, 'best_model.pt'))

        update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
        update_progress_bar.update()

        # Increment update counter, exit if maximum number of updates is reached
        update += 1
        if update >= n_updates:
            break

# Save best model for early stopping
if best_validation_loss > val_loss:
    torch.save(Cnn, os.path.join(results_path, 'best_model.pt'))

update_progress_bar.close()
print('Finished Training!')

########################################################################################################################

print(f"Computing scores for best model")
net = torch.load(os.path.join(results_path, 'best_model.pt'))
test_loss = Evaluation(net, dataloader=testset_fixed_loader, device=device)
val_loss = Evaluation(net, dataloader=validation_set_fixed_loader, device=device)
train_loss = Evaluation(net, dataloader=trainset_fixed_loader, device=device)

print(f"Scores:")
print(f"test loss: {test_loss}")
print(f"validation loss: {val_loss}")
print(f"training loss: {train_loss}")

# the results of the best model
with open(os.path.join(results_path, 'result.txt'), 'w') as ob:
    print(f"Scores:", file=ob)
    print(f"test loss: {test_loss}", file=ob)
    print(f"validation loss: {val_loss}", file=ob)
    print(f"training loss: {train_loss}", file=ob)

# predictions

''''
import torch
from evaluation import Evaluation, make_predictions
import os

# Loading trainable parameters of a model (the module must already be defined)
CNN = torch.load("best_model.pt")

testset_path = 'challenge_testset.pkl'
device = 'cpu'
results_path = 'C:\\Users\\moham\\PycharmProjects\\python2'
plotpath = 'C:\\Users\\moham\\PycharmProjects\\python2'

make_predictions(CNN, inputsfilepath=testset_path,
                 predictionsfilepath=os.path.join(results_path, "challenge_predictions.pkl"),
                 device=device)

'''