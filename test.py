import torch
import dill as pkl
import numpy as np

# Loading trainable parameters of a model (the module must already be defined)
CNN = torch.load("best_model.pt")

targets_path = 'challenge_targets.pkl'

with open(targets_path, "rb") as tfh:
    targets = pkl.load(tfh)
print(targets[1])
print('--------------------------------------------------------')

prediction_path = 'challenge_predictions.pkl'

with open(prediction_path, "rb") as tfh:
    predictions = pkl.load(tfh)
print(predictions[1])
result = []
for output, target in zip(predictions, targets):
    mse = np.mean((target - output) ** 2)
    result.append(mse)

# print the mean value:
print(sum(result)/250)