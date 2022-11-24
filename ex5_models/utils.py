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

Utility functions: compared models for exercise 5

"""
import os
import numpy as np
from matplotlib import pyplot as plt


def plot(inputs, targets, predictions, path, update):
    """Plotting the training/validation inputs, targets, and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 3)
    ax[1, 2].remove()
    true_inputs = np.copy(inputs[:, 0])
    predicted_inputs = np.copy(inputs[:, 0])
    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        true_inputs[i].reshape(-1)[np.asarray(inputs[i, 1], dtype=np.bool).reshape(-1)] = target.reshape(-1)
        predicted_inputs[i].reshape(-1)[np.asarray(inputs[i, 1], dtype=np.bool).reshape(-1)] = prediction.reshape(-1)
    for i in range(len(inputs)):
        vmin = min(predicted_inputs[i].min(), true_inputs[i].min())
        vmax = max(predicted_inputs[i].max(), true_inputs[i].max())
        ax[0, 0].clear()
        ax[0, 0].set_title('input')
        ax[0, 0].imshow(inputs[i, 0], cmap=plt.cm.gray, interpolation='none')
        ax[0, 0].set_axis_off()
        ax[0, 1].clear()
        ax[0, 1].set_title('true image')
        ax[0, 1].imshow(true_inputs[i], cmap=plt.cm.gray, interpolation='none')
        ax[0, 1].set_axis_off()
        ax[0, 2].clear()
        ax[0, 2].set_title('imputed')
        ax[0, 2].imshow(predicted_inputs[i], cmap=plt.cm.gray, interpolation='none')  # , cmap=plt.cm.gray, vmin=0, vmax=1
        ax[0, 2].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('target')
        ax[1, 0].imshow(targets[i], cmap=plt.cm.gray, interpolation='none', vmin=vmin, vmax=vmax)
        ax[1, 1].clear()
        ax[1, 1].set_title('prediction')
        ax[1, 1].imshow(predictions[i], cmap=plt.cm.gray, interpolation='none', vmin=vmin, vmax=vmax)
        fig.tight_layout()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=1000)
    del fig

