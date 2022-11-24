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

Model architectures: compared models for exercise 5

"""

import numpy as np
import torch


class ModelA(torch.nn.Module):
    """Model A "architecture" used in leaderboard - only computes the mean"""
    def __init__(self):
        super(ModelA, self).__init__()
    
    def forward(self, x):
        image_datas = x[:, :1]
        target_masks = x[:, -1:].to(dtype=torch.bool)
        means = [image_datas[i, torch.logical_not(target_masks[i])].mean() for i in range(len(x))]
        preds = [torch.full_like(image_datas[i, target_masks[i]], fill_value=means[i]) for i in range(len(x))]
        return preds


class ModelB(torch.nn.Module):
    def __init__(self, n_hidden_layers: int = 2, n_kernels: int = 32, kernel_size: int = 7):
        """Model B architecture used in leaderboard - a small CNN"""
        super(ModelB, self).__init__()
        
        cnn = []
        n_in_channels = 2
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size/2)))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1, kernel_size=kernel_size,
                                            bias=True, padding=int(kernel_size/2))
    
    def forward(self, x):
        target_masks = x[:, -1:].to(dtype=torch.bool)
        cnn_outs = self.hidden_layers(x)
        preds = self.output_layer(cnn_outs)
        preds = [preds[i, target_masks[i]] for i in range(len(x))]
        return preds
