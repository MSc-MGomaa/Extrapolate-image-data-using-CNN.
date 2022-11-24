# Model B uses a CNN consisting of 3 layers (=2 layers + 1 output layer), a kernel size of
# 7, and 32 kernels per CNN layer to predict the cropped-out pixel values.
import torch


class CNN_Model(torch.nn.Module):
    def __init__(self):
        number_of_hidden_layers = 2
        number_of_kernels = 32
        size_of_kernel = 7
        super(CNN_Model, self).__init__()

        cnn = []
        # number of channels = the image array and the crop array = 2
        number_input_channels = 2
        for i in range(number_of_hidden_layers):
            cnn.append(torch.nn.Conv2d(in_channels=number_input_channels, out_channels=number_of_kernels,
                                       kernel_size=size_of_kernel, bias=True,
                                       # padding 0s to have the same shape as the input,
                                       # otherwise the spatial dimension will shrink at each layer you apply!.
                                       padding=int(size_of_kernel/2)))
            # activation
            cnn.append(torch.nn.ReLU())
            number_input_channels = number_of_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)
        # without activation
        self.output_layer = torch.nn.Conv2d(in_channels=number_input_channels, out_channels=1,
                                            kernel_size=size_of_kernel, bias=True,
                                            padding=int(size_of_kernel / 2))

    def forward(self, x):
        hidden_layers_result = self.hidden_layers(x)
        predictions = self.output_layer(hidden_layers_result)
        # prediction now has the same shape as the input, we need to use the boolean mask to retrieve only the target.
        # using the crop_array to obtain the predicted pixel values in the cropped out rectangle:
        target_masks = x[:, -1:].to(dtype=torch.bool)
        predictions = [predictions[i, target_masks[i]] for i in range(len(x))]

        return predictions
