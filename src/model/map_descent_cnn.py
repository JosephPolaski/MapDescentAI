import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MapDescentCNN(nn.Module):

    def __init__(self, number_of_classes : int):
        super(MapDescentCNN, self).__init__()

        # establish convolutional layers with increasing channels to improve complex pattern learning
        # convolution_layer_1: this will detect basic edges and corners
        # convolution_layer_2: this will detect morecomplex shapes
        self.convolution_layer_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.convolution_layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # establish fully connected layers for using features to make predictions
        self.fully_connected_layer_1 = nn.Linear(32 * 16 * 16, 128)
        self.fully_connected_layer_2 =nn.Linear(128, number_of_classes)

        # max pooling layer for downsampling features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def backward_pass(self, logits : torch.Tensor, labels):
        pass

    def calculate_loss(self, logits : torch.Tensor, labels):
        pass

    def forward_pass(self, images: torch.Tensor) -> torch.Tensor:
        """
            Pass images through two convolution layers to capture patterns
            and fully connected layers to compute raw class scores (logits)            
        """
        relu_convolution_1 = F.relu(self.convolution_layer_1(images)) # apply rectified linear unit function to images
        output_convolution_1 = self.pool(relu_convolution_1)

        relu_convolution_2 = F.relu(self.convolution_layer_2(output_convolution_1)) # apply rectified linear unit function to output of layer 1
        output_convolution_2 = self.pool(relu_convolution_2)

        # flatten features for fully connected layers
        flat_features = torch.flatten(output_convolution_2, start_dim=1)

        output_fc_1 = F.relu(self.fully_connected_layer_1(flat_features))
        logits = self.fully_connected_layer_2(output_fc_1)       

        return logits