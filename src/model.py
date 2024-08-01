import torch
import torch.nn as nn
from torchvision import models

class SkinClassificationModel(nn.Module):
    def __init__(self, input_channels, output_channels, architecture):
        super(SkinClassificationModel, self).__init__()
        if architecture == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
        elif architecture == "resnet50":
            self.base_model = models.resnet50(pretrained=True)
        else:
            raise ValueError("Unsupported architecture: {}".format(architecture))
        
        # Modify the first convolution layer to accept the specified number of input channels
        self.base_model.conv1 = nn.Conv2d(input_channels,
                                          64,
                                          kernel_size=(7, 7),
                                          stride=(2, 2),
                                          padding=(3, 3),
                                          bias=False)
        
        # Update the final fully connected layer for binary classification
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, output_channels)

    def forward(self, x):
        return self.base_model(x)

def get_model(input_channels, output_channels, architecture):
    return SkinClassificationModel(input_channels, output_channels, architecture)