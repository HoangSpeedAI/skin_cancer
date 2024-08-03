import torch
import torch.nn as nn
from torchvision import models

class SkinClassificationModel(nn.Module):
    def __init__(self, input_channels, output_channels, architecture):
        super(SkinClassificationModel, self).__init__()
        
        # Load the appropriate pretrained model
        # Update the final fully connected layer for binary classification
        if architecture == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, output_channels)
        elif architecture == "resnet50":
            self.base_model = models.resnet50(pretrained=True)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, output_channels)
        elif architecture == "efficientnet_b1":
            self.base_model = models.efficientnet_b1(pretrained=True)
            lastconv_output_channels = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(lastconv_output_channels, output_channels),
            )
        elif architecture == "efficientnet_b3":
            self.base_model = models.efficientnet_b3(pretrained=True)
            lastconv_output_channels = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(lastconv_output_channels, output_channels),
            )
        elif architecture == "efficientnet_b5":
            self.base_model = models.efficientnet_b5(pretrained=True)
            lastconv_output_channels = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(lastconv_output_channels, output_channels),
            )
        else:
            raise ValueError("Unsupported architecture: {}".format(architecture))

        # Freeze all layers of the base model
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        
        # # Unfreeze the final classification layer
        # if architecture.startswith('efficientnet'):
        #     for param in self.base_model.classifier.parameters():
        #         param.requires_grad = True
        # else:
        #     for param in self.base_model.fc.parameters():
        #         param.requires_grad = True

    def forward(self, x):
        return self.base_model(x) 


def get_model(input_channels, output_channels, architecture):
    return SkinClassificationModel(input_channels, output_channels, architecture)