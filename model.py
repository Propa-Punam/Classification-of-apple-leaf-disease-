import torch
import torch.nn as nn
from torchvision import models
import timm

# Define the ResNet-ViT hybrid model with Dropout
class ResNetViT(nn.Module):
    def __init__(self):
        super(ResNetViT, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the classification layer

        # Load pre-trained Vision Transformer (ViT)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove the classification layer

        self.dropout = nn.Dropout(0.5)  # Dropout with a 50% rate

    def forward(self, x):
        x_resnet = self.resnet(x)
        x_vit = self.vit(x)
        x = torch.cat((x_resnet, x_vit), dim=1)
        x = self.dropout(x)  # Apply dropout
        return x
