import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWithMetadata(nn.Module):
    def __init__(self, metadata_input_dim, out_dim=1):
        """
        Args:
            metadata_input_dim (int): Number of input features in the metadata MLP
            out_dim (int): Number of outputs for the final classification (1 for binary)
        """
        super().__init__()
        
        # 1) Load a pretrained ResNet with optimizations
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze early layers to reduce computation
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False
        
        # Remove the final FC layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # 2) Optimized metadata MLP
        self.metadata_mlp = nn.Sequential(
            nn.BatchNorm1d(metadata_input_dim),  # Normalize inputs
            nn.Linear(metadata_input_dim, 16),
            nn.ReLU(inplace=True),  # inplace operations save memory
        )
        
        # 3) Final classifier
        self.classifier = nn.Linear(512 + 16, out_dim)

    def forward(self, images, metadata):
        # Extract features from images
        img_features = self.resnet(images)
        # Extract features from metadata
        meta_features = self.metadata_mlp(metadata)
        # Concatenate
        combined = torch.cat([img_features, meta_features], dim=1)
        # Classifier
        logits = self.classifier(combined)
        return logits