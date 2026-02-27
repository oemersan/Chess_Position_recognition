import torch
import torch.nn as nn
import torchvision.models as models

class CornerDetectorLinear(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 8)

    def forward(self, x):
        return self.backbone(x)
    
if __name__ == "__main__":
    model = CornerDetectorLinear(pretrained=True)
    print(model)