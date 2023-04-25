from torch import nn
from torchvision import models


class ExampleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        self.set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        self.model_ft = model_ft

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model_ft(x)