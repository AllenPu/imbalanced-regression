import numpy as np
import torchvision.models as models
import torch.nn as nn


def resent18_regression():
    model = models.resnet18(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, 1)
    )
    return model


def resnet18_cls(num_cls = 100):
    model = models.resnet18(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, num_cls)
    )
    return model