import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG16_Weights,
    VGG19_Weights,
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def get_backbone(name: str, pretrained=True) -> tuple[nn.Module, list[str | None], str]:
    if name == "resnet18":
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == "resnet34":
        backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
    elif name == "resnet50":
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    elif name == "resnet101":
        backbone = models.resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
    elif name == "resnet152":
        backbone = models.resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
    elif name == "vgg16":
        backbone = models.vgg16_bn(weights=VGG16_Weights.DEFAULT if pretrained else None).features
    elif name == "vgg19":
        backbone = models.vgg19_bn(weights=VGG19_Weights.DEFAULT if pretrained else None).features
    elif name == "densenet121":
        backbone = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "densenet161":
        backbone = models.densenet161(
            weights=DenseNet161_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "densenet169":
        backbone = models.densenet169(
            weights=DenseNet169_Weights.DEFAULT if pretrained else None
        ).features
    elif name == "densenet201":
        backbone = models.densenet201(
            weights=DenseNet201_Weights.DEFAULT if pretrained else None
        ).features
    else:
        raise ValueError("Unknown backbone.")

    if name.startswith("resnet"):
        feature_names = [None, "relu", "layer1", "layer2", "layer3"]
        backbone_output = "layer4"
    elif name == "vgg16":
        feature_names = ["5", "12", "22", "32", "42"]
        backbone_output = "43"
    elif name == "vgg19":
        feature_names = ["5", "12", "25", "38", "51"]
        backbone_output = "52"
    elif name.startswith("densenet"):
        feature_names = [None, "relu0", "denseblock1", "denseblock2", "denseblock3"]
        backbone_output = "denseblock4"
    else:
        raise ValueError("Unknown backbone.")

    return backbone, feature_names, backbone_output
