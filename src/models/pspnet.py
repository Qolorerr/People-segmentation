import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils import initialize_weights, get_backbone


class PyramidSceneModule(nn.Module):
    def __init__(self, in_channels: int, bin_sizes: list[int], norm_layer: nn.Module):
        super().__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, b_s, norm_layer) for b_s in bin_sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + (out_channels * len(bin_sizes)),
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    @staticmethod
    def _make_stages(in_channels: int, out_channels: int, bin_sz: int, norm_layer: nn.Module) -> nn.Sequential:
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features: Tensor) -> Tensor:
        h, w = features.shape[2], features.shape[3]
        pyramids = [features]
        pyramids.extend(
            [
                F.interpolate(stage(features), size=(h, w), mode="bilinear", align_corners=True)
                for stage in self.stages
            ]
        )
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_bn: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        backbone_info = get_backbone(backbone_name, pretrained=pretrained)
        self.backbone, self.shortcut_features, self.bb_out_name = backbone_info

        backbone_out_channels = self.backbone.fc.in_features
        self.main_branch = nn.Sequential(
            PyramidSceneModule(backbone_out_channels, bin_sizes=[1, 2, 3, 6], norm_layer=nn.BatchNorm2d),
            nn.Conv2d(backbone_out_channels // 4, num_classes, kernel_size=1),
        )

        initialize_weights(self.main_branch)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            self.freeze_backbone_params()

    def forward(self, x: torch.Tensor):
        input_size = (x.size()[2], x.size()[3])
        x, _ = self.forward_backbone(x)

        x = self.main_branch(x)
        x = F.interpolate(x, size=input_size, mode="bilinear")
        x = x[:, :, : input_size[0], : input_size[1]]
        return x

    def forward_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str | None, torch.Tensor | None]]:
        features: dict[str | None, torch.Tensor | None] = {}
        if None in self.shortcut_features:
            features[None] = None

        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def freeze_backbone_params(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
