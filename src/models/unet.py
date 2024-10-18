import torch
from torch import nn

from src.utils import get_backbone

FeaturesT = dict[str | None, torch.Tensor | None]


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        decoder_filters: tuple[int] = (512, 256, 128, 64, 32),  # (256, 128, 64, 32, 16)
        parametric_upsampling: bool = False,
        skip_connection_names: list[str] | None = None,
        decoder_use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        # encoder/backbone
        self.backbone_name = backbone_name
        backbone_info = get_backbone(backbone_name, pretrained=pretrained)
        self.backbone, self.shortcut_features, self.bb_out_name = backbone_info

        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if skip_connection_names is not None:
            self.shortcut_features = skip_connection_names

        # decoder
        self.upsample_blocks = nn.ModuleList()
        # avoiding having more blocks than skip connections
        decoder_filters = decoder_filters[: len(self.shortcut_features)]
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, (filters_in, filters_out) in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(
                UpsampleBlock(
                    channels_in=filters_in,
                    channels_out=filters_out,
                    skip_in=shortcut_chs[num_blocks - i - 1],
                    parametric=parametric_upsampling,
                    use_bn=decoder_use_batchnorm,
                )
            )

        self.final_conv = nn.Conv2d(decoder_filters[-1], num_classes, kernel_size=1)

        if freeze_encoder:
            self.freeze_encoder_parameters()

    def forward(self, *input):
        x, features = self.forward_backbone(*input)

        for skip_feature_name, upsample_block in zip(
            self.shortcut_features[::-1], self.upsample_blocks
        ):
            x = upsample_block(x, features[skip_feature_name])

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, FeaturesT]:
        features: FeaturesT = {}
        if None in self.shortcut_features:
            features[None] = None

        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self) -> tuple[list[int], int]:
        x = torch.zeros(1, 3, 224, 224)
        has_fullres_features = (
            self.backbone_name.startswith("vgg") or self.backbone_name == "unet_encoder"
        )
        channels = [] if has_fullres_features else [0]

        out_channels = 1
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break

        return channels, out_channels

    def freeze_encoder_parameters(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int | None = None,
        skip_in: int = 0,
        use_bn: bool = True,
        parametric: bool = True,
    ) -> None:
        super().__init__()

        self.parametric = parametric
        channels_out = channels_in / 2 if channels_out is None else channels_out

        if parametric:
            # options: kernel=2 padding=0, kernel=4 padding=1
            self.up: nn.Upsample | nn.ConvTranspose2d = nn.ConvTranspose2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=not use_bn,
            )
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            channels_in = channels_in + skip_in
            self.conv1 = nn.Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=(not use_bn),
            )

        self.bn1 = nn.BatchNorm2d(channels_out) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        conv2_in = channels_out if not parametric else channels_out + skip_in
        self.conv2 = nn.Conv2d(
            in_channels=conv2_in,
            out_channels=channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_bn,
        )
        self.bn2 = nn.BatchNorm2d(channels_out) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if self.parametric:
            x = self.bn1(x)
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
