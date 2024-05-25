from typing import cast

import torch
from torch import Tensor
from torchvision.utils import make_grid


class Visualization:
    def __init__(self, palette: list[list[int]],
                 threshold_value: float = 0.5):
        self.palette = torch.Tensor(palette)
        self.threshold_value = threshold_value
        self.visual = []

    def add_to_visual(self, output: Tensor, target: Tensor) -> None:
        if len(self.visual) >= 15:
            return
        predict = cast(torch.Tensor, output > self.threshold_value).float()
        outputs = torch.argmax(predict, dim=1).long()
        targets = torch.argmax(target, dim=1).long()
        self.visual.append([outputs.data.cpu()[0], targets.data.cpu()[0]])

    def flush_visual(self) -> Tensor:
        images = []
        for output, target in self.visual:
            output, target = self._colorize_mask(output), self._colorize_mask(target)
            images.extend([target, output])
        images = torch.stack(images, 0)
        print(images.shape)
        images = make_grid(images.cpu(), nrow=2, padding=5)
        print(images.shape)
        self.visual = []
        return images

    def _colorize_mask(self, mask: Tensor) -> Tensor:
        height, width = mask.shape
        colorized_mask = torch.zeros(height, width, 3)
        for color_id, rgb in enumerate(self.palette):
            colorized_mask[mask == color_id, :] = rgb
        return colorized_mask.permute(2, 0, 1) / 255.
