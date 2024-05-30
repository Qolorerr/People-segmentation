from typing import cast

import torch
from torch import Tensor
from torchvision.utils import make_grid


class Visualization:
    def __init__(self, palette: list[list[int]],
                 mean: list[float],
                 std: list[float],
                 threshold_value: float = 0.5,
                 number_of_images: int = 15):
        self.palette = Tensor(palette)
        self.mean = Tensor(mean)
        self.std = Tensor(std)
        self.threshold_value = threshold_value
        self.number_of_images = number_of_images
        self.visual = []

    def add_to_visual(self, inputs: Tensor, output: Tensor, target: Tensor | None = None) -> None:
        if len(self.visual) >= self.number_of_images:
            return
        predict = cast(torch.Tensor, output > self.threshold_value).float()
        outputs = torch.argmax(predict, dim=1).long()
        if target is not None:
            targets = torch.argmax(target, dim=1).long()
            self.visual.append([inputs.data.cpu()[0].permute(1, 2, 0), outputs.data.cpu()[0], targets.data.cpu()[0]])
        else:
            self.visual.append([inputs.data.cpu()[0].permute(1, 2, 0), outputs.data.cpu()[0]])

    def flush_visual(self) -> Tensor:
        if not self.visual:
            return torch.zeros(1, 1, 1)
        images = []
        if len(self.visual[0]) == 3:
            for inputs, output, target in self.visual:
                inputs = self._denormalize(inputs)
                output, target = self._colorize_mask(output), self._colorize_mask(target)
                images.extend([inputs, target, output])
        elif len(self.visual[0]) == 2:
            for inputs, output in self.visual:
                inputs = self._denormalize(inputs)
                output = self._colorize_mask(output)
                images.extend([inputs, output])
        images = torch.stack(images, 0)
        images = make_grid(images.cpu(), nrow=len(self.visual[0]), padding=5)
        self.visual = []
        return images

    def _colorize_mask(self, mask: Tensor) -> Tensor:
        height, width = mask.shape
        colorized_mask = torch.zeros(height, width, 3)
        for color_id, rgb in enumerate(self.palette):
            colorized_mask[mask == color_id, :] = rgb
        return colorized_mask.permute(2, 0, 1) / 255.

    def _denormalize(self, images: Tensor) -> Tensor:
        images = images * self.std + self.mean
        return images.permute(2, 0, 1)
