import torch
from torch import nn
from torch.nn import functional as F  # noqa


EPSILON = 1e-7


class MulticlassDiceLoss(nn.Module):
    def __init__(self, eps: float = EPSILON) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(logits, dim=1)

        intersection = (targets * probas).sum((0, 2, 3)).clamp_min(self.eps)
        cardinality = (targets + probas).sum((0, 2, 3)).clamp_min(self.eps)

        dice_coefficient = 2.0 * intersection / cardinality

        dice_loss = 1.0 - dice_coefficient

        return dice_loss.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight: float, dice_eps: float = EPSILON) -> None:
        super().__init__()
        self.weight = weight
        self.dice = MulticlassDiceLoss(eps=dice_eps)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dice(logits, targets) * self.weight, self.ce(logits, targets)
