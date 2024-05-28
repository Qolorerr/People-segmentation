from typing import Literal

import torch
from torch import nn, Tensor, einsum
from torch.nn import functional as F  # noqa


EPSILON = 1e-7


class MulticlassDiceLoss(nn.Module):
    def __init__(self, eps: float = EPSILON) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)

        intersection = (targets * probs).sum((0, 2, 3)).clamp_min(self.eps)
        cardinality = (targets + probs).sum((0, 2, 3)).clamp_min(self.eps)

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


def compute_dice_per_channel(
    probs: Tensor,
    targets: Tensor,
    epsilon: float = EPSILON,
    weights: Tensor | None = None,
    mask_zeros: bool = False,
) -> Tensor:
    """input and target are `N x C x Spatial`, weights are `C x 1`."""

    assert probs.size() == targets.size()

    probs = probs.transpose(1, 0).flatten(2)
    targets = targets.transpose(1, 0).flatten(2).float()

    if mask_zeros:
        mask = torch.any(((probs.sum(dim=2) > 0.0) | (targets.sum(dim=2) > 0.0)), dim=0)
        if not torch.any(mask):
            return torch.zeros(probs.shape[0])

        probs = probs[:, mask]
        targets = targets[:, mask]

    numerator = (probs * targets).sum(-1)
    if weights is not None:
        numerator = weights * numerator

    denominator = (probs + targets).sum(-1)

    return torch.mean(2 * (numerator / denominator.clamp(min=epsilon)), dim=1)


NormalizationOptionT = Literal["sigmoid", "softmax", "none"]


class BCEDiceBoundaryLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.25,
        gamma: float = 0.25,
        is_3d: bool = True,
        normalization: NormalizationOptionT = "sigmoid",
        weights: Tensor | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(
            normalization=normalization, epsilon=epsilon, weights=weights
        )
        self.boundary = BoundaryLoss(is_3d=is_3d)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return (
            self.alpha * self.bce(inputs, targets)
            + self.beta * self.dice(inputs, targets)
            + self.gamma * self.boundary(torch.sigmoid(inputs), targets)
        )


class DiceLoss(nn.Module):
    def __init__(
        self,
        normalization: NormalizationOptionT = "sigmoid",
        weights: Tensor | None = None,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()

        if normalization == "sigmoid":
            self.normalization: nn.Module = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        elif normalization == "none":
            self.normalization = nn.Identity()
        else:
            raise ValueError

        self.register_buffer("weights", weights)

        self.epsilon = epsilon

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        probs = self.normalization(inputs)
        per_channel_dice = compute_dice_per_channel(
            probs=probs, targets=targets, epsilon=self.epsilon, weights=self.weights
        )
        return 1.0 - torch.mean(per_channel_dice)


class BoundaryLoss(nn.Module):
    def __init__(self, is_3d: bool = True) -> None:
        super().__init__()

        self.pattern = "bcdwh,bcdwh->bcdwh" if is_3d else "bcwh,bcwh->bcwh"

    def forward(self, probas: Tensor, targets: torch.Tensor) -> torch.Tensor:
        return einsum(self.pattern, probas, targets).mean()
