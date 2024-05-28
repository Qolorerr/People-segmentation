from collections import namedtuple
from typing import cast

import torch
from torch import Tensor, nn


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def average_str(self) -> str:
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


LongTensorT = torch.LongTensor
PixAccIoUMetricT = namedtuple("PixAccIoUMetricT", ["IoU", "pixAcc"])


class PixAccIoUMetric(nn.Module):
    def __init__(
            self,
            classes_num: int,
            ignore_index: int | None = None,
            reduction: str | None = None,
            class_weights: list[float] | None = None,
            threshold_value: float = 0.5,
    ) -> None:
        super().__init__()

        self.cls_num = classes_num
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.class_weights = class_weights
        self.threshold_value = threshold_value

    @torch.no_grad()
    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
    ) -> PixAccIoUMetricT:
        # from
        # https://github.com/qubvel/segmentation_models.pytorch/blob/master
        # /segmentation_models_pytorch/metrics/functional.py

        predict = cast(torch.Tensor, output > self.threshold_value).float()

        outputs = torch.argmax(predict, dim=1).long()
        targets = torch.argmax(target, dim=1).long()

        batch_size, height, width = outputs.shape

        if self.ignore_index is not None:
            ignore = cast(torch.Tensor, targets == self.ignore_index)
            outputs = torch.where(ignore, -1, outputs)
            targets = torch.where(ignore, -1, targets)

        tp_count = cast(
            LongTensorT, torch.zeros(batch_size, self.cls_num, dtype=torch.long)
        )
        fp_count = cast(
            LongTensorT, torch.zeros(batch_size, self.cls_num, dtype=torch.long)
        )
        fn_count = cast(
            LongTensorT, torch.zeros(batch_size, self.cls_num, dtype=torch.long)
        )
        tn_count = cast(
            LongTensorT, torch.zeros(batch_size, self.cls_num, dtype=torch.long)
        )

        for i in range(batch_size):
            target_i = targets[i]
            output_i = outputs[i]
            mask = output_i == target_i
            matched = torch.where(mask, target_i, -1)
            tp = torch.histc(
                matched.float(), bins=self.cls_num, min=0, max=self.cls_num - 1
            )
            fp = (
                    torch.histc(
                        output_i.float(), bins=self.cls_num, min=0, max=self.cls_num - 1
                    )
                    - tp
            )
            fn = (
                    torch.histc(
                        target_i.float(), bins=self.cls_num, min=0, max=self.cls_num - 1
                    )
                    - tp
            )

            tp_count[i] = tp.long()
            fp_count[i] = fp.long()
            fn_count[i] = fn.long()
            tn_count[i] = (height * width - (tp + fp + fn)).long()

        return PixAccIoUMetricT(
            _compute_iou_metric(
                tp=tp_count,
                fp=fp_count,
                fn=fn_count,
                reduction=self.reduction,
                class_weights=self.class_weights,
            ), _compute_pix_acc_metric(
                tp=tp_count, tn=tn_count, height=height, width=width
            ))


def _compute_pix_acc_metric(
        tp: LongTensorT,
        tn: LongTensorT,
        height: int,
        width: int,
) -> torch.Tensor:
    if height * width == 0:
        return torch.tensor(0.0)

    batch_size, cls_num = tp.shape

    tp = cast(LongTensorT, tp.sum())
    tn = cast(LongTensorT, tn.sum())

    return (tp + tn) / (batch_size * cls_num * height * width)


def _compute_iou_metric(
        tp: LongTensorT,
        fp: LongTensorT,
        fn: LongTensorT,
        reduction: str | None = None,
        class_weights: list[float] | None = None,
) -> torch.Tensor:
    if class_weights is None and reduction is not None and "weighted" in reduction:
        raise ValueError(
            f"Class weights should be provided for `{reduction}` reduction."
        )

    class_weights = class_weights if class_weights is not None else 1.0
    class_weights = torch.tensor(class_weights).to(tp.device)
    class_weights = class_weights / class_weights.sum()

    if reduction == "micro":
        tp = cast(LongTensorT, tp.sum())
        fp = cast(LongTensorT, fp.sum())
        fn = cast(LongTensorT, fn.sum())
        score = _iou_score(tp, fp, fn)

    elif reduction == "macro":
        tp = cast(LongTensorT, tp.sum(0))
        fp = cast(LongTensorT, fp.sum(0))
        fn = cast(LongTensorT, fn.sum(0))
        score = _handle_zero_division(_iou_score(tp, fp, fn))
        score = (score * class_weights).mean()

    elif reduction == "weighted":
        tp = cast(LongTensorT, tp.sum(0))
        fp = cast(LongTensorT, fp.sum(0))
        fn = cast(LongTensorT, fn.sum(0))
        score = _handle_zero_division(_iou_score(tp, fp, fn))
        score = (score * class_weights).sum()

    elif reduction == "micro-imagewise":
        tp = cast(LongTensorT, tp.sum(1))
        fp = cast(LongTensorT, fp.sum(1))
        fn = cast(LongTensorT, fn.sum(1))
        score = _handle_zero_division(_iou_score(tp, fp, fn))
        score = score.mean()

    elif reduction in {"macro-imagewise", "weighted-imagewise"}:
        score = _iou_score(tp, fp, fn)
        score = (score.mean(0) * class_weights).mean()

    elif reduction == "none" or reduction is None:
        score = _iou_score(tp, fp, fn)

    else:
        raise ValueError(
            "`reduction` should be in [micro, macro, weighted, micro-imagewise, "
            "macro-imagesize, weighted-imagewise, none, None]"
        )

    return score


def _iou_score(tp: LongTensorT, fp: LongTensorT, fn: LongTensorT) -> torch.Tensor:
    return tp / (tp + fp + fn)


def _handle_zero_division(x: torch.Tensor) -> torch.Tensor:
    nans = torch.isnan(x)
    value = torch.tensor(0.0, dtype=x.dtype).to(x.device)
    x = torch.where(nans, value, x)
    return x
