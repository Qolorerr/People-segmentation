import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import AverageMeter, PixAccIoUMetricT


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metric: nn.Module,
        resume: str,
        config: DictConfig,
        accelerator: Accelerator,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.loss = loss
        self.metric = metric
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config["trainer"]["val"]
        self.start_epoch = 1
        self.accelerator = accelerator

        # SETTING THE DEVICE
        self.model = self.accelerator.prepare(self.model)

        # CONFIGS
        cfg_trainer = self.config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]

        # OPTIMIZER
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = instantiate(self.config.optimizer, trainable_params)
        # TODO: lr_scheduler
        # self.lr_scheduler = instantiate(
        #     self.config.lr_scheduler, self.optimizer, self.epochs, len(train_loader)
        # )
        self.lr_scheduler = None
        self.optimizer = self.accelerator.prepare(
            self.optimizer #, self.lr_scheduler
        )

        # MONITORING
        self.monitor = cfg_trainer.get("monitor", "off")
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]
            self.mnt_best = -math.inf if self.mnt_mode == "max" else math.inf
            self.early_stoping = cfg_trainer.get("early_stop", math.inf)
            self.not_improved_count = 0

        # CHECKPOINTS & TENSORBOARD
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = str(os.path.join(cfg_trainer["log_dir"], self.config["name"], start_time))
        self.writer = tensorboard.SummaryWriter(writer_dir)
        self.checkpoint_dir = os.path.join(cfg_trainer["save_dir"], self.config["name"], start_time)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.threshold_value = cfg_trainer.get("threshold_value", 0.5)

        if resume:
            self._resume_checkpoint(resume)

        self.wrt_mode, self.wrt_step = "train", 0
        self.log_step = config["trainer"].get(
            "log_per_iter", int(np.sqrt(self.train_loader.batch_size))
        )
        if config["trainer"]["log_per_iter"]:
            self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        self.train_loader, self.val_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader
        )

    def _train_epoch(self, epoch: int) -> dict[str, np.float_]:
        self.model.train()
        if self.config["model"]["freeze_bn"]:
            self.model.freeze_bn()
        self.wrt_mode = "train"

        self._reset_metrics()
        tbar = tqdm(self.train_loader, desc="Train")
        for batch_idx, (inputs, targets) in enumerate(tbar):
            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)

            # METRICS
            self.total_loss.update(loss.item())
            metrics = self.metric(outputs, targets)
            self._update_accuracy_metrics(metrics)
            self._print_metrics(tbar, epoch, metrics, "TRAIN")

            self.accelerator.backward(loss)
            self.optimizer.step()

            # LOGGING & TENSORBOARD
            self._calc_wrt_step(epoch, batch_idx, loss.item())

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # METRICS TO TENSORBOARD
        self._log_accuracy_metrics()
        # TODO: Find out what this doing
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f"{self.wrt_mode}/Learning_rate_{i}", opt_group["lr"], self.wrt_step
            )

        # RETURN LOSS & METRICS
        log = {"loss": self.total_loss.avg, "pixAcc": self.total_pixAcc.avg, "mIoU": self.total_IoU.avg}
        return log

    def _valid_epoch(self, epoch: int) -> dict[str, np.float_]:
        if self.val_loader is None:
            self.logger.warning(
                "Not data loader was passed for the validation step, No validation is performed !"
            )
            return {}
        self.logger.info("###### EVALUATION ######")

        self.model.eval()
        self.wrt_mode = "val"

        self._reset_metrics()
        tbar = tqdm(self.val_loader, desc="Validation")
        for inputs, targets in tbar:
            # LOSS
            with torch.no_grad():
                output = self.model(inputs)
                loss = self.loss(output, targets)
                self.total_loss.update(loss.item())
                metrics = self.metric(output, targets)
                self._update_accuracy_metrics(metrics)
                self._print_metrics(tbar, epoch, metrics, "EVAL")

        # TODO: Visualisation of images

        # METRICS TO TENSORBOARD
        self.wrt_step = epoch * len(self.val_loader)
        self.writer.add_scalar(f"{self.wrt_mode}/loss", self.total_loss.avg, self.wrt_step)
        self._log_accuracy_metrics()

        log = {"loss": self.total_loss.avg, "pixAcc": self.total_pixAcc.avg, "mIoU": self.total_IoU.avg}

        return log

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs + 1):
            results = self._train_epoch(epoch)
            improved = False
            log = {"epoch": epoch, **results}

            if self.do_validation and epoch % self.config["trainer"]["val_per_epochs"] == 0:
                results = self._valid_epoch(epoch)

                # LOGGING INFO
                self.logger.info(f"\n## Info for epoch {epoch} ## ")
                for k, v in results.items():
                    self.logger.info(f"{str(k):15s}: {v}")

                # CHECKING IF THIS IS THE BEST MODEL
                if self.mnt_mode != "off":
                    try:
                        if self.mnt_mode == "min":
                            improved = log[self.mnt_metric] < self.mnt_best
                        else:
                            improved = log[self.mnt_metric] > self.mnt_best
                    except KeyError:
                        self.logger.warning(
                            f"The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops."
                        )
                        break

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        self.not_improved_count = 0
                    else:
                        self.not_improved_count += 1

                    if self.not_improved_count > self.early_stoping:
                        self.logger.info(
                            f"\nPerformance didn't improve for {self.early_stoping} epochs"
                        )
                        self.logger.warning("Training Stopped")
                        break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=improved)

    def _calc_wrt_step(self, epoch: int, batch_idx: int, loss_item) -> None:
        if batch_idx % self.log_step == 0:
            self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
            self.writer.add_scalar(f"{self.wrt_mode}/loss", loss_item, self.wrt_step)

    def _log_accuracy_metrics(self) -> None:
        for k, v in [("pixAcc", self.total_pixAcc.avg), ("mIoU", self.total_IoU.avg)]:
            self.writer.add_scalar(f"{self.wrt_mode}/{k}", v, self.wrt_step)

    def _print_metrics(self, tbar: tqdm, epoch: int, metrics: PixAccIoUMetricT, mode: str = "TRAIN") -> None:
        message = "{} ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |"
        message = message.format(mode, epoch, self.total_loss.avg, metrics.pixAcc, metrics.IoU)
        tbar.set_description(message)

    def _reset_metrics(self) -> None:
        self.total_pixAcc = AverageMeter("Total pixel accuracy")
        self.total_IoU = AverageMeter("Total mean IoU")
        self.total_loss = AverageMeter("Total loss")

    def _update_accuracy_metrics(self, metrics: PixAccIoUMetricT) -> None:
        self.total_IoU.update(metrics.IoU)
        self.total_pixAcc.update(metrics.pixAcc)

    def _save_checkpoint(self, epoch: int, save_best=False) -> None:
        state = {
            "arch": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = os.path.join(self.checkpoint_dir, f"checkpoint-epoch{epoch}.pth")
        self.logger.info(f"\nSaving a checkpoint: {filename} ...")
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f"best_model.pth")
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path: str) -> None:
        self.logger.info(f"Loading checkpoint : {resume_path}")
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.not_improved_count = 0

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                {"Warning! Current model is not the same as the one in the checkpoint"}
            )
        self.model.load_state_dict(checkpoint["state_dict"])
