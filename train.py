import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src import Trainer


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.resume:
        resume = cfg.resume
        cfg = DictConfig(torch.load(cfg.resume)["config"])
        cfg.resume = resume

    # DATA LOADERS
    train_loader: DataLoader = instantiate(cfg.train_loader)
    val_loader: DataLoader = instantiate(cfg.val_loader)

    # MODEL
    model: nn.Module = instantiate(cfg.model, train_loader.dataset.num_classes)

    # LOSS & METRIC
    loss: nn.Module = instantiate(cfg.loss)
    metric: nn.Module = instantiate(cfg.metric, classes_num=train_loader.dataset.num_classes)

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        metric=metric,
        resume=cfg.resume,
        config=cfg,
        accelerator=Accelerator(),
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.train()


if __name__ == "__main__":
    main()
