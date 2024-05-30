import hydra
import torch.nn as nn
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src import Trainer


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):

    # DATA LOADERS
    train_loader: DataLoader = instantiate(cfg.train_loader)
    val_loader: DataLoader = instantiate(cfg.val_loader)

    accelerator: Accelerator = instantiate(cfg.accelerator)

    # MODEL
    model: nn.Module = instantiate(cfg.model, num_classes=train_loader.dataset.num_classes)

    # LOSS & METRIC
    loss: nn.Module = instantiate(cfg.loss)
    metric: nn.Module = instantiate(cfg.metric, classes_num=train_loader.dataset.num_classes)
    visualizer = instantiate(cfg.visualizer)

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        metric=metric,
        resume=cfg.resume,
        config=cfg,
        accelerator=accelerator,
        train_loader=train_loader,
        val_loader=val_loader,
        visualizer=visualizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
