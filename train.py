import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    if cfg.resume:
        resume = cfg.resume
        cfg = DictConfig(torch.load(cfg.resume)["config"])
        cfg.resume = resume

    # DATA LOADERS
    train_loader: DataLoader = instantiate(cfg.train_loader)
    val_loader: DataLoader = instantiate(cfg.val_loader)

    # # MODEL
    # model: BaseModel = instantiate(cfg.model, train_loader.dataset.num_classes)
    #
    # # LOSS
    # loss: nn.Module = instantiate(cfg.loss)
    #
    # # TRAINING
    # trainer = Trainer(
    #     model=model,
    #     loss=loss,
    #     resume=cfg.resume,
    #     config=cfg,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    # )
    #
    # trainer.train()


if __name__ == "__main__":
    main()
