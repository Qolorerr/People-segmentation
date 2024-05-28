import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src import Tester


@hydra.main(version_base="1.2", config_path="config", config_name="test_config")
def main(cfg: DictConfig):
    assert cfg.save_file

    save_cfg = DictConfig(torch.load(cfg.save_file)["config"])
    cfg.name = save_cfg.name
    cfg.model = save_cfg.model

    # DATA LOADERS
    test_loader: DataLoader = instantiate(cfg.test_loader)

    accelerator: Accelerator = instantiate(cfg.accelerator)

    # MODEL
    model: nn.Module = instantiate(cfg.model, num_classes=test_loader.dataset.num_classes)

    visualizer = instantiate(cfg.visualizer)

    # TRAINING
    tester = Tester(
        model=model,
        resume=cfg.save_file,
        config=cfg,
        accelerator=accelerator,
        test_loader=test_loader,
        visualizer=visualizer,
    )

    tester.test()


if __name__ == "__main__":
    main()
