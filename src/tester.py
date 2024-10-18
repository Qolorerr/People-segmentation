import logging
import os
from datetime import datetime

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch import nn
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import Visualization


class Tester:
    def __init__(
        self,
        model: nn.Module,
        resume: str,
        config: DictConfig,
        accelerator: Accelerator,
        test_loader: DataLoader,
        visualizer: Visualization,
    ):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.visualizer = visualizer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.accelerator = accelerator

        # SETTING THE DEVICE
        self.model = self.accelerator.prepare(self.model)

        # CONFIGS
        cfg_tester = self.config["tester"]

        # TENSORBOARD
        start_time = datetime.now().strftime("%m-%d_%H-%M")
        writer_dir = str(os.path.join(cfg_tester["log_dir"], self.config["name"], start_time))
        self.writer = tensorboard.SummaryWriter(writer_dir)
        info_to_write = [
            "crop_size",
            "threshold_value",
            "test_loader",
            "model",
            "save_file",
        ]
        for info in info_to_write:
            self.writer.add_text(f"info/{info}", str(self.config[info]))

        self.threshold_value = self.config.get("threshold_value", 0.5)

        self._resume_checkpoint(resume)

        self.wrt_mode, self.wrt_step = "test", 0

        self.num_classes = self.test_loader.dataset.num_classes

        self.test_loader = self.accelerator.prepare(self.test_loader)

    def test(self) -> None:
        self.model.eval()

        tbar = tqdm(self.test_loader, desc="Testing")
        for inputs in tbar:
            with torch.no_grad():
                output = self.model(inputs)
                self.visualizer.add_to_visual(inputs, output)

        # WRITING & VISUALIZING THE MASKS
        test_img = self.visualizer.flush_visual()
        self.writer.add_image(
            tag=f"{self.wrt_mode}/inputs_predictions",
            img_tensor=test_img,
            global_step=self.wrt_step,
            dataformats="CHW",
        )
        input("Enter anything")

    def _resume_checkpoint(self, resume_path: str) -> None:
        self.logger.info(f"Loading checkpoint : {resume_path}")
        checkpoint = torch.load(resume_path)

        self.model.load_state_dict(checkpoint["state_dict"])
