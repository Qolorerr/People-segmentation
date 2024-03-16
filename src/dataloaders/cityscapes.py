import os
from glob import glob

import cv2
import numpy as np

from src.base import BaseDataset


class CityscapesDataset(BaseDataset):
    def __init__(self, id_to_train_id: dict[int, int], mode: str = "fine", **kwargs):
        self.id_to_train_id = id_to_train_id
        self.num_classes = len(set(id_to_train_id.values()))
        self.mode = mode
        super().__init__(**kwargs)

    def _set_files(self) -> None:
        # Check mode
        assert (self.mode == "fine" and self.split in ["train", "val"]) or (
            self.mode == "coarse" and self.split in ["train", "train_extra", "val"]
        )

        # Get images and labels folders
        if self.mode == "fine":
            image_folder_path = os.path.join(
                self.root, "leftImg8bit_trainvaltest", "leftImg8bit", self.split
            )
            label_folder_path = os.path.join(self.root, "gtFine_trainvaltest", "gtFine", self.split)
        else:
            image_sub_folder = (
                "leftImg8bit_trainextra"
                if self.split == "train_extra"
                else "leftImg8bit_trainvaltest"
            )
            image_folder_path = os.path.join(self.root, image_sub_folder, "leftImg8bit", self.split)
            label_folder_path = os.path.join(self.root, "gtCoarse", "gtCoarse", self.split)
        assert os.listdir(image_folder_path) == os.listdir(label_folder_path)

        # Get all images and labels paths
        image_paths, label_paths = [], []
        for city in os.listdir(image_folder_path):
            image_paths.extend(sorted(glob(os.path.join(image_folder_path, city, "*.png"))))
            label_paths.extend(
                sorted(glob(os.path.join(label_folder_path, city, "*_gtFine_labelIds.png")))
            )
        self.files = list(zip(image_paths, label_paths))

    def _load_data(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB).astype(np.int32)
        label = np.vectorize(self.id_to_train_id.__getitem__)(label)
        return image, label, image_id
