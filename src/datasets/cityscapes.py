import os
from glob import glob

import cv2
import numpy as np
from numpy.typing import NDArray

from src.base import BaseDataset


class CityscapesDataset(BaseDataset):
    def __init__(self, id_to_train_id: dict[int, int], mode: str = "fine", load_limit: int | None = None, **kwargs):
        self.id_to_train_id = id_to_train_id
        self.num_classes = len(set(id_to_train_id.values()))
        self.mode = mode
        self.load_limit = load_limit
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
            if self.load_limit is not None and len(image_paths) > self.load_limit:
                break
        self.files = list(zip(image_paths, label_paths))
        # if self.load_limit:
        #     self.files = self.files[:self.load_limit]

    def _set_test_files(self) -> None:
        assert (self.mode == "fine" and self.split == "test")

        # Get images and labels folders
        image_folder_path = os.path.join(
            self.root, "leftImg8bit_trainvaltest", "leftImg8bit", self.split
        )

        # Get all images and labels paths
        image_paths = []
        for city in os.listdir(image_folder_path):
            image_paths.extend(sorted(glob(os.path.join(image_folder_path, city, "*.png"))))
            if self.load_limit is not None and len(image_paths) > self.load_limit:
                break
        self.files = image_paths
        # if self.load_limit:
        #     self.files = self.files[:self.load_limit]

    def _convert_to_segmentation_mask(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        height, width = mask.shape
        labels = sorted(set(self.id_to_train_id.values()))
        segmentation_mask = np.zeros((height, width, len(labels)), dtype=np.float32)
        for label_index, label in enumerate(labels):
            segmentation_mask[:, :, label_index] = (mask == label).astype(np.float32)

        return segmentation_mask

    def _load_data(self, index: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        image_path, label_path = self.files[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        for k, v in self.id_to_train_id.items():
            label[label == k] = v
        label = self._convert_to_segmentation_mask(label)
        return image, label

    def _load_test_data(self, index: int) -> NDArray[np.uint8]:
        image_path = self.files[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        return image
