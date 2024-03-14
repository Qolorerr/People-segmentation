import os
from glob import glob

import cv2
import numpy as np

from src.base import BaseDataset, BaseDataLoader


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
            image_folder_path = os.path.join(self.root, "leftImg8bit_trainvaltest", "leftImg8bit", self.split)
            label_folder_path = os.path.join(self.root, "gtFine_trainvaltest", "gtFine", self.split)
        else:
            image_sub_folder = "leftImg8bit_trainextra" if self.split == "train_extra" else "leftImg8bit_trainvaltest"
            image_folder_path = os.path.join(self.root, image_sub_folder, "leftImg8bit", self.split)
            label_folder_path = os.path.join(self.root, "gtCoarse", "gtCoarse", self.split)
        assert os.listdir(image_folder_path) == os.listdir(label_folder_path)

        # Get all images and labels paths
        image_paths, label_paths = [], []
        for city in os.listdir(image_folder_path):
            image_paths.extend(sorted(glob(os.path.join(image_folder_path, city, "*.png"))))
            label_paths.extend(sorted(glob(os.path.join(label_folder_path, city, "*_gtFine_labelIds.png"))))
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


class Cityscapes(BaseDataLoader):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            split: str,
            palette: list[int],
            id_to_train_id: dict[int, int],
            crop_size: int | None = None,
            base_size: int | None = None,
            scale: bool = True,
            num_workers: int = 1,
            mode: str = "fine",
            val: bool = False,
            augment: bool = False,
            shuffle: bool = False,
            flip: bool = False,
            rotate: bool = False,
            blur: bool = False,
            return_id: bool = False,
    ):
        # TODO: discuss about
        self.mean = [0.28689529, 0.32513294, 0.28389176]
        self.std = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            "root": data_dir,
            "split": split,
            "mean": self.mean,
            "std": self.std,
            "augment": augment,
            "crop_size": crop_size,
            "base_size": base_size,
            "scale": scale,
            "flip": flip,
            "blur": blur,
            "rotate": rotate,
            "return_id": return_id,
            "val": val,
            "palette": palette
        }

        self.dataset = CityscapesDataset(id_to_train_id=id_to_train_id, mode=mode, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, num_workers)
