import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import albumentations as A


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        mean: list[float],
        std: list[float],
        palette: list[int],
        base_size: int | None = None,
        augment: bool = True,
        val: bool = False,
        crop_size: int | None = None,
        scale: bool = True,
        flip: bool = True,
        rotate: bool = False,
        blur: bool = False,
        return_id: bool = False,
    ):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        self.base_size = base_size
        self.scale = scale
        self.flip = flip
        self.rotate = rotate
        self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.train_augmentation: A.Compose
        self.val_augmentation: A.Compose
        self._init_transforms()
        self.return_id = return_id
        self.palette = palette

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index: int):
        raise NotImplementedError

    def _init_transforms(self) -> None:
        train_transform = []
        val_transform = []
        if self.base_size:
            size_transform = [
                A.LongestMaxSize(self.base_size),
                A.RandomScale(scale_limit=0.5, always_apply=self.scale, p=0),
            ]
            train_transform += size_transform

        if self.rotate:
            rotate_transform = [A.Rotate(limit=10)]
            train_transform += rotate_transform

        if self.crop_size:
            crop_transform = [
                A.SmallestMaxSize(self.crop_size),
                A.CenterCrop(height=self.crop_size, width=self.crop_size),
            ]
            train_transform += crop_transform
            val_transform += crop_transform

        if self.flip:
            flip_transform = [A.Flip()]
            train_transform += flip_transform

        if self.blur:
            blur_transform = [A.Blur()]
            train_transform += blur_transform

        normalize_transform = [A.Normalize(mean=self.mean, std=self.std), ToTensorV2()]
        train_transform += normalize_transform
        val_transform += normalize_transform
        self.train_augmentation = A.Compose(train_transform)
        self.val_augmentation = A.Compose(val_transform)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str] | tuple[np.ndarray, np.ndarray]:
        image, label, image_id = self._load_data(index)
        if self.val:
            transformed = self.val_augmentation(image=image, mask=label)
            image, label = transformed["image"], transformed["mask"]
        elif self.augment:
            transformed = self.train_augmentation(image=image, mask=label)
            image, label = transformed["image"], transformed["mask"]
        if self.return_id:
            return image, label, image_id
        return image, label

    def __repr__(self) -> str:
        fmt_str = f"Dataset: {self.__class__.__name__}\n"
        fmt_str += f"\t# data: {self.__len__()}\n"
        fmt_str += f"\tSplit: {self.split}\n"
        fmt_str += f"\tRoot: {self.root}"
        return fmt_str
