import numpy as np
from torch.utils.data import Dataset
import albumentations


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        palette: list[int],
        transforms: list[albumentations.BasicTransform],
        return_id: bool = False,
    ):
        self.root = root
        self.split = split
        self.files = []
        self._set_files()
        self.transformation = albumentations.Compose(transforms)
        self.return_id = return_id
        self.palette = palette

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, np.ndarray, str] | tuple[np.ndarray, np.ndarray]:
        image, label, image_id = self._load_data(index)
        transformed = self.transformation(image=image, mask=label)
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
