import numpy as np
from numpy.typing import NDArray
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
        self.transforms = transforms
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
    ) -> tuple[NDArray[np.float_], NDArray[np.uint8]]:
        image, label = self._load_data(index)
        # print(image.shape, label.shape)
        # print(image.dtype, label.dtype)
        transformed = self.transforms(image=image, mask=label)
        image, label = transformed["image"], transformed["mask"]
        return image.float(), label.float()

    def __repr__(self) -> str:
        fmt_str = f"Dataset: {self.__class__.__name__}\n"
        fmt_str += f"\t# data: {self.__len__()}\n"
        fmt_str += f"\tSplit: {self.split}\n"
        fmt_str += f"\tRoot: {self.root}"
        return fmt_str
