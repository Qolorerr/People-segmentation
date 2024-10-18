import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
import albumentations


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transforms: list[albumentations.BasicTransform],
        return_id: bool = False,
    ):
        self.root = root
        self.split = split
        self.files = []
        if self.split == "test":
            self._set_test_files()
        else:
            self._set_files()
        self.transforms = transforms
        self.return_id = return_id

    def _set_files(self):
        raise NotImplementedError

    def _set_test_files(self):
        raise NotImplementedError

    def _load_data(self, index: int):
        raise NotImplementedError

    def _load_test_data(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self, index: int
    ) -> tuple[NDArray[np.float_], NDArray[np.uint8]] | NDArray[np.float_]:

        if self.split == "test":
            return self._get_test_item(index)

        image, label = self._load_data(index)
        transformed = self.transforms(image=image, mask=label)
        image, label = transformed["image"], transformed["mask"]
        return image.float(), label.float()

    def _get_test_item(self, index: int) -> NDArray[np.float_]:
        image = self._load_test_data(index)
        transformed = self.transforms(image=image)
        image = transformed["image"]
        return image.float()

    def __repr__(self) -> str:
        fmt_str = f"Dataset: {self.__class__.__name__}\n"
        fmt_str += f"\t# data: {self.__len__()}\n"
        fmt_str += f"\tSplit: {self.split}\n"
        fmt_str += f"\tRoot: {self.root}"
        return fmt_str
