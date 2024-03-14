from torch.utils.data import DataLoader

from src.base.base_dataset import BaseDataset


class BaseDataLoader(DataLoader):
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        super().__init__(**init_kwargs)
