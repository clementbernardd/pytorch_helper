"""Class that implements Dataloader Abstract"""
from abc import abstractmethod
from typing import Dict, List, Optional, Union

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from cthunder.features.preprocess_abstract import PreprocessAbstract
from cthunder.utils.utils import instantiate_class_from_init


class DataModuleAbstract(LightningDataModule):
    """
    Class that loads the different dataloaders for training, validation and testing sets.

    Methods:
        __init__(self, preprocesses, batch_size, shuffle, device, *args, **kwargs)
        prepare_data(self)
        collate_fn(self)
        _get_dataloader_from_split(self, split)
        train_dataloader()
        val_dataloader()
        test_dataloader()
    """

    def __init__(
        self,
        dataset_init: Dict,
        preprocesses: Optional[List[PreprocessAbstract]],
        batch_size: int,
        device: Union[Optional[str], torch.device],
        shuffle: bool = False,
        num_workers: int = 0,
        *args,
        **kwargs,
    ):
        """
        Initialisation of the class
        Args:
        :param dataset_init: a dictionary with the "class_path" and "init_args" to instantiate
            the dataset
        :param preprocesses: the list of different preprocessing to do
        :param batch_size: the number of examples for a batch
        :param shuffle: whether to shuffle or not the train dataset.
        :param device: which device to use
        :param num_workers: the number of workers for the DataLoader.

        Attributes:
        all_shuffle : dictionary of shuffle value for training, validation and test sets.
        datasets: dictionary with Dataset for training, validation and test sets.
        """
        super().__init__()
        self.dataset_init = dataset_init
        self.preprocesses = preprocesses
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.datasets: Dict[str, Dataset] = {
            "train": Dataset(),
            "valid": Dataset(),
            "test": Dataset(),
        }
        self.all_shuffle = {"train": True, "valid": False, "test": False}
        self.prepare_data(*args, **kwargs)

    @staticmethod
    def instantiate_dataset(dataset_init: Dict, *args, **kwargs) -> Dataset:
        """Instantiate the dataset with the given arguments.

        Args:
             :param dataset_init: Dict of the form {'class_path': ..., 'init_args': ...}
        :return an instance of the Dataset
        """
        dataset_init["init_args"] = {**dataset_init.get("init_args", {}), **kwargs}
        return instantiate_class_from_init(init=dataset_init)

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """Init the train, valid and test datasets.
            Should fill the dictionary self.datasets
            Should download the data if necessary.

        Example :
            self.datasets['train'] = DatasetAbstract(*args, **kwargs)
            self.datasets['valid'] = DatasetAbstract(*args, **kwargs)
            self.datasets['test'] = DatasetAbstract(*args, **kwargs)

        """
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch, *args, **kwargs):
        """Collate function to merge element to create a batch."""
        return None

    def _get_dataloader_from_split(self, split: str) -> DataLoader:
        """
        Returns the dataloader from the self.datasets with the needed arguments.

        :param split: which dataset (train, valid or test)
        :return a dataloader with the hyperparameters from the initialisation
        :raise NotImplementedError if the self.datasets has not been implemented
        """
        # Check the class of 'collate_fn' : if this is from abstract class, return None
        if self.collate_fn.__qualname__.split(".")[0] == "DataModuleAbstract":
            collate_fn = None
        else:
            collate_fn = self.collate_fn
        if split in self.datasets:
            return DataLoader(
                self.datasets[split],  # type : ignore
                batch_size=self.batch_size,
                shuffle=self.all_shuffle.get(split, False),
                collate_fn=collate_fn,
                num_workers=self.num_workers,
            )
        else:
            raise NotImplementedError(f"Split not in dataset keys : {split}")

    def train_dataloader(self):
        """Return the training dataloader."""
        return self._get_dataloader_from_split("train")

    def val_dataloader(self):
        """Return the validation dataloader."""
        return self._get_dataloader_from_split("valid")

    def test_dataloader(self):
        """Return the testing dataloader."""
        return self._get_dataloader_from_split("test")
