"""Dataset abstract file for Pytorch Dataset implementation."""
from abc import abstractmethod
from typing import List, Optional

from torch.utils.data import Dataset

from cthunder.features.preprocess_abstract import PreprocessAbstract
from cthunder.data.sample import Sample


class DatasetAbstract(Dataset):
    """Dataset abstract class that defines the main functions to use."""

    def __init__(
        self,
        preprocesses: Optional[List[PreprocessAbstract]] = None,
        device: Optional[str] = None,
        *args,
        **kwargs
    ):
        """ """
        self.preprocesses = preprocesses
        self.device = device
        self.setup_data(*args, **kwargs)

    @abstractmethod
    def read_data(self, item: int) -> Sample:
        """Return one element of the data (a line for a dataframe for instance)"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def setup_data(self, *args, **kwargs) -> None:
        """Initialise the data by loading the data (like dataframe, etc)"""
        raise NotImplementedError

    def preprocess(self, sample: Sample, *args, **kwargs) -> Sample:
        """
        Do the preprocessing.
        :param sample:
        :param args:
        :param kwargs:
        :return:
        """
        if self.preprocesses is not None:
            for process in self.preprocesses:
                sample = process.transform(sample, *args, **kwargs)
        return sample

    def __getitem__(self, item):
        """Should use at least 'self.preprocess(sample)'."""
        sample = self.read_data(item)
        sample = self.preprocess(sample)
        x, y = sample.read_to_torch(self.device)
        return x, y
