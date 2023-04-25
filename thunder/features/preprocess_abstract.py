"""Preprocess abstract class to define the main function to use."""
from abc import abstractmethod

from helper.pytorch_helper.thunder.data.sample import Sample


class PreprocessAbstract:
    """Class that represents the different preprocessing."""

    @abstractmethod
    def transform(self, obj: Sample, *args, **kwargs):
        """Preprocess the data."""
        raise NotImplementedError
