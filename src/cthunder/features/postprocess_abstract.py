"""
Class that implements the abstract interface for the postprocessing.
"""
from abc import abstractmethod

from cthunder.data.sample import Sample


class PostprocessAbstract:
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, obj: Sample):
        raise NotImplementedError
