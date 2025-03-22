"""Class to define the different logging functions."""
from abc import abstractmethod
from typing import Any


class LoggerAbstract:
    def __init__(self, *args, **kwargs):
        self.init_logger(*args, **kwargs)

    @abstractmethod
    def init_logger(self, *args, **kwargs):
        """Init the logger"""
        raise NotImplementedError

    @abstractmethod
    def log_loss(self, split: str, loss: Any, *args, **kwargs):
        """Log the loss."""
        raise NotImplementedError

    @abstractmethod
    def log_metric(self, split: str, score: Any, metric_name: str, *args, **kwargs):
        """Log the metric."""
        raise NotImplementedError

    @abstractmethod
    def log_metric_end_epoch(
        self, split: str, score: Any, metric_name: str, *args, **kwargs
    ):
        """Log at the end of epoch."""
        raise NotImplementedError
