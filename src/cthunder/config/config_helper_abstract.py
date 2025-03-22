"""Class that convert a file with hyperparameters to python readable variables."""
from abc import abstractmethod
from typing import Any, List, Optional, Union

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn

from cthunder.data.datamodule_abstract import DataModuleAbstract
from cthunder.data.dataset_abstract import DatasetAbstract
from cthunder.features.preprocess_abstract import PreprocessAbstract
from cthunder.loggers.logger_abstract import LoggerAbstract
from cthunder.pl_model.abstract_pl_model import AbstractPlModule


class ConfigHelperAbstract:
    def __init__(self, config_path: Union[str, DictConfig], *args, **kwargs):
        """Initialise the helper with a file.

        Args:
            :param config_path: the path to a configuration file.
        """
        self.config_path = config_path
        self.config = self.read_config(self.config_path)
        # Dictionary to convert names to given functions.
        self.name_to_function = {
            "preprocesses": self.get_preprocesses,
            "dataset": self.get_dataset,
            "datamodule": self.get_datamodule,
            "model": self.get_model,
            "pl_model": self.get_pl_model,
            "trainer": self.get_trainer,
            "log_dir": self.get_log_dir,
            "checkpoint_dir": self.get_checkpoint_dir,
            "model_name": self.get_model_name,
            "logger": self.get_logger,
        }

    @abstractmethod
    def read_config(self, config_path: Union[str, DictConfig]) -> Any:
        """Read the content of the config file."""
        raise NotImplementedError

    @abstractmethod
    def get_preprocesses(self) -> Optional[List[PreprocessAbstract]]:
        """
        Return the preprocesses from the config file.
        :return: a list of preprocesses initialised with they parameters
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self) -> Optional[DatasetAbstract]:
        """
        Return the Dataset class from the config file.
        :return: the Dataset instantiate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_datamodule(self) -> DataModuleAbstract:
        """
        Return the DataLoader class from the config file.
        It initialises the datasets and preprocesses
        :return: the DataLoader instantiate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> Optional[nn.Module]:
        """
        Return the pytorch model from config file.
        :return pytorch model from class in the config file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_pl_model(self) -> Optional[AbstractPlModule]:
        """
        Return the pytorch lightning model from config file.
        :return: the pytorch lightning model from the config file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trainer(self) -> Optional[Trainer]:
        """Return the trainer from config file."""
        raise NotImplementedError

    @abstractmethod
    def get_log_dir(self) -> Optional[str]:
        """Return the directory where should be stored the logs."""
        raise NotImplementedError

    @abstractmethod
    def get_checkpoint_dir(self) -> Optional[str]:
        """Return where are stored the checkpoints."""
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model if specified."""
        raise NotImplementedError

    @abstractmethod
    def get_logger(self) -> List[LoggerAbstract]:
        """Return a list of logger."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoints(self, log_path: str, checkpoint_path: str) -> None:
        """
        Save the hyperparameters file and add the model checkpoint.
        :param log_path: path to the logs
        :param checkpoint_path: path where is stored the model checkpoint.
        """
        raise NotImplementedError

    @staticmethod
    def get_experiment_class(self) -> Any:
        """
        Return the class given by the name in the config file
        """
        raise NotImplementedError
