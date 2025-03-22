"""Class that maps the information from config.yaml file."""
import os.path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from omegaconf import DictConfig
from lightning import Trainer
from torch import nn

from cthunder.config.config_helper_abstract import ConfigHelperAbstract
from cthunder.data.datamodule_abstract import DataModuleAbstract
from cthunder.data.dataset_abstract import DatasetAbstract

from cthunder.features.preprocess_abstract import PreprocessAbstract
from cthunder.loggers.logger_abstract import LoggerAbstract
from cthunder.pl_model.abstract_pl_model import AbstractPlModule
from cthunder.utils.utils import open_yml, save_to_yaml, instantiate_class_from_init


class ConfigHelperYAML(ConfigHelperAbstract):
    """Class that return the main elements from the yaml config file."""

    def __init__(self, config_path: Union[str, DictConfig], *args, **kwargs):
        """
        Load the config.yml or config.yaml file.
        :param path_to_config: the path to the yaml config file or a dict with the given parameters
        """
        super().__init__(config_path, *args, **kwargs)
        self.init_path()

    def init_path(self):
        """Initialise the path where to search the instances of class."""
        self.path = self.config.get("path", "")

    def read_config(self, config_path: Union[str, DictConfig]):
        """Read the content of the config file."""
        if isinstance(config_path, str):
            config_file = open_yml(config_path)
        elif isinstance(config_path, DictConfig):
            config_file = config_path
        else:
            logger.debug("NO CONFIGURATION PATH")
            return None
        if config_file is not None:
            return config_file
        else:
            logger.debug("CONFIG PATH NOT READABLE")

    def _get_value_from_keys(
        self, keys: Union[List[str], str], config: Optional[Dict] = None
    ) -> Any:
        """
        Return the value from nested keys.
        :param keys: List of keys in the config file
        :param config: the dictionary where to search the elements.
        :return: the value corresponding.
        """
        config = config if config is not None else self.config
        if isinstance(keys, str):
            output = config.get(keys, None)
            return dict(output) if output is not None else output
        elif isinstance(keys, list):
            if len(keys) == 1:
                output = config.get(keys[0], None)
                return dict(output) if output is not None else output
            else:
                output = self._get_value_from_keys(
                    keys[1:],
                    config.get(keys[0], {}),
                )
                return dict(output) if output is not None else output
        else:
            raise AttributeError(
                "KEYS TO SEARCH IN CONFIG FILE BAD TYPE : SHOULD BE STR OR LIST, "
                f"AND GOT VALUE OF TYPE {type(keys)} : {keys}"
            )

    def get_preprocesses(self) -> Optional[List[PreprocessAbstract]]:
        """
        Return the preprocesses from the yaml config file.
        :return: a list of preprocesses initialised with they parameters
        """
        preprocesses: List[PreprocessAbstract] = []
        list_preprocesses = self._get_value_from_keys("preprocesses", None)
        if list_preprocesses is None:
            return preprocesses
        for name in list_preprocesses:
            params = self._get_value_from_keys(["preprocesses", name])
            class_inst = instantiate_class_from_init(init=params)
            if class_inst is not None:
                preprocesses.append(class_inst)
        return preprocesses

    def get_dataset(self) -> Optional[DatasetAbstract]:
        """
        Return the Dataset class from the yaml config file.
        This function doesn't use the hyperparameters of the dataset.
        :return: the Dataset instantiate.
        """
        dataset_params = self._get_value_from_keys(["dataset"])
        if dataset_params is None:
            return None
        dataset_class = instantiate_class_from_init(init=dataset_params)
        if dataset_class is not None:
            return dataset_class
        else:
            raise AttributeError("FAIL TO LOAD DATASET")

    def get_datamodule(self) -> DataModuleAbstract:
        """
        Return the DataLoader class from the yaml config file.
        It initialises the datasets and preprocesses
        It gets also the hyperparameters of the dataset.
        :return: the DataLoader instantiate.
        """
        # Get the preprocesses
        preprocess = self.get_preprocesses()
        dataset_params = self._get_value_from_keys(["dataset"])
        # Get the dataloader params and class
        datamodule_params = self._get_value_from_keys(["datamodule"])
        # Fill the last parameters : preprocesses and dataset class
        params = {"preprocesses": preprocess, "dataset_init": dataset_params}
        datamodule_params["init_args"] = {
            **datamodule_params.get("init_args", {}),
            **params,
        }
        datamodule = instantiate_class_from_init(init=datamodule_params)
        return datamodule

    def get_model(self) -> Optional[nn.Module]:
        """
        Return the pytorch model from yaml.
        :return pytorch model from class in the config file.
        """
        model_params = self._get_value_from_keys("model")
        if model_params is None:
            return None
        model = instantiate_class_from_init(model_params)
        return model

    def get_pl_model(self) -> Optional[AbstractPlModule]:
        """
        Return the pytorch lightning model from yaml information.
        :return: the pytorch lightning model from the config file.
        """
        pl_model_params = self._get_value_from_keys(["pl_model"])
        if pl_model_params is None:
            return None
        model = self.get_model()
        pl_model_params["init_args"] = {
            **pl_model_params.get("init_args", {}),
            **{"model": model},
        }
        pl_model = instantiate_class_from_init(pl_model_params)
        return pl_model

    def get_trainer(self) -> Trainer:
        """Return the trainer form yaml file."""
        params = self._get_value_from_keys(["trainer"])
        if params is not None:
            trainer = instantiate_class_from_init(params)
        else:
            trainer = Trainer()
        return trainer

    def get_log_dir(self) -> Optional[str]:
        """Return the directory where should be stored the logs."""
        log_dir = self._get_value_from_keys(["log_dir"])
        return log_dir

    def get_checkpoint_dir(self) -> Optional[str]:
        """Return where are stored the checkpoints."""
        chkpt_dir = self._get_value_from_keys(["checkpoint_dir"])
        return chkpt_dir

    def get_model_name(self) -> str:
        """Return the name of the model if specified."""
        model_name = self._get_value_from_keys(["model_name"])
        model_name = "" if model_name is None else model_name
        return model_name

    def get_logger(self) -> List[LoggerAbstract]:
        """Return a list of logger."""
        pass

    def save_checkpoints(self, log_path: str, checkpoint_path: str) -> None:
        """
        Save the hyperparameters file and add the model checkpoint.
        :param log_path: path to the logs
        :param checkpoint_path: path where is stored the model checkpoint.
        """
        params = self.config.get("pl_model", {}).get("init_args", {})
        params["checkpoint_path"] = checkpoint_path
        path_to_save = os.path.join(log_path, "config.yml")
        save_to_yaml(path_to_save, self.config)
        logger.debug(f"YAML CONFIG SAVED IN : {path_to_save}")

    def get_experiment_class(self, *args, **kwargs) -> Any:
        """
        Return the class given by the name in the config file
        """
        params = self.config.get("experiment", None)
        if params is None:
            raise NotImplementedError("NO EXPERIMENT CLASS FOUND IN CONFIG")
        params = dict(params)
        params["init_args"] = {
            **params.get("init_args", {}),
            **{"config_path": self.config_path},
        }
        experiment = instantiate_class_from_init(init=params)
        return experiment
