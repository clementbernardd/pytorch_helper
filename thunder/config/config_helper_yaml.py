"""Class that maps the information from config.yaml file."""
import os.path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from omegaconf import DictConfig
from lightning import Trainer
from lightning.pytorch.cli import instantiate_class
from torch import nn

from thunder.config.config_helper_abstract import ConfigHelperAbstract
from thunder.data.datamodule_abstract import DataModuleAbstract
from thunder.data.dataset_abstract import DatasetAbstract

from thunder.features.preprocess_abstract import PreprocessAbstract
from thunder.loggers.logger_abstract import LoggerAbstract
from thunder.pl_model.abstract_pl_model import AbstractPlModule
from thunder.utils.utils import instantiate_class_from_name, open_yml, save_to_yaml


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
            return config.get(keys, None)
        elif isinstance(keys, list):
            if len(keys) == 1:
                return config.get(keys[0], None)
            else:
                return self._get_value_from_keys(
                    keys[1:],
                    config.get(keys[0], {}),
                )
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
            preprocess_class = self._get_value_from_keys(["preprocesses", name, "class"])
            class_inst = instantiate_class_from_name(self.path, preprocess_class)
            params = self._get_value_from_keys(["preprocesses", name, "params"])
            params = params.copy() if params is not None else {}
            if class_inst is not None:
                preprocesses.append(class_inst(**params))
        return preprocesses

    def get_dataset(self) -> Optional[DatasetAbstract]:
        """
        Return the Dataset class from the yaml config file.
        This function doesn't use the hyperparameters of the dataset.
        :return: the Dataset instantiate.
        """
        dataset_path = self._get_value_from_keys(["dataset", "class"])
        if dataset_path is None:
            return None
        dataset_class = instantiate_class_from_name(self.path, dataset_path)
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
        # Get the dataset class
        dataset_class = self.get_dataset()
        dataset_params = self._get_value_from_keys(["dataset", "params"])
        dataset_params = {} if dataset_params is None else dataset_params
        # Get the dataloader params and class
        datamodule_path = self._get_value_from_keys(["datamodule", "class"])
        datamodule_class = instantiate_class_from_name(self.path, datamodule_path)
        datamodule_params = dict(self._get_value_from_keys(["datamodule", "params"]).copy())
        # Fill the last parameters : preprocesses and dataset class
        datamodule_params["preprocesses"] = preprocess
        datamodule_params["dataset_class"] = dataset_class
        breakpoint()
        datamodule_params = {**dataset_params, **datamodule_params}
        if datamodule_class is not None:
            dataloader = datamodule_class(
                **datamodule_params,
            )
        else:
            raise AttributeError("FAIL TO LOAD DATALAODER")
        return dataloader

    def get_model(self) -> Optional[nn.Module]:
        """
        Return the pytorch model from yaml.
        :return pytorch model from class in the config file.
        """
        model_path = self._get_value_from_keys(["model", "class"])
        if model_path is None:
            return None
        model_class = instantiate_class_from_name(self.path, model_path)
        params = self._get_value_from_keys(["model", "params"]).copy()
        if model_class is not None:
            return model_class(**params)
        else:
            raise AttributeError("MODEL NOT LOADED")

    def get_pl_model(self) -> Optional[AbstractPlModule]:
        """
        Return the pytorch lightning model from yaml information.
        :return: the pytorch lightning model from the config file.
        """
        pl_model_path = self._get_value_from_keys(["pl_model", "class"])
        if pl_model_path is None:
            return None
        pl_model_class = instantiate_class_from_name(self.path, pl_model_path)
        if pl_model_class is not None:
            params = dict(self._get_value_from_keys(["pl_model", "params"]).copy())
            model = self.get_model()
            params["model"] = model
            checkpoint = params.get("checkpoint_path", None)
            # Load weight if checkpoint is in the params
            if checkpoint is not None and os.path.exists(checkpoint):
                pl_model = pl_model_class.load_from_checkpoint(**params)
                logger.debug(f"MODEL LOADED FROM {checkpoint}")
            else:
                pl_model = pl_model_class(**params)
            return pl_model
        else:
            raise AttributeError("PL MODEL NOT LOADED")

    def get_trainer(self) -> Trainer:
        """Return the trainer form yaml file."""
        params = self._get_value_from_keys(["trainer", "params"])
        if params is not None:
            trainer = Trainer(**params)
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
        params = self.config.get("pl_model", {}).get("params", {})
        params["checkpoint_path"] = checkpoint_path
        path_to_save = os.path.join(log_path, "config.yml")
        save_to_yaml(path_to_save, self.config)
        logger.debug(f"YAML CONFIG SAVED IN : {path_to_save}")

    def get_experiment_class(self) -> Any:
        """
        Return the class given by the name in the config file
        """
        class_name = self.config.get("experiment", {}).get("class", None)
        if class_name is None:
            logger.debug("NO EXPERIMENT CLASS FOUND IN CONFIG")
        experiment_ = instantiate_class_from_name(self.path, class_name)
        return experiment_