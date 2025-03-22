import inspect
import os
import shutil
from typing import List, Optional

from loguru import logger
from pytorch_lightning import Trainer
from torch import nn

from cthunder.config.config_helper_abstract import ConfigHelperAbstract
from cthunder.data.datamodule_abstract import DataModuleAbstract
from cthunder.data.dataset_abstract import DatasetAbstract
from cthunder.features.preprocess_abstract import PreprocessAbstract
from cthunder.loggers.logger_abstract import LoggerAbstract
from cthunder.pl_model.abstract_pl_model import AbstractPlModule


class ConfigHelperPython(ConfigHelperAbstract):
    """Class that return the main elements from the python config file."""

    def __init__(self, config_path: Optional[str], *args, **kwargs):
        """
        Load the config.py file.
        :param path_to_config: the path to the python config file.
        """
        super().__init__(config_path, *args, **kwargs)

    def read_config(self, config_path: Optional[str]):
        """Read the content of the config file."""
        pass

    def get_preprocesses(self) -> Optional[List[PreprocessAbstract]]:
        """
        Return the preprocesses from the config file.
        :return: a list of preprocesses initialised with they parameters
        """
        classes = HPKind.PREPROCESS.get("classes", None)
        params = HPKind.PREPROCESS.get("params", None)
        if classes is not None:
            preprocesses = []
            for classe, param in zip(classes, params):
                preprocesses.append(classe(**param))
            return preprocesses
        else:
            logger.debug("CLASSES FOR PREPROCESSING IS None")
            return None

    def get_dataset(self) -> Optional[DatasetAbstract]:
        """
        Return the Dataset class from the config file.
        :return: the Dataset instantiate.
        """
        dataset_class = HPKind.DATASET.get("class", None)
        if dataset_class is not None:
            return dataset_class
        else:
            raise AttributeError("FAIL TO LOAD DATASET")

    def get_datamodule(self) -> DataModuleAbstract:
        """
        Return the DataLoader class from the config file.
        It initialises the datasets and preprocesses
        :return: the DataLoader instantiate.
        """
        # Get the preprocesses
        preprocess = self.get_preprocesses()
        # Get the dataset class
        dataset_class = self.get_dataset()
        # Get the dataset params
        dataset_params = HPKind.DATASET.get("params", {})
        # Get the dataloader params and class
        datamodule_class = HPKind.DATAMODULE.get("class", None)
        datamodule_params = HPKind.DATAMODULE.get("params", {})
        # Fill the last parameters : preprocesses and dataset class
        datamodule_params["preprocesses"] = preprocess
        datamodule_params["dataset_class"] = dataset_class
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
        Return the pytorch model from config file.
        :return pytorch model from class in the config file.
        """
        model_class = HPKind.MODEL.get("class", None)
        params = HPKind.MODEL.get("params", {})
        if model_class is not None:
            return model_class(**params)
        else:
            raise AttributeError("MODEL NOT LOADED")

    def get_pl_model(self) -> Optional[AbstractPlModule]:
        """
        Return the pytorch lightning model from config file information.
        :return: the pytorch lightning model from the config file.
        """
        pl_model_class = HPKind.PLMODEL.get("class", None)
        if pl_model_class is not None:
            params = HPKind.PLMODEL.get("params", {})
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

    def get_trainer(self) -> Optional[Trainer]:
        """Return the trainer form yaml file."""
        params = HPKind.TRAINER.get("params", {})
        if params is not None:
            trainer = Trainer(**params)
            return trainer
        else:
            logger.debug("Trainer not initialized")
            return None

    def get_log_dir(self) -> Optional[str]:
        """Return the directory where should be stored the logs."""
        log_dir = HPKind.LOG_DIR
        return log_dir

    def get_checkpoint_dir(self) -> Optional[str]:
        """Return where are stored the checkpoints."""
        chkpt_dir = HPKind.CHECKPOINT_DIR
        return chkpt_dir

    def get_model_name(self) -> str:
        """Return the name of the model if specified."""
        model_name = HPKind.MODEL_NAME
        model_name = "" if model_name is None else model_name
        return model_name

    def get_logger(self) -> List[LoggerAbstract]:
        """Return a list of logger."""
        loggers = HPKind.LOGGERS
        return loggers

    def save_checkpoints(self, log_path: str, checkpoint_path: str) -> None:
        """
        Save the hyperparameters file and add the model checkpoint.
        :param log_path: path to the logs
        :param checkpoint_path: path where is stored the model checkpoint.
        """
        path_to_save = os.path.join(log_path, "config.py")
        shutil.copyfile(inspect.getfile(HPKind), path_to_save)
        logger.debug(f"PYTHON CONFIG SAVED IN : {path_to_save}")
