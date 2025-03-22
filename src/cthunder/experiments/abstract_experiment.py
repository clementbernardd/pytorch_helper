"""Abstract Class for Experiment using Pytorch Lightning."""
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from lightning.pytorch import Callback, LightningDataModule, Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from cthunder.config.config_helper_abstract import ConfigHelperAbstract
from cthunder.config.config_helper_python import ConfigHelperPython
from cthunder.config.config_helper_yaml import ConfigHelperYAML
from cthunder.data.datamodule_abstract import DataModuleAbstract
from cthunder.features.preprocess_abstract import PreprocessAbstract
from cthunder.loggers.logger_abstract import LoggerAbstract
from cthunder.pl_model.abstract_pl_model import AbstractPlModule
from cthunder.utils.utils import save_json



@dataclass
class AbstractExperiment:
    """Abstract experiment to setup the training, evaluation or inference.
    variables:
        :param pl_model: the pytorch lightning model
        :param datamodule: the datamodule with the different dataloaders
        :param preprocesses: the different preprocesses to use
        :param postprocesses: the different postprocesses to use
        :param callbacks: the pytorch lightning callbacks
        :param trainer: the pytorch lightning trainer
        :param config: the config helper class to convert yaml hp

    """

    pl_model: Optional[Union[LightningModule, AbstractPlModule]] = None
    datamodule: Optional[Union[LightningDataModule, DataModuleAbstract]] = None
    preprocesses: Optional[List[PreprocessAbstract]] = None
    postprocesses: Optional[List[PostprocessAbstract]] = None
    callbacks: Optional[List[Callback]] = None
    trainer: Optional[Trainer] = None
    config: Optional[ConfigHelperAbstract] = None

    def __init__(self, *args, **kwargs):
        """
        Abstract experiment. Setup for the training, evaluation and inference.
        :param args:
        :param kwargs:
        """
        self.custom_loggers = None
        self.init_config_helper(*args, **kwargs)
        self.set_seed(seed=77)

    def set_seed(self, seed: int = 77):
        """
        Set the seed for numpy, random, torch
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def init_config_helper(
        self,
        config_class: Type[ConfigHelperAbstract] = None,
        config_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Initialise the config helper to get the hyperparameters.
        Args:
            :param config_class: the class to instantiate the config helper
            :param config_path: path to the configuration file.
        """
        if config_path is not None:
            if isinstance(config_path, DictConfig):
                self.config_path = None
                self.config = ConfigHelperYAML(config_path)
                return None
            elif config_path.endswith(".py"):
                config_class = ConfigHelperPython
            elif config_path.endswith(".yml") or config_path.endswith(".yaml"):
                config_class = ConfigHelperYAML
            else:
                logger.debug(f"BAD EXTENSION FOR CONFIG FILE")
                return None
        self.config_path = config_path
        # Initialise the config helper with the path
        self.config = config_class(self.config_path)

    def train(self, *args, **kwargs):
        """Training loop."""
        self.setup_train(*args, **kwargs)
        self.trainer.fit(self.pl_model, self.datamodule)
        self.post_train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """Do the evaluation process."""
        self.setup_eval(*args, **kwargs)
        self.trainer.test(self.pl_model, self.datamodule)
        self.post_eval(*args, **kwargs)

    def validate(self, *args, **kwargs):
        """Do the validation process."""
        self.setup_eval(*args, **kwargs)
        self.trainer.validate(self.pl_model, self.datamodule)
        self.post_val(*args, **kwargs)

    def predict(self, path_to_save: str, *args, **kwargs):
        self.setup_eval(*args, **kwargs)
        if self.trainer is not None and self.datamodule is not None:
            predictions = self.trainer.predict(
                self.pl_model, self.datamodule.test_dataloader()
            )
            predictions = self._convert_preds_to_lists(predictions)
            save_json(predictions, path_to_save)  # type: ignore
        else:
            logger.debug("TRAINER IS NOT : COULD NOT PREDICT")

    def _convert_preds_to_lists(self, predictions: Any):
        """Convert the predictions from the batches to list."""
        y, preds = [x["y"] for x in predictions], [x["preds"] for x in predictions]
        y = np.squeeze(y).reshape(-1).tolist()
        preds = np.squeeze(preds).reshape(-1).tolist()
        output = {"y_true": y, "y_pred": preds}
        return output

    def post_val(self, *args, **kwargs):
        """Steps to do after validation."""
        pass

    def post_eval(self, *args, **kwargs):
        """Steps to do after evaluation (save predictions, ...)"""
        pass

    def post_train(self, *args, **kwargs):
        """Save the config file and change the checkpoint path."""
        log_path = os.path.join(
            self.trainer.logger.save_dir,
            "lightning_logs",
            f"version_{self.trainer.logger.version}",
        )
        self.config.save_checkpoints(log_path, self.checkpoints.best_model_path)

    def setup_eval(self, *args, **kwargs):
        """Do the setup for the evaluation process."""
        self.setup_train(*args, **kwargs)
        self.pl_model.eval()

    def setup_val(self, *args, **kwargs):
        """Do the setup for the validation."""
        self.setup_train(*args, **kwargs)

    def setup_train(self, *args, **kwargs):
        """Initialise the parameters for the training."""
        self.init_preprocesses(*args, **kwargs)
        self.init_postprocesses(*args, **kwargs)
        self.init_datamodule(*args, **kwargs)
        self.init_logger(*args, **kwargs)
        self.init_pl_model(*args, **kwargs)
        self.init_callbacks(*args, **kwargs)
        self.setup_trainer(*args, **kwargs)

    def init_pl_model(
        self,
        pl_model: Optional[AbstractPlModule] = None,
        custom_loggers: Optional[List[LoggerAbstract]] = None,
        *args,
        **kwargs,
    ):
        """Initialise the pytorch lightning model.
        :param pl_model: the pytorch lightning class that does the training process.
        """
        self.pl_model = self._init_module(pl_model, "pl_model", AbstractPlModule)
        if self.pl_model is not None:
            self.pl_model.config_path = self.config_path  # type: ignore

    def _init_module(
        self, module: Any, name: str, class_instance: Any = None, *args, **kwargs
    ):
        """
        Return the value if exists in the config file.
        :param module: the object to return if not None
        :param name: name of the element to return.
        :param class_instance: the instance of the class that module should be
        :return:
        """
        condition = (
            isinstance(module, class_instance)
            if class_instance is not None
            else (module is not None)
        )
        if condition:
            return module
        else:
            if self.config is not None:
                assert name in self.config.name_to_function
                return self.config.name_to_function.get(name)(*args, **kwargs)  # type: ignore
            else:
                logger.debug("METHOD NOT FOUND IN CONFIG HELPER")

    def init_datamodule(
        self, datamodule: Optional[DataModuleAbstract] = None, *args, **kwargs
    ):
        """
        Initialise the dataloaders.
        :param datamodule: the datamodule that returns the different dataloaders.
        """
        self.datamodule = self._init_module(
            datamodule, "datamodule", DataModuleAbstract
        )

    def init_preprocesses(
        self, preprocesses: Optional[List[PreprocessAbstract]] = None, *args, **kwargs
    ):
        """Initialise the preprocessing.
        Args:
            :param preprocesses: a list of different preprocesses.
        """
        self.preprocesses = self._init_module(preprocesses, "preprocesses")

    def init_postprocesses(
        self, postprocesses: Optional[List[PostprocessAbstract]] = None, *args, **kwargs
    ):
        """Initialise the postprocesses.
        Args:
             :param postprocesses: a list of different postprocesses.
        """
        self.postprocesses = self._init_module(postprocesses, "postprocesses")

    def init_callbacks(self, *args, **kwargs):
        """Initialise the callbacks from pytorch lightning module."""
        self.callbacks = []

    def init_logger(self, log_dir: Optional[str] = None, *args, **kwargs):
        """
        Initialise the logger.
        :param log_dir: directory where will be stored the logs.
        """
        log_dir = self._init_module(log_dir, "log_dir")
        log_dir = log_dir if log_dir is not None else "logs"
        logger = CSVLogger(save_dir=log_dir)
        return logger

    def setup_trainer(self, trainer_params: Optional[Dict] = None, *args, **kwargs):
        """
        Set the trainer (from pytorch lightning module)
        :param trainer_params: parameters for the trainer.
        """
        self.checkpoints = self.init_model_checkpoint_callback(*args, **kwargs)
        progress_bar = self.init_progress_bar(*args, **kwargs)
        callbacks = [*self.callbacks, *[self.checkpoints, progress_bar]]  # type: ignore
        logger = self.init_logger(*args, **kwargs)
        if trainer_params is not None:
            # Initialise with current parameters
            trainer = Trainer(**trainer_params)
        elif trainer_params is None and self.config is not None:
            # Initialise with the config helper
            trainer = self.config.get_trainer()  # type: ignore
        else:
            raise AttributeError("TRAINER CAN'T BE INSTANTIATE")
        trainer.callbacks = callbacks
        trainer.logger = logger
        self.trainer = trainer

    def init_progress_bar(self, *args, **kwargs):
        """Init the progress bar."""
        return TQDMProgressBar()

    def init_model_checkpoint_callback(
        self,
        checkpoint_dir: Optional[str] = "models",
        monitor_loss: Optional[str] = "val_loss",
        model_name: Optional[str] = "example",
        save_top_k: Optional[int] = 1,
        *args,
        **kwargs,
    ):
        """Initialise the callback that specify where are stored the checkpoints."""
        if self.config is not None:
            checkpoint_dir = self._init_module(checkpoint_dir, "checkpoint_dir")
            model_name = self._init_module(model_name, "model_name")
            monitor_loss = "val_loss" if monitor_loss is None else monitor_loss
            save_top_k = 1 if save_top_k is None else save_top_k
        filename = f"{model_name}-{{{monitor_loss}:.2f}}-{{epoch}}"
        return ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=filename,
            monitor=monitor_loss,
            mode="min" if "loss" in monitor_loss else "max",  # type: ignore
            save_top_k=save_top_k,  # type: ignore
        )

    def setup_inference(self, *args, **kwargs):
        """Setup the model for inference."""
        pass

    def infer(self, *args, **kwargs):
        """Do the inference."""
        pass

