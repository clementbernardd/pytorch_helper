"""Abstract PytorchLightningModule class."""
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall

from cthunder.features.postprocess_abstract import PostprocessAbstract
from cthunder.loggers.default_logger import DefaultLogger
from cthunder.loggers.logger_abstract import LoggerAbstract
from cthunder.utils.utils import instantiate_class_from_init


class AbstractPlModule(LightningModule):
    """
    Class that defines all the needed functions to be used for deep learning approches.
    """

    def __init__(
        self,
        model: nn.Module,
        custom_loggers: Optional[List[LoggerAbstract]] = None,
        loss_args: Any = None,
        postprocesses: Optional[List[PostprocessAbstract]] = None,
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialise the model, hyperparameters, loss.
        :param model: the pytorch model to use
        :param custom_loggers: a list of custom loggers.
        :param loss_args: the path to the loss to use for the training.
        :param checkpoint_path: the path to the checkpoint to load.
        """
        super().__init__()
        self.model = model
        self.custom_loggers = (
            custom_loggers if custom_loggers is not None else [DefaultLogger()]
        )
        # Variables to fill
        self.metrics_dict_train: Dict[str, Any] = {}
        self.metrics_dict_val: Dict[str, Any] = {}
        self.metrics_dict_test: Dict[str, Any] = {}
        self.config_path = (
            None  # Used only to keep track of the path of the config file
        )
        self.loss = instantiate_class_from_init(loss_args)
        # Setups
        self.setup_hp(*args, **kwargs)
        self.setup_metrics(*args, **kwargs)
        self.postprocesses = postprocesses

    def forward(self, x, y):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_nb):
        """Compute the logits and the loss."""
        x, y = self.interpret_batch(batch)
        logits = self.forward(x, y)
        loss = self.loss(logits, y)
        preds = self.postprocess_eval(logits)
        self.log_metrics(y, preds, loss, "train")
        return loss

    def interpret_batch(self, batch):
        """Convert batch to x and y"""
        x, y = batch
        return x, y

    def setup_hp(self, lr: float, optimizer_args: Dict, scheduler_args: Optional[Dict] = None, *args, **kwargs):
        """Set the hyperparameters.
        Args:
            :param lr: learning rate
            :param optimizer_args: dictionary with the class path and init args
        """
        self.lr = lr
        self.optimizer_args = OmegaConf.to_container(optimizer_args)  # convert to dict
        self.scheduler_args = OmegaConf.to_container(scheduler_args) if scheduler_args is not None else None

    def configure_optimizers(self):
        """Configure the optimizer"""
        if self.optimizer_args is None:
            logger.debug("OPTIMIZER NAME NOT FOUND")
        self.optimizer_args["init_args"]["params"] = self.model.parameters()
        optimizer = instantiate_class_from_init(self.optimizer_args)
        if self.scheduler_args is not None:
            scheduler_args = self.scheduler_args
            scheduler_args["init_args"]["optimizer"] = optimizer
            scheduler = instantiate_class_from_init(self.scheduler_args)
            return [optimizer], [scheduler]
        return optimizer

    def setup_metrics(self, num_classes: int = 2, *args, **kwargs):
        """Set the different metrics.

        Args:
             :param num_classes: number of labels
        """
        self.metrics_dict_train = self._get_metrics(num_classes)
        self.metrics_dict_test = self._get_metrics(num_classes)
        self.metrics_dict_val = self._get_metrics(num_classes)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Do prediction for the given dataset"""
        x, y = self.interpret_batch(batch)
        logits = self.forward(x, y)
        preds = self.postprocess_eval(logits, dim=1).cpu().numpy()
        return {"y": y.cpu().numpy(), "preds": preds}

    def _get_metrics(self, num_classes: int) -> nn.ModuleDict:
        """
        Return a ModuleDict of different metrics.
        Args:
            :param num_classes: number of labels
        """
        task = "binary" if num_classes == 2 else "multiclass"
        return nn.ModuleDict(
            {
                "Accuracy": Accuracy(task=task, num_classes=num_classes),
                "F1 score": F1Score(task=task, num_classes=num_classes),
                "Recall": Recall(task=task, num_classes=num_classes),
                "Precision": Precision(task=task, num_classes=num_classes),
            }
        )

    def postprocess_eval(self, logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Return the argmax of the logits
        :param logits: the output of the forward.
        :param dim: the dimension where to take the argmax.
        :return the argmax for dimension 1.
        """
        preds = torch.argmax(logits, dim=dim)
        return preds

    def evaluate(self, batch, batch_nb, split: str) -> None:
        """
        :param split: which split (test or validation) for logging
        :return:
        """
        x, y = self.interpret_batch(batch)
        logits = self.forward(x, y)
        loss = self.loss(logits, y)  # type: ignore
        preds = self.postprocess_eval(logits)
        self.log_metrics(y, preds, loss, split)

    def log_metrics(self, y, preds, loss, split) -> None:
        """Log the different metrics."""
        if split in ["train", "training"]:
            metrics_dict = self.metrics_dict_train
        elif split in ["val", "valid", "validation"]:
            metrics_dict = self.metrics_dict_val
        elif split in ["test", "testing"]:
            metrics_dict = self.metrics_dict_test
        else:
            logger.debug(f"SPLIT NOT FOUND : {split}")
            return None
        self._log_metrics(y, preds, loss, split, metrics_dict)

    def _compute_scores(self, metric_dict: Dict, preds: torch.Tensor, y: torch.Tensor):
        """Compute the scores."""
        scores = {}
        for metric_name, metric in metric_dict.items():
            score = metric(preds, y)
            scores[metric_name] = score
        return scores

    def _compute_scores_epoch(self, metric_dict: Dict):
        """Compute the score at the end of an epoch. Reset also the scores after."""
        scores = {}
        for metric_name, metric in metric_dict.items():
            score = metric.compute()
            scores[metric_name] = score
            metric.reset()
        return scores

    def _log_metrics(self, y, preds, loss, split, metric_dict) -> None:
        """Log the metric with the given dictionary of metrics."""
        scores = self._compute_scores(metric_dict, preds, y)
        if self.custom_loggers is not None:
            # Loop over the different loggers
            for c_logger in self.custom_loggers:
                # Log the loss. The 'pl_model=self' is useful only for default logger
                c_logger.log_loss(split, loss, pl_model=self)
                # Log the metrics
                self._log_metrics_to_logger(c_logger, split, scores)
                if self.scheduler_args is not None:
                    lr = self.lr_schedulers().get_last_lr()[0]
                    c_logger.log_value("Learning rate", lr)

    def _log_metrics_to_logger(
        self,
        logger: LoggerAbstract,
        split: str,
        scores: Dict,
    ):
        """Log metrics to the custom logger."""
        for metric_name, score in scores.items():
            # Error if the metric isn't in the same device.
            if split == "train":
                # Log only for the training set
                logger.log_metric(split, score, metric_name, pl_model=self)

    def _log_metrics_to_logger_end_epoch(
        self, logger: LoggerAbstract, scores: Dict, split: str
    ):
        """Log metrics at the end of epoch."""
        for metric_name, score in scores.items():
            if isinstance(score, Dict):
                for m_name, c_score in score.items():
                    logger.log_metric_end_epoch(split, c_score, metric_name+"_"+m_name, pl_model=self)
            else:
                logger.log_metric_end_epoch(split, score, metric_name, pl_model=self)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "test")

    def _shared_epoch_end(self, metric_dict: Dict, split: str):
        """Log at the end of the epoch."""
        if self.custom_loggers is not None:
            scores = self._compute_scores_epoch(metric_dict)
            for c_logger in self.custom_loggers:
                self._log_metrics_to_logger_end_epoch(c_logger, scores, split)

    def on_train_epoch_end(self) -> None:
        """Logging at the end of each training epoch."""
        self._shared_epoch_end(self.metrics_dict_train, "train")

    def on_validation_epoch_end(self) -> None:
        """Logging at the end of validation epoch"""
        self._shared_epoch_end(self.metrics_dict_val, "val")

    def on_test_epoch_end(self) -> None:
        """Logging at the end of test epoch"""
        self._shared_epoch_end(self.metrics_dict_test, "test")
