"""Abstract PytorchLightningModule class."""
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from loguru import logger
from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall

from thunder.enum.name_to_fn import NAME_TO_LOSS, NAME_TO_OPTIMIZER
from thunder.loggers.default_logger import DefaultLogger
from thunder.loggers.logger_abstract import LoggerAbstract


class AbstractPlModule(LightningModule):
    """
    Class that defines all the needed functions to be used for deep learning approches.
    """
    def __init__(
        self,
        model: nn.Module,
        custom_loggers: Optional[List[LoggerAbstract]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialise the model, hyperparameters, loss.
        :param model: the pytorch model to use
        :param custom_loggers: a list of custom loggers.
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.model = model
        self.custom_loggers = custom_loggers if custom_loggers is not None else [DefaultLogger()]
        # Variables to fill
        self.metrics_dict_train: Dict[str, Any] = {}
        self.metrics_dict_val: Dict[str, Any] = {}
        self.metrics_dict_test: Dict[str, Any] = {}
        self.config_path = None  # Used only to keep track of the path of the config file
        # Setups
        self.setup_hp(*args, **kwargs)
        self.setup_loss(*args, **kwargs)
        self.setup_metrics(*args, **kwargs)

    def setup_loss(self,loss_name: str, *args, **kwargs):
        """Set the loss according to a name.
            :param loss_name: the name of the loss.
        """
        loss = NAME_TO_LOSS.get(loss_name, None)
        if loss is None:
            raise NotImplementedError("LOSS NOT FOUND")
        self.loss = loss()

    def forward(self, x, y):
        output = self.model(x)
        logits = F.log_softmax(output, dim=1)
        return logits

    def training_step(self, batch, batch_nb):
        """Compute the logits and the loss."""
        x, y = self.interpret_batch(batch)
        logits = self.forward(x, y)
        loss = self.loss(logits, y)
        preds = self.postprocess_eval(logits, dim=1)
        self.log_metrics(y, preds, loss, "train")
        return loss

    def interpret_batch(self, batch):
        """Convert batch to x and y"""
        x, y = batch
        return x, y

    def setup_hp(self, lr: float,optimizer_name: str, *args, **kwargs):
        """Set the hyperparameters.
        Args:
            :param lr: learning rate
            :optimizer_name: the name of the optimizer.
        """
        self.lr = lr
        self.optimizer_class = NAME_TO_OPTIMIZER.get(optimizer_name, None)

    def configure_optimizers(self):
        """Configure the optimizer """
        if self.optimizer_class is None:
            logger.debug("OPTIMIZER NAME NOT FOUND")
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
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

    def evaluate(self, batch, split: str) -> None:
        """
        :param split: which split (test or validation) for logging
        :return:
        """
        x, y = self.interpret_batch(batch)
        logits = self.forward(x, y)
        loss = self.loss(logits, y)  # type: ignore
        preds = self.postprocess_eval(logits, dim=1)
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

    def _log_metrics_to_logger_end_epoch(self, logger: LoggerAbstract, scores: Dict, split: str):
        """Log metrics at the end of epoch."""
        for metric_name, score in scores.items():
            logger.log_metric_end_epoch(split, score, metric_name, pl_model=self)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

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
