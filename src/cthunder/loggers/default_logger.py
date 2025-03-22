"""Default logger of pytorch lightning."""
from typing import Any

from cthunder.loggers.logger_abstract import LoggerAbstract


class DefaultLogger(LoggerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_logger(self, *args, **kwargs):
        """Nothing to do for initialisation."""
        pass

    def log_loss(self, split: str, loss: Any, *args, **kwargs):
        """Log the loss."""
        # Get the model from kwargs
        pl_model = kwargs.get("pl_model", None)
        if pl_model is not None:
            pl_model.log(f"{split}_loss", loss, prog_bar=True)

    def log_metric(self, split: str, score: Any, metric_name: str, *args, **kwargs):
        """Log the metric."""
        # Get the model from kwargs
        pl_model = kwargs.get("pl_model", None)
        if pl_model is not None:
            pl_model.log(f"{split}_{metric_name}_step", score)

    def log_metric_end_epoch(
        self, split: str, score: Any, metric_name: str, *args, **kwargs
    ):
        """Log at the end of epoch."""
        # Get the model from kwargs
        pl_model = kwargs.get("pl_model", None)
        if pl_model is not None:
            pl_model.log(f"{split}_{metric_name}_total", score)
