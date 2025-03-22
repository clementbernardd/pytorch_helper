"""Weight&Biases logger."""
import os
from typing import Any, Optional, Dict

from dotenv import load_dotenv

import wandb
from omegaconf import OmegaConf

from cthunder.loggers.logger_abstract import LoggerAbstract


class WandbLogger(LoggerAbstract):
    """Wandb custom logger."""

    def __init__(self, name: str = "rna3d", is_local: bool = False, *args, **kwargs):
        """
        Should have in `.env'. file the given key : WB_LOCAL_KEY
        """
        self.name = name
        self.is_local = is_local
        super().__init__(*args, **kwargs)

    def init_logger(self, config: Optional[Dict] = {},*args, **kwargs):
        """Init the logger"""
        # Load the variables in .env file
        load_dotenv()
        wb_token = os.environ.get("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = wb_token
        config = OmegaConf.to_container(config, resolve=True)
        mode = "offline" if self.is_local else "online"
        wandb.init(project="rnattention", name=self.name, config=config, mode=mode, dir="logs")

    def log_loss(self, split: str, loss: Any, *args, **kwargs):
        """Log the loss."""
        wandb.log({f"{split}_loss": loss})

    def log_metric(self, split: str, score: Any, metric_name: str, *args, **kwargs):
        """Log the metric."""
        wandb.log({f"{split}_{metric_name}": score})

    def log_metric_end_epoch(
        self, split: str, score: Any, metric_name: str, *args, **kwargs
    ):
        """Log at the end of epoch."""
        wandb.log({f"{split}_{metric_name}_epoch": score})

    def log_value(self, name: str, value: Any, *args, **kwargs):
        wandb.log({name: value})
