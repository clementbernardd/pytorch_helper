"""Class that does the main functions of a deep learning model."""
from typing import Dict, Union

import hydra
from loguru import logger
from omegaconf import DictConfig

from cthunder.config.config_helper_yaml import ConfigHelperYAML
from cthunder.experiments.abstract_experiment import AbstractExperiment


class CLIHelper:
    def __init__(self, config_path: Union[str, Dict], *args, **kwargs):
        self.config_path = config_path
        self.config_helper = ConfigHelperYAML(config_path)

    @staticmethod
    @hydra.main(version_base=None)
    def cli(cfg: DictConfig) -> None:
        """
        Launch the appropriate command.
        It will launch what is in the `command` parameter.
        Args:
            cfg: a dictionary with all the hyperparameters.
        """
        command = cfg.get("command", None)
        if command is None:
            logger.debug(
                "NO LAUNCHING COMMAND SPECIFIED. IT SHOULD BE IN THE 'command' name"
            )
        cli_helper = CLIHelper(config_path=cfg)
        experiment = cli_helper.config_helper.get_experiment_class()
        name_to_function = {
            "train": experiment.train,
            "test": cli_helper.test,
            "infer": cli_helper.infer,
            "predict": experiment.predict,
        }
        if command in name_to_function:
            name_to_function[command](**cfg)
        else:
            logger.debug(f"{command} IS NOT A COMMAND AVAILABLE")

    def train(self, experiment: AbstractExperiment, *args, **kwargs):
        """
        Do the training process
        :param experiment: the experiment to use
        :return:
        """
        experiment.train(*args, **kwargs)

    def test(self, experiment: AbstractExperiment, *args, **kwargs):
        """
        Do the evaluation process
        :param args:
        :param kwargs:
        :return:
        """
        experiment.test(*args, **kwargs)

    def infer(self, *args, **kwargs):
        """
        Do inference for a given input.
        :param args:
        :param kwargs:
        :return:
        """
        logger.debug("INFERENCE CLI")


if __name__ == "__main__":
    CLIHelper.cli()

