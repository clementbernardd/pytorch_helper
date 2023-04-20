"""File for the sample, which describes the content of data."""
from abc import abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch


class Sample:
    """
    Class that represents data.
    """

    def read_to_torch(self, device: Optional[str], *args, **kwargs):
        """Read the data to torch format."""
        x, y = self.convert_to_x_and_y()
        x, y = self.convert_to_torch(x, device), self.convert_to_torch(y, device)
        return x, y

    @abstractmethod
    def convert_to_x_and_y(self):
        """Convert the sample into X and y"""
        raise NotImplementedError

    @staticmethod
    def _get_device(device: Optional[str]) -> torch.device:
        """
        Return torch.device of the given device
        :param device: the device for pytorch code.
        :return: torch device
        """
        if isinstance(device, str):
            torch_device = torch.device(device)
        elif isinstance(device, torch.device):
            torch_device = device
        else:
            torch_device = torch.device("cpu")
        return torch_device

    @staticmethod
    def convert_to_torch(
        inputs: Union[np.ndarray, List, torch.Tensor, Dict], device: Optional[str]
    ) -> Union[torch.Tensor, Dict]:
        """Convert the inputs into tensor

        :param inputs: elements to be converted to torch format.
        :param device: the device for pytorch code.
        """
        torch_device = Sample._get_device(device)
        if isinstance(inputs, list):
            output = torch.tensor(inputs).to(torch_device)
        elif isinstance(inputs, np.ndarray):
            output = torch.from_numpy(inputs).to(torch_device)
        elif isinstance(inputs, torch.Tensor):
            output = inputs.to(torch_device)
        elif isinstance(inputs, Dict):
            output = Sample._convert_to_torch_dict(inputs, device)  # type: ignore
        else:
            raise NotImplementedError(
                f"Conversion to torch not possible for type : {type(inputs)}"
            )
        return output

    @staticmethod
    def _convert_to_torch_dict(inputs: Dict, device: str) -> Dict:
        """
        Convert the dict inputs to torch.
        :param inputs: dictionary of elements to be converted to torch.
        :return: dictionary with torch elements as values
        """
        new_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, Dict):
                raise TypeError("Nested Dictionary found for conversion")
            else:
                new_inputs[key] = Sample.convert_to_torch(value, device)
        return new_inputs
