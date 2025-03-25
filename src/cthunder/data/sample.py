"""File for the sample, which describes the content of data."""
from abc import abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch


class Sample:
    """
    Class that represents data.
    """

    def read_to_torch(self, *args, **kwargs):
        """Read the data to torch format."""
        x, y = self.convert_to_x_and_y()
        x, y = self.convert_to_torch(x), self.convert_to_torch(y)
        return x, y

    @abstractmethod
    def convert_to_x_and_y(self):
        """Convert the sample into X and y"""
        raise NotImplementedError

    @staticmethod
    def convert_to_torch(
        inputs: Union[np.ndarray, List, torch.Tensor, Dict]
    ) -> Union[torch.Tensor, Dict]:
        """Convert the inputs into tensor

        :param inputs: elements to be converted to torch format.
        :param device: the device for pytorch code.
        """
        if isinstance(inputs, list):
            output = torch.tensor(inputs)
        elif isinstance(inputs, np.ndarray):
            output = torch.from_numpy(inputs)
        elif isinstance(inputs, torch.Tensor):
            output = inputs
        elif isinstance(inputs, Dict):
            output = Sample._convert_to_torch_dict(inputs)  # type: ignore
        else:
            raise NotImplementedError(
                f"Conversion to torch not possible for type : {type(inputs)}"
            )
        return output

    @staticmethod
    def _convert_to_torch_dict(inputs: Dict) -> Dict:
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
                new_inputs[key] = Sample.convert_to_torch(value)
        return new_inputs
