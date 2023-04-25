from loguru import logger
from torchvision.datasets import CIFAR10

from thunder.data.datamodule_abstract import DataModuleAbstract
from torchvision import transforms

class ExampleDataModule(DataModuleAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self, *args, **kwargs):
        """Init the train, valid and test datasets.
            Should fill the dictionary self.datasets
            Should download the data if necessary.

        Example :
            self.datasets['train'] = DatasetAbstract(*args, **kwargs)
            self.datasets['valid'] = DatasetAbstract(*args, **kwargs)
            self.datasets['test'] = DatasetAbstract(*args, **kwargs)
        """
        data_means, data_std = [0.49139968,0.48215841,0.44653091], [0.24703223,0.24348513,0.26158784]
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_means, data_std)
        ])
        self.datasets['train'] = self.instantiate_dataset(self.dataset_init, **{'train': True, "transform": transformation})
        self.datasets['valid'] = self.instantiate_dataset(self.dataset_init, **{'train': False, "transform": transformation})
        self.datasets['test'] = self.instantiate_dataset(self.dataset_init, **{'train': False, "transform": transformation})
