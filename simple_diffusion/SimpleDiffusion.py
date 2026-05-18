from baloot import acceleration_device
from simple_diffusion.constants import DatasetStringType


class SimpleDiffusion:
    def __init__(self):
        params = locals().pop("self")
        self.__set_parameters(params=params)
        self.device = acceleration_device()

    def __set_parameters(self, *, params: dict):
        for key, value in params:
            setattr(self, key, value)

    def train(self, dataset: DatasetStringType):
        pass
