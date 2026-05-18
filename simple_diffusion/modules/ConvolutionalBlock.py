import torch
from typing import Type


class ConvolutionalBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dimension: int,
        activation: Type[torch.nn.Module] = torch.nn.GELU,
    ):
        super().__init__()
        self.abel = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cain = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.activation = activation()
        self.time_network = torch.nn.Linear(time_dimension, out_channels)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.abel(x))
        time_embedding = self.activation(self.time_network(time))
        time_embedding = time_embedding.view(
            time_embedding.size(0), time_embedding.size(1), 1, 1
        )
        x = x + time_embedding
        x = self.activation(self.cain(x))
        return x
