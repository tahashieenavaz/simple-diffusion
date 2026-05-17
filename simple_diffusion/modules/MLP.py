import torch
from typing import Type


class MLP(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: Type[torch.nn.Module] = torch.nn.GELU
    ):
        self.alpha = torch.nn.Linear(input_dimension, hidden_dimension)
        self.beta = torch.nn.Linear(hidden_dimension, output_dimension)
        self.activation = activation()

    def forward(self, x):
        x = self.alpha(x)
        x = self.activation(x)
        x = self.beta(x)
        return x
