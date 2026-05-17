import torch


class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dimension: int):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.activation = torch.nn.GELU()
        self.time_mlp = torch.nn.Linear(time_dimension, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        time_emb = self.activation(self.time_mlp(t))
        time_emb = time_emb.view(time_emb.size(0), time_emb.size(1), 1, 1)
        x = x + time_emb
        x = self.activation(self.conv2(x))
        return x
