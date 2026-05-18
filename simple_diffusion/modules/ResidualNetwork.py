import torch


class ResidualNetwork(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dimension: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(time_dimension, out_channels)
        )
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.norm1 = torch.nn.GroupNorm(8, in_channels)
        self.norm2 = torch.nn.GroupNorm(8, out_channels)
        self.activation = torch.nn.SiLU()
        self.shortcut = (
            torch.nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )

    def forward(self, x, t):
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.mlp(t)[:, :, None, None]
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.shortcut(x)
