import torch
import math
from .ConvolutionalBlock import ConvolutionalBlock
from .MLP import MLP


class UShapedNetwork(torch.nn.Module):
    def __init__(self, in_channels: int = 1, time_dimension: int = 128):
        super().__init__()
        self.time_dimension = time_dimension

        self.time_network = MLP(
            input_dimension=time_dimension,
            hidden_dimension=time_dimension,
            output_dimension=time_dimension,
        )

        self.alpha = ConvolutionalBlock(in_channels, 64, time_dimension)
        self.beta = ConvolutionalBlock(64, 128, time_dimension)
        self.gamma = ConvolutionalBlock(128, 256, time_dimension)
        self.pool = torch.nn.MaxPool2d(2)

        self.up1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvolutionalBlock(256, 128, time_dimension)

        self.up2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvolutionalBlock(128, 64, time_dimension)

        self.final_conv = torch.nn.Conv2d(64, in_channels, kernel_size=1)

    def get_time_embedding(self, t, dim):
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

    def forward(self, x, t):
        time_embedding = self.get_time_embedding(t, self.time_dim)
        time_embedding = self.time_network(time_embedding)

        # 2. Encoder (Save outputs for skip connections)
        e1 = self.alpha(x, time_embedding)  # [B, 64, 28, 28]
        e2 = self.beta(self.pool(e1), time_embedding)  # [B, 128, 14, 14]

        # 3. Bottleneck
        b = self.gamma(self.pool(e2), time_embedding)  # [B, 256, 7, 7]

        # 4. Decoder (Upsample and concatenate skip connections)
        d1 = self.up1(b)  # [B, 128, 14, 14]
        d1 = torch.cat([d1, e2], dim=1)  # Concat skip connection
        d1 = self.dec1(d1, time_embedding)  # [B, 128, 14, 14]

        d2 = self.up2(d1)  # [B, 64, 28, 28]
        d2 = torch.cat([d2, e1], dim=1)  # Concat skip connection
        d2 = self.dec2(d2, time_embedding)  # [B, 64, 28, 28]

        # 5. Output noise prediction
        return self.final_conv(d2)
