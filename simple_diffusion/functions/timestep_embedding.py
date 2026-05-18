import math
import torch


def timestep_embedding(
    timesteps: torch.Tensor, dimension: int, high_bound: int = 10_000
) -> torch.Tensor:
    half_dimension = dimension // 2
    frequencies = torch.exp(
        -math.log(high_bound)
        * torch.arange(half_dimension, device=timesteps.device)
        / half_dimension
    )
    arguments = timesteps.float[:, None] * frequencies[None, :]
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
