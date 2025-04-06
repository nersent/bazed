import copy
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates, approximate="tanh")


def get_linear_out_dim_for_activation(act, dim):
    if isinstance(act, GEGLU):
        return 2 * dim
    return dim

@dataclass
class BaseConfig:
    def clone(self):
        return copy.deepcopy(self)

    def new(self, **kwargs):
        new_config = self.clone()
        new_config.__dict__.update(self.__dict__)
        new_config.__dict__.update(kwargs)
        return new_config

def gelu(x):
    return nn.functional.gelu(x, approximate="tanh")

@dataclass
class FeedForwardConfig(BaseConfig):
    inner_dim: Optional[int] = None
    embed_dim: int = 0
    dropout: float = 0.0
    activation: Callable = gelu
    gated: bool = False

    def get_inner_dim(self):
        if self.inner_dim is not None:
            return self.inner_dim
        return 4 * self.embed_dim



class FeedForward(torch.nn.Module):
    config: FeedForwardConfig

    def __init__(self, config: FeedForwardConfig):
        super().__init__()

        self.config = config

        inner_dim = config.get_inner_dim()
        linear_out_dim = get_linear_out_dim_for_activation(config.activation, inner_dim)

        self.input = nn.Linear(config.embed_dim, linear_out_dim, bias=False)
        self.output = nn.Linear(inner_dim, config.embed_dim, bias=False)
        if config.gated:
            self.gate = nn.Linear(config.embed_dim, linear_out_dim, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)

        self.activation_fn = config.activation
    
    @torch.compile()
    def forward(self, x, **kwargs):
        input = self.input(x)
        
        if self.config.gated:
            x = self.activation_fn(self.gate(x)) * input
        else:
            x = self.activation_fn(input)
        x = self.output(x)
        x = self.dropout(x)
        return x

