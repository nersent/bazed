import torch
import torch.nn as nn

import torch.nn.functional as F

from typing import Any, Optional

from transformers import PreTrainedModel, PretrainedConfig

from ml.mlp import FeedForward, FeedForwardConfig
from ml.rotary import RotaryEmbedding
from ml.attention import Attention, AttentionConfig, QKVStrategy

eps = 1e-5


class TransformerConfig(PretrainedConfig):
    def __init__(
        self,
        embed_dim: int = 512,
        num_decoder_layers: int = 4,
        initializer_range=0.02,
        context_size=512,
        vocab_size: int = 0,
        padding_idx: int = 0,
        activation: str = "silu",
        num_heads: int = 1,
        num_kv_heads: int = 1,
        grouped_heads: bool = False,
        qkv_strategy: QKVStrategy = QKVStrategy.SPLIT,
        rotary: bool = False,
        intermediate_size: Optional[int] = None,
        rope_theta: float = 10_000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim

        self.intermediate_size = intermediate_size  # None handled by FeedForwardConfig
        self.num_decoder_layers = num_decoder_layers
        self.initializer_range = initializer_range
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.activation = activation

        self.rope_theta = rope_theta
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.grouped_heads = grouped_heads
        self.qkv_strategy = qkv_strategy
        self.rotary = rotary

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x).square()
      

      
class Transformer(PreTrainedModel):
    config_class = TransformerConfig
    config: TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
        )
        
        self.decoder = nn.ModuleList(
            [
                Block(
                    config=config,
                    time_mixer=Attention(
                        AttentionConfig(
                            embed_dim=config.embed_dim,
                            num_heads=config.num_heads,
                            num_kv_heads=config.num_kv_heads,
                            grouped_heads=config.grouped_heads,
                            qkv_strategy=config.qkv_strategy,
                            bias=False,
                            out_proj_bias=False,
                            causal=True,
                        ),
                    ),
                    channel_mixer=FeedForward(
                        config=FeedForwardConfig(
                            embed_dim=config.embed_dim,
                            activation=ReluSquared(),
                            gated=False,
                            inner_dim=config.intermediate_size,
                            dropout=0.1,
                        )
                    )
                )
                for i in range(config.num_decoder_layers)
            ]
        )

        if config.rotary:
            self.rotary_emb = RotaryEmbedding((config.embed_dim // config.num_heads), base=self.config.rope_theta, max_position_embeddings=self.config.context_size)

        self.ln = nn.LayerNorm(config.embed_dim)

        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
        self.head.weight.data.zero_()

    def _init_weights(self, module):
        """Initialize the weights."""
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Module):
        self.token_embedding = value

    def set_output_embeddings(self, value: nn.Module):
        self.head = value

    def get_output_embeddings(self) -> nn.Module:
        return self.head
      
    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        x = self.token_embedding(input_ids)
      
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)

        rotary_emb = None
        if self.config.rotary:
            rotary_emb = self.rotary_emb(x, position_ids)
            
        for i in range(len(self.decoder)):
            x = self.decoder[i](
                x,
                rotary_emb=rotary_emb,
            )

        x = self.ln(x)
        x = self.head(x)
        
        x = 30 * torch.tanh(x / 30)
        x = x.float()

        return x


class Block(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        time_mixer: Any,
        channel_mixer: Optional[Any] = None,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        self.time_mixer = time_mixer
        self.channel_mixer = channel_mixer
        

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ):        
        res = x
        x = self.time_mixer(self.ln1(x), rotary_emb=rotary_emb)
        x = x + res

        if self.channel_mixer is not None:
            res = x
            x = self.channel_mixer(self.ln2(x))
            x = x + res

        return x
    
