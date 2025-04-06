# enum
from dataclasses import dataclass
from typing import Optional
from einops import rearrange
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch

from ml.mlp import BaseConfig
from ml.rotary import apply_rotary_pos_emb


class SdpaKernel(str):
  AUTO = "auto"
  FLASH = "flash"
  MEMORY_EFFICIENT = "memory_efficient"
  MATH = "math"

class QKVStrategy(str):
  FUSED = "fused"
  SPLIT = "split"
  FUSED_KV = "fused_kv"

@dataclass
class AttentionConfig(BaseConfig):
    embed_dim: int
    dropout: float = 0.0
    num_heads: int = 1
    kv_dim: Optional[int] = None
    bias: bool = False
    out_proj_bias: bool = False
    num_kv_heads: Optional[int] = None
    grouped_heads: bool = False
    qkv_strategy: str = QKVStrategy.SPLIT
    causal: bool = False

    def get_kv_dim(self):
        if self.kv_dim is not None:
            return self.kv_dim
        return self.embed_dim
      
    def get_kv_heads(self):
        if self.num_kv_heads is not None:
            return self.num_kv_heads
        return self.num_heads

#torch.compile()
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
  
class Attention(torch.nn.Module):
    config: AttentionConfig

    def __init__(self, config: AttentionConfig):
        super().__init__()

        self.config = config

        self.head_dim = config.embed_dim // config.num_heads
        self.kv_heads = config.get_kv_heads()
        kv_dim = config.get_kv_dim()
        kv_out_dim = self.head_dim * self.kv_heads

        if config.qkv_strategy == QKVStrategy.FUSED:
            if self.kv_heads != config.num_heads or config.grouped_heads:
              raise ValueError("Fused QKV only works with equal number of heads for query and key/value")
            
            self.qkv = nn.Linear(config.embed_dim, 3 * self.head_dim * self.config.num_heads, bias=config.bias)
        elif config.qkv_strategy == QKVStrategy.FUSED_KV:
            self.q = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
            self.kv = nn.Linear(kv_dim, 2 * kv_out_dim, bias=config.bias)
        elif config.qkv_strategy == QKVStrategy.SPLIT:
            self.q = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
            self.k = nn.Linear(kv_dim, kv_out_dim, bias=config.bias)
            self.v = nn.Linear(kv_dim, kv_out_dim, bias=config.bias)
        else:
            raise ValueError(f"Invalid qkv_strategy: {config.qkv_strategy}")
          
        self.num_kv_groups = config.num_heads // self.kv_heads

        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.out_proj_bias)
        self.dropout = nn.Dropout(config.dropout)
        
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def init_weights(self):
        gain = (8 * 16) ** -0.25
        nn.init.xavier_normal_(self.q.weight, gain=1)
        nn.init.xavier_normal_(self.k.weight, gain=1)
        nn.init.xavier_normal_(self.v.weight, gain=gain)
        nn.init.xavier_normal_(self.out_proj.weight, gain=gain)

    #torch.compile()
    def qkv_fwd(self, x: torch.Tensor, cross_states: torch.Tensor):
        if self.config.qkv_strategy == QKVStrategy.FUSED:
            qkv = self.qkv(x)
            qkv = rearrange(qkv, 'b n (h d) -> b n h d', h=self.config.num_heads)
            query, key, value = qkv.split(self.head_dim, dim=-1)
        elif self.config.qkv_strategy == QKVStrategy.FUSED_KV:
            query = self.q(x)
            kv = self.kv(cross_states)
            key, value = kv.split(self.config.embed_dim, dim=-1)
        elif self.config.qkv_strategy == QKVStrategy.SPLIT:
            query = self.q(x)
            key = self.k(cross_states)
            value = self.v(cross_states)
            
        if self.config.qkv_strategy != QKVStrategy.FUSED:
            query = rearrange(query, 'b n (h d) -> b n h d', h=self.config.num_heads)
            key = rearrange(key, 'b n (h d) -> b n h d', h=self.kv_heads)
            value = rearrange(value, 'b n (h d) -> b n h d', h=self.kv_heads)
        
        return query, key, value

    # @torch.compile()
    def forward(self, x, rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]]=None, cross_states=None, mask=None, **kwargs):
        if self.config.qkv_strategy == QKVStrategy.FUSED and cross_states is not None:
            raise NotImplementedError()

        if cross_states is None:
            cross_states = x

        if self.training:
            dropout_p = self.config.dropout
        else:
            dropout_p = 0.0

        qkv = self.qkv_fwd(x, cross_states)
            
           
        query, key, value = qkv
    
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query.device.type == "cuda" and mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            
    
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        if rotary_emb is not None:
            cos, sin = rotary_emb
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if self.config.grouped_heads:
            key = repeat_kv(key, self.num_kv_groups)
            value = repeat_kv(value, self.num_kv_groups)

        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION if self.training else SDPBackend.FLASH_ATTENTION):
          x = torch.nn.functional.scaled_dot_product_attention(
              query,
              key,
              value,
              dropout_p=dropout_p,
              is_causal=self.config.causal,
          )
                
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.out_proj(x)
        x = self.dropout(x)

        return x