import logging
import torch
from torch import nn
from .ModelConfig import ModelConfig
from .Attention import Attention
from .MLP import MLP
from .RMSNormal import RMSNormal



class DecoderLayer(nn.Module):
    def __init__(
            self,
            layer_id: int,
            args: ModelConfig,
    ):
        super().__init__()

        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = MLP(
            dim = args.dim, 
            hidden_dim = args.hidden_dim,
            multiple_of = args.mutiple_of, 
            dropout = args.dropout,
        )

        self.attention_norm = RMSNormal(args.dim, args.normal_eps)
        self.ffn_norm = RMSNormal(args.dim, args.normal_eps)

    def forward(
            self, 
            x : torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        # 前向传播
        # 1. 输入 x 经过归一化层
        # 2. 进行注意力计算
        # 3. 结果与输入 x 相加得到 h
        # 4. h 经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与 h 相加得到输出
        x_norm = self.attention_norm(x)
        x_attn = self.attention.forward(x_norm, freqs_cos, freqs_sin)
        h = x + x_attn
        h_norm = self.ffn_norm(h)
        out = self.feed_forward.forward(h_norm)
        return h + out
        
