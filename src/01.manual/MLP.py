import logging
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
            dropout: float,
    ):
        super().__init__()
        
        # 如果没有指定隐藏层的维度，将其设置成输入维度的4倍
        # 然后将其减少到2/3，最后确保它是 multiple_of 的倍数
        if hidden_dim is None :
            hidden_dim = 4 * dim
            hidden_dim = (2*hidden_dim) // 3
            hidden_dim = multiple_of * (hidden_dim + multiple_of - 1) // multiple_of

        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
    ):
        # 前向传播
        # 1. 输入 x 通过第一层线性变换和 SILU 激活函数
        # 2. 结果乘以输入 x 经过第三层线性变换的结果
        # 3. 通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))