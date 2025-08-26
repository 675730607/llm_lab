import logging
import torch
import math
from .ModelConfig import ModelConfig
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class Attention(nn.Module) :
    def __init__(self, args: ModelConfig):
        super().__init__()
        
        # 根据是否指定 n_kv_heads，确定用于键/值的头的数量
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键/值头数整除
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1
        model_parallel_size = 1
        # 本地键值头数，等于总头数除以模型并行处理大小
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        # 输出权重矩阵
        self.wo = nn.Linear(args.dim, args.dim, bias = False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存 dropout 概率
        self.dropout = args.dropout

        # 检查是否使用 Flash Attention
        self.flash = False # hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # 如果不支持 Flash Attention，则使用手动实现的注意力机制，并设置mask
            logger.warn("using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来消息
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
    ):
        # 获取批次大小大序列长度[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 计算查询（Q）、键（K）、值（V)
        xq: torch.Tensor = self.wq(x)
        xk: torch.Tensor = self.wk(x)
        xv: torch.Tensor = self.wv(x)
        # 调整形状以适应头的维度
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入（RoPE）
        # FIXME xq xk 的维度可能不同
        xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cos, freqs_sin) 

        # 对键和值进行扩展以适应重复次数
        xk = Attention.repeat_kv(xk, self.n_rep)
        xv = Attention.repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持 Flash Attention，选择实现方式
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.train else 0.0,
                is_causal=True,
            )
        else:
            # 使用手动实现的注意力机制
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最后投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

        




    def repeat_kv(x : torch.Tensor, n_rep : int) -> torch.Tensor :
        # 获取输入张量的维度
        # 批量大小，序列长度，键/值对头的数量，每个头的维度大小
        bs, slen, n_kv_heads, head_dim = x.shape

        # 如果重复次数为1，直接返回原始张量
        if n_rep == 1:
            return x
        
        # 1. 在第三个维度（键/值对头的数量）后添加一个维度, shape = torch.Size([2, 3, 4, 1, 5])
        # 2. 使用 expend 将新添加的维度扩展到 n_rep 大小，实现键/值对的重复效果
        # 3. 将扩展后的维度合并回键/值对头的数量中
        s1 = x[:, :, :, None, :] # 等价于 x.reshape(bs, slen, n_kv_heads, 1, head_dim)
        s2 = s1.expand(bs, slen, n_kv_heads, n_rep, head_dim)
        s3 = s2.reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        logger.debug("repeat_kv x:\n%s", x)
        logger.debug("repeat_kv s1: shape %s\n%s", s1.shape, s1)
        logger.debug("repeat_kv s1:\n%s", s1)
        logger.debug("repeat_kv s2:\n%s", s2)
        logger.debug("repeat_kv s3:\n%s", s3)
        return s3
    
    # 计算旋转嵌入的实部和虚部
    # 注意：此处的 dim 是 dim/n_head，因为我们是对每个head进行旋转嵌入
    def precompute_freqs_cis(
            dim: int,
            end: int,
            theta: float = 10000.0
    ) -> tuple[torch.Tensor, torch.Tensor] :
        # 生成从0开始，步长为2的序列，长度是 dim 的一半
        seq = torch.arange(0, dim, 2)[:dim//2].float()
        logger.debug("precompute_freqs_cis seq:\n%s", seq)

        # 每个元素除以dim，再去theta的倒数，得到频率
        freqs = 1.0 / (theta ** (seq / dim))
        logger.debug("precompute_freqs_cis freqs:\n%s", freqs)

        # 生成从0到end的序列，得到时间序列，end通常是序列的最大长度，
        t = torch.arange(end, device=freqs.device)
        logger.debug("precompute_freqs_cis t:\n%s", t)

        # 计算频率外积，得到二维的频率矩阵，每一行是时间序列 t 乘以频率序列 freqs 的元素
        fouter = torch.outer(t, freqs).float()
        logger.debug("precompute_freqs_cis fouter:\n%s", fouter)

        # 计算频率矩阵 freqs 的余弦值，得到旋转嵌入的实部
        freqs_cos = torch.cos(fouter)
        logger.debug("precompute_freqs_cis freqs_cos:\n%s", freqs_cos)

        # 计算频率矩阵 freqs 的正弦值，得到旋转嵌入的虚部
        freqs_sin = torch.sin(fouter)
        logger.debug("precompute_freqs_cis freqs_sin:\n%s", freqs_sin)
        
        return freqs_cos, freqs_sin
    
    # 调整 freqs_cis 的形状，是其在进行广播操作是与 x 对齐
    def reshape_for_broadcast(
            freqs_cis: torch.Tensor,
            x: torch.Tensor,
    ):
        ndim = x.ndim
        assert ndim > 1
        # freqs_cis 的形状与 x 的第1维和最后一维相等
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])

        # 构造一个新的形状，除了第一维和最后一维其他维度都是1
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

        # 将 freqs_cis 调整成新的形状
        return freqs_cis.view(shape)
    
    def apply_rotary_emb(
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 将查询和键张量重塑形状，以分离实部和虚部
        xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
        xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

        # 塑性频率张量以进行广播
        freqs_cos = Attention.reshape_for_broadcast(freqs_cos, xq_r)
        freqs_sin = Attention.reshape_for_broadcast(freqs_sin, xq_r)

        # 应用旋转，分别计算旋转后的实部和虚部
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        # 将最后两个维度合并，并还原成原始张量的形状
        xq_out = torch.stack([xq_out_r, xq_out_i], dim =-1 ).flatten(3)
        xk_out = torch.stack([xk_out_r, xk_out_i], dim = -1).flatten(3)
        
        return xq_out, xk_out