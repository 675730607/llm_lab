import logging
from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .ModelConfig import ModelConfig
from .DecoderLayer import DecoderLayer
from .RMSNormal import RMSNormal
from .Attention import Attention

logger = logging.getLogger(__name__)

class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss = Optional[torch.Tensor]

    def __init__(
            self,
            args: ModelConfig,
    ):
        super().__init__(args)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        # dropout 层
        self.dropout = nn.Dropout(args.dropout)

        # Decoder 层
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        # 归一化层
        self.norm = RMSNormal(args.dim, args.normal_eps)

        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重和输出层的权总共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = Attention.precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len,
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # 初始化所有权重
        self.apply(self._init_weights)

        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters() :
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                nn.init.normal_(
                    p, 
                    mean=0.0,
                    std=0.02/math.sqrt(2 * args.n_layers),
                )
        
        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast() # 输出容器
        # 不分割的模块列表
        self._no_split_modules = [name for name, _ in self.named_parameters()] 


    def _init_weights(self, module: nn.Module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean = 0.0,
                std = 0.02,
            )
            if module.bias is not None :
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, 
                mean = 0.0,
                std = 0.02,
            )

    def forward(
            self,
            tokens: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        - tokens: 输入张量
        - targets: 目标张量
        - kwargs: 其他关键字参数

        - self.OUT: CausalLMOutputWithPast，包含 logits 和损失
        """

        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        # 前向传导函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入层和 dropout 层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # 获取相对位置嵌入的词频
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]
            
        # 通过 docoder 层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                reduction='none',
            )
        else:
            # 推理时的小优化，只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT


    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        stop_id = None,
        max_new_token=256,
        temperature = 1.0,
        top_k = None,
    ):
        """
        给定输入序列 idx（形状为（bz,seq_len)的正整形张量），通过多次生成新 token 来完成序列
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用 kv cache
        """
        index = idx.shape[1]
        for _ in range(max_new_token):
            # 如果序列上下文过长，截断它到最大程度
            if idx.size(1) <= self.args.max_seq_len:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logger.debug("idx_cond before attention %s", idx_cond)
            out = self(idx_cond)
            logits = out.logits
            logger.debug("logits after attention %s", logits)
            # 只保留最后一个时间步的输出
            logits = logits[:, -1, :] 
            logger.debug("logits after time %s", logits)

            if temperature == 0.0:
                # 选择最有可能的索引
                 _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                logger.debug("logits after temperature %s", logits)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logger.debug("v after topk %s", v)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim = -1)
                logger.debug("probs after softmax %s", probs)
                idx_next = torch.multinomial(probs, num_samples=1)
                logger.debug("idx_next after multinomial %s", idx_next)

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim = 1)
        
        # 只返回生成的 token
        return idx[: , index:]