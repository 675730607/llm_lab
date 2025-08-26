from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig) : 
    model_type = "Tiny-Y"

    def __init__ (
        self,
        dim: int = 768, # 模型维度
        n_layers: int = 12, # transformer层数
        n_heads: int = 16, # 注意力机制的头数
        n_kv_heads: int = 8, # 键值头的数量
        vocab_size: int = 6144, # 词汇表大小
        hidden_dim: int = None, # 隐藏层维度
        mutiple_of: int = 64,
        normal_eps: float = 1e-5, # 归一化层 eps
        max_seq_len: int = 512, # 最大序列长度
        dropout: float = 0.0, # dropout 概率
        flash_attention: bool = True, # 是否使用 Flash Attention 机制
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mutiple_of = mutiple_of
        self.normal_eps = normal_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attention = flash_attention
        super().__init__(**kwargs)