import sys
import logging
import unittest
import torch
from llm.ModelConfig  import ModelConfig
from llm.Transformer import Transformer


logging.basicConfig(
    stream = sys.stdout,
    level = logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(pathname)s:%(lineno)d  %(message)s"
)
logger = logging.getLogger(__name__)


class TestTransformer(unittest.TestCase):
    def test_gengerate(self):
        args = ModelConfig(
            dim = 16, # 模型维度
            n_layers = 2, # transformer层数
            n_heads= 4, # 注意力机制的头数
            n_kv_heads = 2, # 键值头的数量
            vocab_size = 32, # 词汇表大小
            hidden_dim = None, # 隐藏层维度
            mutiple_of = 2,
            normal_eps = 1e-5, # 归一化层 eps
            max_seq_len = 64, # 最大序列长度
            dropout = 0.0, # dropout 概率
            flash_attention = True, # 是否使用 Flash Attention 机制
        )

        model = Transformer(args)
        # 计算 model 的全部参数
        num_params = sum(p.numel() for p in model.parameters())
        logger.info("number of model parameters: %d", num_params)

        x = torch.randint(0, args.vocab_size, (1, 50)) # [bs, seq_len]
        out = model.generate(x, 
                             max_new_token=args.max_seq_len,
                             temperature = 0.5,
                             top_k = 3, 
                             )
        self.assertEqual(out.shape, (1, args.max_seq_len))

    
    def test_forward(self):
        args = ModelConfig()

        model = Transformer(args)

        x = torch.randint(0, args.vocab_size, (1, 50)) # [bs, seq_len]
        out = model(x)
        self.assertEqual(out.logits.shape, (1, 1, args.vocab_size))
