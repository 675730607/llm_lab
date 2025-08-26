import sys
import unittest
import logging
import torch
from llm.Attention import Attention
from llm.ModelConfig import ModelConfig

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(pathname)s:%(lineno)d  %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug("sys.path=%s", sys.path)

class TestAttention(unittest.TestCase) :
    def test_repeat_kv(self) :
        x = torch.randn(2, 3, 4, 5)
        
        output = Attention.repeat_kv(x, 2)
        self.assertEqual(output.shape, (2, 3, 8, 5))

    def test_precompute_freqs_cis(self):
        dim = 8
        end = 32
        fcos, fsin = Attention.precompute_freqs_cis(dim, end)
        self.assertEqual(fcos.shape, fsin.shape)

    def test_reshape_for_broadcast(self):
        freqs_cis = torch.randn(2, 4)
        x = torch.randn(4, 2, 8, 4)

        output = Attention.reshape_for_broadcast(freqs_cis, x)
        self.assertEqual(output.shape, (1, 2, 1, 4))
        logger.debug("reshape_for_broadcast freqs_cis:\n%s", freqs_cis)
        logger.debug("reshape_for_broadcast output:\n%s", output)

        
    def test_apply_rotary_emb(self):
        dim = 288
        n_head = 6
        slen = 50
        xq = torch.randn(1, slen, n_head, dim//n_head) 
        xk = torch.randn(1, slen, n_head, dim//n_head)

        # 使用 precompute_freqs_cis 函数获取 sin和cos
        cos, sin = Attention.precompute_freqs_cis(dim//n_head, slen)
        logger.debug("cos.shape:\n%s", cos.shape)
        logger.debug("cos.shape:\n%s", sin.shape)
        xq_out, xk_out = Attention.apply_rotary_emb(xq, xk, cos, sin)

        xq_out.shape, xk_out.shape
        logger.debug("xq_out.shape:\n%s", xq_out.shape)
        logger.debug("xk_out.shape:\n%s", xk_out.shape)

    def test_forward(self):
        args = ModelConfig()
        attention = Attention(args)

        # 模拟输入数据
        batch_size = 1
        seq_len = 50  # 假设实际使用的序列长度为50
        dim = args.dim
        x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量

        freqs_cos, freqs_sin = Attention.precompute_freqs_cis(dim//args.n_heads, seq_len)

        # 运行Attention模型
        output = attention(x, freqs_cos, freqs_sin)

        # attention出来之后的形状 依然是[batch_size, seq_len, dim]
        print("Output shape:", output.shape)

if __name__ == "__main__" :
    unittest.main()