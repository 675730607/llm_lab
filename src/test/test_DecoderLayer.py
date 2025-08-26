import sys
import logging
import unittest
import torch
from llm.ModelConfig import ModelConfig
from llm.DecoderLayer import DecoderLayer
from llm.Attention import Attention

logging.basicConfig(
    stream = sys.stdout,
    level = logging.DEBUG,
    format = "%(asctime)s  %(levelname)s  %(pathname)s:%(lineno)d  %(message)s",
)
logger = logging.getLogger(__name__)


class TestDecoderLayer(unittest.TestCase):
    def test_forward(self):
        args = ModelConfig()
        freqs_cos, freqs_sin = Attention.precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len,
        )
        decoder = DecoderLayer(0, args)

        x = torch.randn(1, args.max_seq_len, args.dim)
        out = decoder.forward(x, freqs_cos, freqs_sin)

        self.assertEqual(out.shape, (1, args.max_seq_len, args.dim))
