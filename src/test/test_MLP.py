import sys
import logging
import unittest
import torch
from llm.MLP import MLP
from llm.ModelConfig import ModelConfig



logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(pathname)s:%(lineno)d  %(message)s"
)
logger = logging.getLogger(__name__)


class TestMLP(unittest.TestCase):
    def test_format(self):
        args = ModelConfig()
        mlp = MLP(args.dim, args.hidden_dim, args.mutiple_of, args.dropout)

        x = torch.randn(1, 50, args.dim)
        output = mlp.forward(x)
        self.assertEqual(output.shape, (1, 50, args.dim))