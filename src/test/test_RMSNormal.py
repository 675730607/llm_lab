import sys
import unittest
import logging
import torch
from llm.RMSNormal import RMSNormal

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(pathname)s  %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug("sys.path=%s", sys.path)

class TestRMSNormal(unittest.TestCase) :
    def test_forwad(self) :
        dim = 4
        eps = 1e-5

        x = torch.Tensor([[[1,  2,  3, 4],
         [ 5, 6, 7, 8],
         [9,  10, 11,  12]],

        [[1,  2,  3, 4],
         [ 5, 6, 7, 8],
         [9,  10, 11,  12]]])
        
        norm = RMSNormal(dim, eps)
        output = norm.forward(x)
        self.assertEqual(output.shape, (2, 3, dim))


if __name__ == "__main__" :
    unittest.main()