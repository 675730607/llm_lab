import torch
import logging
import torch.nn as nn
logger = logging.getLogger(__name__)

class RMSNormal(nn.Module) :
    def __init__(self, dim : int, eps : float):
        super().__init__()
        self.eps = eps # eps 是为了防止除以0的情况
        self.weight = nn.Parameter(torch.ones(dim)) # weight是一个可学习的参数，全部初始化为1

    def _normal(self, x):
        r"""
        计算 RMSNormal 的核心部分
        """

        # x 的平方均值
        logger.debug("normal input\n%s", x)
        mean = torch.pow(x, 2).mean(-1, keepdim=True) 
        logger.debug("normal pow(2).mean\n%s", mean)
        # rsqrt 是平方根的倒数
        rs = torch.rsqrt(mean + self.eps)
        logger.debug("normal rsqrt\n%s", rs)
        norm = x * rs
        logger.debug("normal norm\n%s", norm)
        return norm
    
    def forward(self, x : torch.Tensor) -> torch.Tensor :
        r"""
        forward 函数是模型的前向传导
        1. 首先将 x 转成 float 类型，然后进行 RMSNormal，最后转换会原来的类型
        2. 最后乘以weight，这是 RMSNormal 的一个可以可学习的缩放因子
        """
        norm = self._normal(x.float()).type_as(x)
        output = norm * self.weight
        logger.debug("normal weight\n%s", self.weight)
        logger.debug("normal output\n%s", output)
        return output
        