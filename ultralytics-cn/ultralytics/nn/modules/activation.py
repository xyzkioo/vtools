# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""激活函数模块。"""

import torch
from torch import nn


class AGLU(nn.Module):
    """AGLU 统一激活函数模块。

    此类基于 AGLU（自适应门控线性单元）方法，实现了一个带可学习参数的激活函数。

    属性：
        act (nn.Softplus)：beta 为负的 Softplus 激活函数。
        lambd (nn.Parameter)：使用均匀分布初始化的可学习 lambda 参数。
        kappa (nn.Parameter)：使用均匀分布初始化的可学习 kappa 参数。

    方法：
        forward：计算统一激活函数的前向传播。

    示例：
        >>> import torch
        >>> m = AGLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([2])

    参考：
        https://github.com/kostas1515/AGLU
    """

    def __init__(self, device=None, dtype=None) -> None:
        """使用可学习参数初始化统一激活函数。"""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda 参数
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa 参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用自适应门控线性单元（AGLU）激活函数。

        此前向方法使用可学习的 lambda 和 kappa 参数实现 AGLU 激活函数，通过变换自适应地组合线性和非线性
        分量。

        参数：
            x (torch.Tensor)：要应用激活函数的输入张量。

        返回：
            (torch.Tensor)：应用 AGLU 激活函数后的输出张量，形状与输入相同。
        """
        lam = torch.clamp(self.lambd, min=0.0001)  # 限制 lambda 的最小值，避免除零
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
