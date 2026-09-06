# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import optim


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """使用 Newton-Schulz 迭代计算矩阵 G 的零次幂或正交化结果。

    此函数实现五次 Newton-Schulz 迭代，以近似正交化输入矩阵 G。迭代系数经过优化，可最大化零点处的收敛斜率，
    生成类似 SVD 中 UV^T 的结果（USV^T = G），同时放宽收敛保证，在优化任务中经过实验证明效果良好。

    参数：
        G (torch.Tensor): 要进行正交化的二维矩阵或三维矩阵批次。
        eps (float, 可选): 添加到范数中的小 epsilon 值，用于保证数值稳定性。默认：1e-7。

    返回：
        (torch.Tensor): 与输入 G 形状相同的正交化矩阵或矩阵批次。

    示例：
        >>> G = torch.randn(128, 64)
        >>> G_ortho = zeropower_via_newtonschulz5(G)
        >>> print(G_ortho.shape)
        torch.Size([128, 64])

    注意：
        - 使用 bfloat16 精度进行计算。
        - 使用固定系数准确执行 5 次 Newton-Schulz 迭代。
        - 当行数大于列数时自动转置，以提高效率。
        - 输出近似于 US'V^T，其中 S' 的对角元素近似服从 Uniform(0.5, 1.5)。
        - 不会生成精确的 UV^T，但在神经网络优化中具有良好的经验效果。
    """
    assert G.ndim in {2, 3}
    X = G.reshape(-1, G.size(-2), G.size(-1)).bfloat16()
    X /= X.norm(dim=(-2, -1), keepdim=True) + eps  # 确保最大奇异值 <= 1
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.transpose(-2, -1)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)  # b * A + c * A @ A
        X = torch.baddbmm(X, B, X, beta=a)  # a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.reshape(G.shape)


def muon_update(
    grad: torch.Tensor | list[torch.Tensor],
    momentum: torch.Tensor | list[torch.Tensor],
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    """使用动量和正交化计算 Muon 优化器更新量。

    此函数对梯度应用动量，可选使用 Nesterov 加速，然后通过 Newton-Schulz 迭代对更新量进行正交化。
    行数相同的矩阵会进行零填充，并在一次批量调用中完成正交化；动量计算使用融合的 foreach 操作，
    避免为每个参数启动内核的额外开销。高阶张量会在正交化前调整形状，每个更新量根据参数维度进行缩放。

    参数：
        grad (torch.Tensor | 列表[torch.Tensor]): 要更新的梯度张量，每个张量至少包含两个维度。
        momentum (torch.Tensor | 列表[torch.Tensor]): 动量缓冲区张量，会原地修改。
        beta (float, 可选): 指数移动平均的动量系数。默认：0.95。
        nesterov (bool, 可选): 是否使用 Nesterov 动量加速。默认：True。

    返回：
        (torch.Tensor | 列表[torch.Tensor]): 正交化后的更新张量，每个张量的形状和数据类型与对应梯度相同。

    示例：
        >>> grad = torch.randn(64, 128)
        >>> momentum = torch.zeros_like(grad)
        >>> update = muon_update(grad, momentum, beta=0.95, nesterov=True)
        >>> print(update.shape)
        torch.Size([64, 128])

    注意：
        - 动量缓冲区原地更新：momentum = beta * momentum + (1-beta) * grad。
        - 使用 Nesterov 时：update = beta * momentum + (1-beta) * grad。
        - 不使用 Nesterov 时：update = momentum。
        - 维度大于 2 的张量会重塑为二维，同时保留第一维。
        - 最终更新量按 sqrt(max(1, dim[-2] / dim[-1])) 缩放，以适配参数维度。
    """
    single = isinstance(grad, torch.Tensor)
    grads, momentums = ([grad], [momentum]) if single else (grad, momentum)
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, grads, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, grads, alpha=1 - beta)
    else:
        updates = list(momentums)
    buckets = {}  # 按（行数、缩放系数）分组转置后的矩阵，使行数 <= 列数以便批量正交化
    for i, u in enumerate(updates):
        m = u.view(len(u), -1) if u.ndim > 2 else u
        transpose = m.size(0) > m.size(1)
        if transpose:
            m = m.transpose(0, 1)
        scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5
        buckets.setdefault((m.size(0), scale, m.device, m.dtype), []).append((i, m, transpose))
    for (_, scale, _, _), items in buckets.items():
        n = max(m.size(1) for _, m, _ in items)
        # 对列进行零填充，使不同形状可以共享一次批处理调用（零值经过 Newton-Schulz 过程后仍为零）。
        X = torch.stack([torch.nn.functional.pad(m, (0, n - m.size(1))) for _, m, _ in items])
        X = zeropower_via_newtonschulz5(X).to(grads[items[0][0]].dtype).mul_(scale)
        for j, (i, m, transpose) in enumerate(items):
            x = X[j, :, : m.size(1)]
            updates[i] = (x.T if transpose else x).reshape(grads[i].shape)
    return updates[0] if single else updates


class MuSGD(optim.Optimizer):
    """结合 Muon 和 SGD 更新的神经网络混合优化器。

    此优化器结合 Muon（通过 Newton-Schulz 迭代进行正交化的动量优化器）和带动量的标准 SGD。
    不同参数组可以选择使用 Muon+SGD 混合方案或纯 SGD。

    参数：
        params (Iterable): 要优化的参数，或用于定义参数组的字典。
        muon (float, 可选): 混合模式下 Muon 更新的权重因子。默认值：0.5。
        sgd (float, 可选): 混合模式下 SGD 更新的权重因子。默认值：0.5。

    属性：
        muon (float): Scaling factor applied to Muon learning rate.
        sgd (float): Scaling factor applied to SGD learning rate in hybrid mode.

    示例：
        >>> param_groups = [
        ...     {
        ...         "params": model.conv_params,
        ...         "lr": 0.02,
        ...         "use_muon": True,
        ...         "momentum": 0.95,
        ...         "nesterov": True,
        ...         "weight_decay": 0.01,
        ...     },
        ...     {
        ...         "params": model.other_params,
        ...         "lr": 0.01,
        ...         "use_muon": False,
        ...         "momentum": 0.9,
        ...         "nesterov": False,
        ...         "weight_decay": 0,
        ...     },
        ... ]
        >>> optimizer = MuSGD(param_groups, muon=0.5, sgd=0.5)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    注意：
        - 'use_muon' 为 True 的参数组同时执行 Muon 和 SGD 更新。
        - 'use_muon' 为 False 的参数组仅执行 SGD 更新。
        - Muon 更新使用正交化，最适合二维及更高维的参数张量。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        """初始化具备 Muon 和 SGD 混合能力的 MuSGD 优化器。

        参数：
            params (Iterable): 要优化的参数，或定义参数组的字典。
            lr (float): 学习率。
            momentum (float): SGD 动量系数。
            weight_decay (float): 权重衰减（L2 惩罚）。
            nesterov (bool): 是否使用 Nesterov 动量。
            use_muon (bool): 是否启用 Muon 更新。
            muon (float): Muon 分量的缩放系数。
            sgd (float): SGD 分量的缩放系数。
        """
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "use_muon": use_muon,
        }
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        """执行一次优化步骤。

        根据每个参数组中的 'use_muon' 标志，执行 Muon+SGD 混合更新或纯 SGD 更新。
        对于启用 Muon 的参数组，参数会同时接收正交化的 Muon 更新和标准 SGD 动量更新。

        参数：
            closure (Callable, 可选): 用于重新评估模型并返回损失的闭包。默认值：None。

        返回：
            (torch.Tensor | None): 如果提供 closure，则返回损失值，否则返回 None。

        注意：
            - 梯度为 None 的参数会跳过。
            - Muon 更新使用 Newton-Schulz 正交化，最适合二维及更高维张量。
            - 混合模式下，权重衰减仅应用于 SGD 部分。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
            for p in params:
                if len(self.state[p]) == 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    if group["use_muon"]:
                        self.state[p]["momentum_buffer_SGD"] = torch.zeros_like(p)
            if group["use_muon"]:
                updates = muon_update(
                    [p.grad for p in params],
                    [self.state[p]["momentum_buffer"] for p in params],
                    beta=momentum,
                    nesterov=nesterov,
                )
                torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
                buffers = [self.state[p]["momentum_buffer_SGD"] for p in params]
                lr *= self.sgd
            else:
                buffers = [self.state[p]["momentum_buffer"] for p in params]
            # SGD 更新
            grads = [p.grad for p in params]
            if group["weight_decay"] != 0:
                grads = torch._foreach_add(grads, params, alpha=group["weight_decay"])
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grads)
            updates = torch._foreach_add(grads, buffers, alpha=momentum) if nesterov else buffers
            torch._foreach_add_(params, updates, alpha=-lr)
        return loss


class Muon(optim.Optimizer):
    """用于非分布式环境的 Muon 优化器。

    此优化器实现 Muon 算法，通过 Newton-Schulz 迭代将基于动量的更新与正交化结合，
    并对参数更新应用权重衰减和学习率缩放。

    参数：
        params (iterable): 要优化的参数迭代器，或用于定义参数组的字典。
        lr (float, 可选): 学习率。默认值：0.02。
        weight_decay (float, 可选): 权重衰减（L2 惩罚）系数。默认值：0。
        momentum (float, 可选): 指数移动平均的动量系数。默认值：0.95。

    属性：
        param_groups (列表): 包含优化设置的参数组列表。
        state (dict): 包含每个参数优化器状态的字典。

    示例：
        >>> model = YourModel()
        >>> optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.95)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    注意：
        - 用于非分布式训练环境。
        - 对所有参数使用带正交化的 Muon 更新。
        - 权重衰减在参数更新前以乘法形式应用。
        - 梯度为 None 的参数会被赋予零梯度，以便同步。
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0, momentum: float = 0.95):
        """初始化使用正交化更新的 Muon 优化器。

        参数：
            params (Iterable): 要优化的参数，或定义参数组的字典。
            lr (float): 学习率。
            weight_decay (float): 乘法形式应用的权重衰减系数。
            momentum (float): 梯度累积的动量系数。
        """
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一次优化步骤。

        对所有参数应用包含动量和正交化的 Muon 更新，并在参数更新前以乘法方式应用权重衰减。

        参数：
            closure (Callable[[], torch.Tensor] | None, 可选): 用于重新评估模型并返回损失的闭包。默认值：None。

        返回：
            (torch.Tensor | None): 如果提供 closure，则返回损失值，否则返回 None。

        示例：
            >>> optimizer = Muon(model.parameters())
            >>> loss = model(inputs)
            >>> loss.backward()
            >>> optimizer.step()

        注意：
            - 梯度为 None 的参数会被赋予零梯度，以便同步。
            - 权重衰减按如下方式应用：p *= (1 - lr * weight_decay)。
            - Muon 更新使用 Newton-Schulz 正交化，最适合二维及更高维张量。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)  # 强制同步
                if len(self.state[p]) == 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
            updates = muon_update(
                [p.grad for p in params], [self.state[p]["momentum_buffer"] for p in params], beta=group["momentum"]
            )
            torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(params, updates, alpha=-group["lr"])

        return loss
