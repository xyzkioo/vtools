# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def select_closest_cond_frames(frame_idx: int, cond_frame_outputs: dict[int, Any], max_cond_frame_num: int):
    """选择距离给定帧索引最近的条件帧。

    参数：
        frame_idx (int): 当前帧索引。
        cond_frame_outputs (dict[int, Any]): 以帧索引为键的条件帧输出字典。
        max_cond_frame_num (int): 要选择的条件帧最大数量。

    返回：
        selected_outputs (dict[int, Any]): 从 cond_frame_outputs 中选出的项目。
        unselected_outputs (dict[int, Any]): cond_frame_outputs 中未选出的项目。

    示例：
        >>> frame_idx = 5
        >>> cond_frame_outputs = {1: "a", 3: "b", 7: "c", 9: "d"}
        >>> max_cond_frame_num = 2
        >>> selected, unselected = select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num)
        >>> print(selected)
        {3: 'b', 7: 'c'}
        >>> print(unselected)
        {1: 'a', 9: 'd'}
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # `frame_idx` 之前最近的条件帧（如果存在）
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # `frame_idx` 之后最近的条件帧（如果存在）
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # 添加时间上其他最近的条件帧，直到达到总数
        # `max_cond_frame_num` 个条件帧。
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000):
    """为给定位置和维度生成一维正弦位置嵌入。

    参数：
        pos_inds (torch.Tensor): 用于生成嵌入的位置索引。
        dim (int): 位置嵌入的维度，必须为偶数。
        temperature (float, 可选): 正弦函数频率的缩放因子。

    返回：
        (torch.Tensor): 正弦位置嵌入，形状为 (pos_inds.shape, dim)。

    示例：
        >>> pos = torch.tensor([0, 1, 2, 3])
        >>> embeddings = get_1d_sine_pe(pos, 128)
        >>> embeddings.shape
        torch.Size([4, 128])
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=pos_inds.dtype, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0):
    """为指定维度的网格初始化一维和二维坐标张量。

    此函数为尺寸为 end_x × end_y 的网格创建坐标张量，生成线性索引张量以及对应的 x、y 坐标张量。

    参数：
        end_x (int): 网格宽度（列数）。
        end_y (int): 网格高度（行数）。
        scale (float): 应用于坐标的缩放因子。
        offset (int): 添加到坐标上的偏移量。

    返回：
        t_x (torch.Tensor): 每个位置的 x 坐标，形状为 (end_x * end_y)。
        t_y (torch.Tensor): 每个位置的 y 坐标，形状为 (end_x * end_y)。

    示例：
        >>> t_x, t_y = init_t_xy(3, 2)
        >>> print(t_x)
        张量([0., 1., 2., 0., 1., 2.])
        >>> print(t_y)
        张量([0., 0., 0., 1., 1., 1.])
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0, scale_pos: float = 1.0):
    """为网格中的二维空间位置计算轴向复指数位置编码。

    此函数使用 x、y 维度各自的频率分量，为二维空间位置网格生成复指数位置编码。

    参数：
        dim (int): 位置编码的维度。
        end_x (int): 二维网格的宽度。
        end_y (int): 二维网格的高度。
        theta (float, 可选): 频率计算的缩放因子。
        scale_pos (float, 可选): 位置坐标的缩放因子。

    返回：
        (torch.Tensor): 复指数位置编码，形状为 (end_x*end_y, dim//2)。

    示例：
        >>> dim, end_x, end_y = 128, 8, 8
        >>> freqs_cis = compute_axial_cis(dim, end_x, end_y)
        >>> freqs_cis.shape
        torch.Size([64, 64])
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y, scale=scale_pos)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """调整频率张量形状，使其可以与输入张量进行广播。

    调整频率张量形状，确保其维度可与输入张量广播。此函数通常用于位置编码操作。

    参数：
        freqs_cis (torch.Tensor): 频率张量，其形状与 x 的最后两个维度匹配。
        x (torch.Tensor): 用于广播的输入张量。

    返回：
        (torch.Tensor): 重塑后的频率张量，可与输入张量进行广播。

    异常：
        AssertionError: freqs_cis 的形状与 x 的最后两个维度不匹配时抛出。
    """
    ndim = x.ndim
    assert ndim >= 2
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """对查询和键张量应用旋转位置编码。

    此函数使用复数频率分量对查询和键张量应用旋转位置编码（RoPE）。RoPE 是一种将相对位置信息注入自注意力机制的技术。

    参数：
        xq (torch.Tensor): 要使用位置信息编码的查询张量。
        xk (torch.Tensor): 要使用位置信息编码的键张量。
        freqs_cis (torch.Tensor): 用于旋转编码的复数频率分量，形状与 xq 的最后两个维度匹配。
        repeat_freqs_k (bool, 可选): 是否沿序列长度维重复频率分量，使其匹配键序列长度。

    返回：
        xq_out (torch.Tensor): Query 张量 with rotary positional encoding applied.
        xk_out (torch.Tensor): Key 张量 with rotary positional encoding applied, or original xk if xk is empty.

    示例：
        >>> import torch
        >>> xq = torch.randn(2, 8, 16, 64)  # [batch, heads, seq_len, dim]
        >>> xk = torch.randn(2, 8, 16, 64)
        >>> freqs_cis = compute_axial_cis(64, 4, 4)  # 4x4 空间网格，维度为 64
        >>> q_encoded, k_encoded = apply_rotary_enc(xq, xk, freqs_cis)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # 由于 dropout，没有需要旋转的键
        return xq_out.type_as(xq).to(xq.device), xk
    # 沿 seq_len 维重复频率，以匹配 k 的序列长度
    if repeat_freqs_k and (r := xk_.shape[-2] // xq_.shape[-2]) > 1:
        # MPS 不支持对复数张量执行 repeat，因此分解为实数表示
        if freqs_cis.device.type == "mps":
            freqs_cis = torch.view_as_real(freqs_cis)
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 3)), r, 1, 1)
            freqs_cis = torch.view_as_complex(freqs_cis.contiguous())
        else:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def window_partition(x: torch.Tensor, window_size: int):
    """将输入张量划分为不重叠窗口，并在需要时添加填充。

    参数：
        x (torch.Tensor): 输入张量，形状为 (B, H, W, C)。
        window_size (int): 每个窗口的尺寸。

    返回：
        windows (torch.Tensor): 划分后的窗口，形状为 (B * num_windows, window_size, window_size, C)。
        padded_h_w (tuple[int, int]): 划分前填充后的高度和宽度。

    示例：
        >>> x = torch.randn(1, 16, 16, 3)
        >>> windows, (Hp, Wp) = window_partition(x, window_size=4)
        >>> print(windows.shape, Hp, Wp)
        torch.Size([16, 4, 4, 3]) 16 16
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]):
    """将窗口序列还原为原始序列，并移除填充。

    此函数反转窗口划分过程，从窗口分段重建原始输入，并移除窗口划分过程中添加的所有填充。

    参数：
        windows (torch.Tensor): 窗口序列的输入张量，形状为 (B * num_windows, window_size, window_size, C)，
            其中 B 为批次大小，num_windows 为窗口数量，window_size 为每个窗口的尺寸，C 为通道数量。
        window_size (int): 每个窗口的尺寸。
        pad_hw (tuple[int, int]): 窗口划分前输入填充后的高度和宽度（Hp, Wp）。
        hw (tuple[int, int]): 输入填充和窗口划分前的原始高度和宽度（H, W）。

    返回：
        (torch.Tensor): 未划分窗口的序列，形状为 (B, H, W, C)，其中 B 为批次大小，H 和 W 为原始高度和宽度，C 为通道数量。

    示例：
        >>> windows = torch.rand(32, 8, 8, 64)  # 32 个 8x8 窗口，包含 64 个通道
        >>> pad_hw = (16, 16)  # 填充后的高度和宽度
        >>> hw = (15, 14)  # 原始高度和宽度
        >>> x = window_unpartition(windows, window_size=8, pad_hw=pad_hw, hw=hw)
        >>> print(x.shape)
        torch.Size([8, 15, 14, 64])
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """根据查询和键的尺寸提取相对位置嵌入。

    参数：
        q_size (int): 查询尺寸。
        k_size (int): 键尺寸。
        rel_pos (torch.Tensor): 相对位置嵌入，形状为 (L, C)，其中 L 是最大相对距离，C 是嵌入维度。

    返回：
        (torch.Tensor): 根据相对位置提取的位置嵌入，形状为 (q_size, k_size, C)。

    示例：
        >>> q_size, k_size = 8, 16
        >>> rel_pos = torch.randn(31, 64)  # 31 = 2 * max(8, 16) - 1
        >>> extracted_pos = get_rel_pos(q_size, k_size, rel_pos)
        >>> print(extracted_pos.shape)
        torch.Size([8, 16, 64])
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # 如有需要，插值相对位置。
    if rel_pos.shape[0] != max_rel_dist:
        # 插值相对位置编码。
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # 如果 q 和 k 的形状不同，则使用较短长度缩放坐标。
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """将分解的相对位置嵌入添加到注意力图中。

    此函数按照 MVITv2 论文中的方法计算并应用分解的相对位置嵌入，通过加入查询位置与键位置之间的空间关系增强注意力机制。

    参数：
        attn (torch.Tensor): 注意力图，形状为 (B, q_h * q_w, k_h * k_w)。
        q (torch.Tensor): 注意力层中的查询张量，形状为 (B, q_h * q_w, C)。
        rel_pos_h (torch.Tensor): 高度轴的相对位置嵌入，形状为 (Lh, C)。
        rel_pos_w (torch.Tensor): 宽度轴的相对位置嵌入，形状为 (Lw, C)。
        q_size (tuple[int, int]): 查询 q 的空间序列尺寸 (q_h, q_w)。
        k_size (tuple[int, int]): 键 k 的空间序列尺寸 (k_h, k_w)。

    返回：
        (torch.Tensor): Updated attention map with added relative positional embeddings, shape (B, q_h * q_w, k_h *
            k_w).

    示例：
        >>> B, C, q_h, q_w, k_h, k_w = 1, 64, 8, 8, 8, 8
        >>> attn = torch.rand(B, q_h * q_w, k_h * k_w)
        >>> q = torch.rand(B, q_h * q_w, C)
        >>> rel_pos_h = torch.rand(2 * max(q_h, k_h) - 1, C)
        >>> rel_pos_w = torch.rand(2 * max(q_w, k_w) - 1, C)
        >>> q_size, k_size = (q_h, q_w), (k_h, k_w)
        >>> updated_attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)
        >>> print(updated_attn.shape)
        torch.Size([1, 64, 64])

    参考：
        https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn


def get_abs_pos(
    abs_pos: torch.Tensor,
    has_cls_token: bool,
    hw: tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> torch.Tensor:
    """计算绝对位置嵌入；必要时调整嵌入尺寸，并移除 cls_token 维度以适配目标尺寸。
    原始嵌入。

    参数：
        abs_pos (torch.Tensor): 绝对位置嵌入，形状为 (1, num_position, C)。
        has_cls_token (bool): 为 True 时，abs_pos 中包含一个 cls token 嵌入。
        hw (tuple[int, int]): 输入图像令牌的尺寸。
        retain_cls_token (bool): 是否保留 cls_token。
        tiling (bool): 是否平铺嵌入，而不是进行插值（类似 abs_win）。

    返回：
        (torch.Tensor): 处理后的绝对位置嵌入；retain_cls_token 为 False 时形状为 (1, H, W, C)，否则为 (1, 1+H*W, C)。
    """
    if retain_cls_token:
        assert has_cls_token

    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile([1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])])[
                :, :, :h, :w
            ]
        else:
            new_abs_pos = F.interpolate(
                new_abs_pos,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

        if not retain_cls_token:
            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            # 添加回 cls_token，并展平空间维度
            assert has_cls_token
            return torch.cat(
                [cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)],
                dim=1,
            )

    else:
        if not retain_cls_token:
            return abs_pos.reshape(1, h, w, -1)
        else:
            assert has_cls_token
            return torch.cat([cls_pos, abs_pos], dim=1)


def concat_rel_pos(
    q: torch.Tensor,
    k: torch.Tensor,
    q_hw: tuple[int, int],
    k_hw: tuple[int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rescale: bool = False,
    relative_coords: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将相对位置系数拼接到 q 和 k 张量，使 qk^T 有效包含相对位置偏置。

    参数：
        q (torch.Tensor): 查询张量，形状为 (B, L_q, C)。
        k (torch.Tensor): 键张量，形状为 (B, L_k, C)。
        q_hw (tuple[int, int]): 查询张量的空间尺寸（高度、宽度）。
        k_hw (tuple[int, int]): 键张量的空间尺寸（高度、宽度）。
        rel_pos_h (torch.Tensor): 高度轴的相对位置嵌入。
        rel_pos_w (torch.Tensor): 宽度轴的相对位置嵌入。
        rescale (bool): 是否为 SDPA 使用重新缩放；由于拼接操作，SDPA 会使用错误的缩放因子。
        relative_coords (torch.Tensor | None): 预先计算的相对坐标索引张量。

    返回：
        q (torch.Tensor): Query 张量 padded so that qk^T accounts for relative position biases.
        k (torch.Tensor): Key 张量 padded so that qk^T accounts for relative position biases.
    """
    q_h, q_w = q_hw
    k_h, k_w = k_hw

    assert (q_h == q_w) and (k_h == k_w), "only square inputs supported"

    if relative_coords is not None:
        Rh = rel_pos_h[relative_coords]
        Rw = rel_pos_w[relative_coords]
    else:
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale  # for sdpa
    # attn 将除以 new_scale，但我们希望 q 除以 old_scale
    scale_ratio = new_scale / old_scale

    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) * new_scale  # (B, q_h, q_w, k_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) * new_scale  # (B, q_h, q_w, k_w)

    eye_h = torch.eye(k_h, dtype=q.dtype, device=q.device)
    eye_w = torch.eye(k_w, dtype=q.dtype, device=q.device)

    eye_h = eye_h.view(1, k_h, 1, k_h).expand([B, k_h, k_w, k_h])
    eye_w = eye_w.view(1, 1, k_w, k_w).expand([B, k_h, k_w, k_w])

    q = torch.cat([r_q * scale_ratio, rel_h, rel_w], dim=-1).view(B, q_h * q_w, -1)
    k = torch.cat([k.view(B, k_h, k_w, -1), eye_h, eye_w], dim=-1).view(B, k_h * k_w, -1)

    return q, k
