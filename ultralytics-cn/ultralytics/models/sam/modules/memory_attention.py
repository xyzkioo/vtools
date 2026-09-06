# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import copy

import torch
from torch import nn

from .blocks import RoPEAttention


class MemoryAttentionLayer(nn.Module):
    """为神经网络实现带自注意力和交叉注意力机制的内存注意力层。

    此类结合自注意力、交叉注意力和前馈组件，处理输入张量并生成基于内存的注意力输出。

    属性：
        d_model (int): 模型隐藏状态的维度。
        dim_feedforward (int): 前馈网络的维度。
        dropout_value (float): 正则化使用的 dropout 比例。
        self_attn (RoPEAttention): 使用 RoPE（旋转位置嵌入）的自注意力机制。
        cross_attn_image (RoPEAttention): 用于图像处理的交叉注意力机制。
        linear1 (nn.Linear): 前馈网络的第一个线性层。
        linear2 (nn.Linear): 前馈网络的第二个线性层。
        norm1 (nn.LayerNorm): 自注意力输出的层归一化。
        norm2 (nn.LayerNorm): 交叉注意力输出的层归一化。
        norm3 (nn.LayerNorm): 前馈网络输出的层归一化。
        dropout1 (nn.Dropout): 自注意力后的 dropout 层。
        dropout2 (nn.Dropout): 交叉注意力后的 dropout 层。
        dropout3 (nn.Dropout): 前馈网络后的 dropout 层。
        activation (nn.ReLU): 前馈网络的激活函数。
        pos_enc_at_attn (bool): 是否在注意力处添加位置编码。
        pos_enc_at_cross_attn_queries (bool): 是否为交叉注意力查询添加位置编码。
        pos_enc_at_cross_attn_keys (bool): 是否为交叉注意力键添加位置编码。

    方法：
        forward: 对输入张量执行完整的内存注意力操作。
        _forward_sa: 对输入张量执行自注意力。
        _forward_ca: 在目标张量和内存张量之间执行交叉注意力。

    示例：
        >>> layer = MemoryAttentionLayer(d_model=256, dim_feedforward=2048, dropout=0.1)
        >>> tgt = torch.randn(1, 100, 256)
        >>> memory = torch.randn(1, 100, 64)
        >>> pos = torch.randn(1, 100, 256)
        >>> query_pos = torch.randn(1, 100, 256)
        >>> output = layer(tgt, memory, pos, query_pos)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_enc_at_attn: bool = False,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = False,
        self_attn: nn.Module | None = None,
        cross_attn: nn.Module | None = None,
    ):
        """初始化带自注意力、交叉注意力和前馈组件的内存注意力层。

        参数：
            d_model (int): 模型维度。
            dim_feedforward (int): 前馈网络维度。
            dropout (float): 正则化使用的 dropout 比例。
            pos_enc_at_attn (bool): 是否在注意力处添加位置编码。
            pos_enc_at_cross_attn_keys (bool): 是否为交叉注意力键添加位置编码。
            pos_enc_at_cross_attn_queries (bool): 是否为交叉注意力查询添加位置编码。
            self_attn (nn.Module | None): 自定义自注意力模块；为 None 时使用默认 RoPEAttention。
            cross_attn (nn.Module | None): 自定义交叉注意力模块；为 None 时使用默认 RoPEAttention。
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attn or RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1)
        self.cross_attn_image = cross_attn or RoPEAttention(
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=64,
        )

        # 实现前馈模型
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # 指定添加位置编码的位置
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        """使用位置编码和 RoPE 注意力机制对输入张量执行自注意力。"""
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None,
        pos: torch.Tensor | None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """使用 RoPEAttention 机制在目标张量和内存张量之间执行交叉注意力。"""
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # 交叉注意力
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """通过自注意力、交叉注意力和前馈网络层处理输入张量。

        参数：
            tgt (torch.Tensor): 用于自注意力的目标张量，形状为 (N, L, D)。
            memory (torch.Tensor): 用于交叉注意力的记忆张量，形状为 (N, S, D)。
            pos (torch.Tensor | None): 记忆张量的位置编码。
            query_pos (torch.Tensor | None): 目标张量的位置编码。
            num_k_exclude_rope (int): Number of keys to exclude from rotary position embedding.

        返回：
            (torch.Tensor): 经过注意力和前馈层处理后的张量，形状为 (N, L, D)。
        """
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    """使用自注意力和交叉注意力机制处理序列数据的内存注意力模块。

    此类实现结合自注意力和交叉注意力的多层注意力机制，用于处理序列数据，尤其适合 Transformer 类架构。

    属性：
        d_model (int): 模型隐藏状态的维度。
        layers (nn.ModuleList): MemoryAttentionLayer 模块列表。
        num_layers (int): 注意力层数量。
        norm (nn.LayerNorm): 应用于输出的层归一化。
        pos_enc_at_input (bool): 是否在输入处应用位置编码。
        batch_first (bool): 输入张量是否采用 batch-first 格式。

    方法：
        forward: 通过注意力层处理输入张量。

    示例：
        >>> d_model = 256
        >>> layer = MemoryAttentionLayer(d_model)
        >>> attention = MemoryAttention(d_model, pos_enc_at_input=True, layer=layer, num_layers=3)
        >>> curr = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
        >>> memory = torch.randn(20, 32, d_model)  # (mem_len, batch_size, d_model)
        >>> curr_pos = torch.randn(10, 32, d_model)
        >>> memory_pos = torch.randn(20, 32, d_model)
        >>> output = attention(curr, memory, curr_pos, memory_pos)
        >>> print(output.shape)
        torch.Size([10, 32, 256])
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # 层是否期望批次维度位于第一维？
    ):
        """使用指定层和归一化配置初始化用于序列数据处理的 MemoryAttention。

        此类实现结合自注意力和交叉注意力的多层注意力机制，用于处理序列数据，尤其适合 Transformer 类架构。

        参数：
            d_model (int): 模型隐藏状态的维度。
            pos_enc_at_input (bool): 是否在输入处应用位置编码。
            layer (nn.Module): 模块中使用的注意力层。
            num_layers (int): 注意力层数量。
            batch_first (bool): 输入张量是否采用 batch-first 格式。
        """
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # 自注意力输入
        memory: torch.Tensor,  # 交叉注意力输入
        curr_pos: torch.Tensor | None = None,  # 自注意力输入的位置编码
        memory_pos: torch.Tensor | None = None,  # 交叉注意力输入的位置编码
        num_obj_ptr_tokens: int = 0,  # 对象指针 token 的数量
    ) -> torch.Tensor:
        """通过注意力层处理输入，并使用位置编码应用自注意力和交叉注意力。

        参数：
            curr (torch.Tensor): 自注意力输入张量，表示当前状态。
            memory (torch.Tensor): 交叉注意力输入张量，表示内存信息。
            curr_pos (torch.Tensor | None): 自注意力输入的位置编码。
            memory_pos (torch.Tensor | None): 交叉注意力输入的位置编码。
            num_obj_ptr_tokens (int): 从旋转位置嵌入中排除的对象指针令牌数量。

        返回：
            (torch.Tensor): 应用注意力层和归一化后的输出张量。

        示例：
            >>> d_model = 256
            >>> layer = MemoryAttentionLayer(d_model)
            >>> attention = MemoryAttention(d_model, pos_enc_at_input=True, layer=layer, num_layers=3)
            >>> curr = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
            >>> memory = torch.randn(20, 32, d_model)  # (mem_len, batch_size, d_model)
            >>> curr_pos = torch.randn(10, 32, d_model)
            >>> memory_pos = torch.randn(20, 32, d_model)
            >>> output = attention(curr, memory, curr_pos, memory_pos)
            >>> print(output.shape)
            torch.Size([10, 32, 256])
        """
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        assert curr.shape[1] == memory.shape[1], "curr 和 memory 的批次大小必须相同"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # 转换为 batch-first 格式
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # 先转换回序列优先格式
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
