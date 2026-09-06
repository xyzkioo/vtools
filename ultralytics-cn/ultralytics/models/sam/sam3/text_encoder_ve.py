# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model_misc import LayerScale


class ResidualAttentionBlock(nn.Module):
    """包含多头注意力、层归一化和 MLP 前馈网络的 Transformer 块。"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        """初始化可配置维度和归一化方式的残差注意力块。"""
        super().__init__()
        # Attention
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        # LayerNorm, LayerScale
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        # MLP
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def attention(
        self, q_x: torch.Tensor, k_x: torch.Tensor = None, v_x: torch.Tensor = None, attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """计算多头注意力，并支持可选的交叉注意力和掩码。"""
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None and attn_mask.dtype != torch.bool:  # 保持布尔掩码不变
            attn_mask = attn_mask.to(q_x.dtype)

        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self, q_x: torch.Tensor, k_x: torch.Tensor = None, v_x: torch.Tensor = None, attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """应用带层归一化和 MLP 的残差注意力，并支持可选交叉注意力。"""
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    """由残差注意力块堆叠而成的 Transformer 编码器，支持可选的梯度检查点。"""

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        compile_mode: str | None = None,
        use_act_checkpoint: bool = False,
    ):
        """初始化具有可配置深度、宽度以及可选编译和检查点功能的 Transformer。"""
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = use_act_checkpoint
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        )

        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)
            if self.grad_checkpointing:
                torch._dynamo.config.optimize_ddp = False

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """在训练期间通过所有 Transformer 块处理输入，并可选使用梯度检查点。"""
        for _, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


def text_global_pool(
    x: torch.Tensor, text: torch.Tensor = None, pool_type: str = "argmax"
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用指定的池化策略从文本嵌入中提取池化表示和词元。
    (first/last/argmax/none).
    """
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        # 从 eot 嵌入中提取特征（eot_token 是每个序列中数值最大的令牌）
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(nn.Module):
    """带因果掩码和灵活池化策略的文本 Transformer 编码器。"""

    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "none",  # no pooling
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        output_tokens: bool = False,
        use_ln_post: bool = True,
        compile_mode: str | None = None,
        use_act_checkpoint: bool = False,
    ):
        """初始化文本 Transformer，包括嵌入层、Transformer 块和池化选项。"""
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
        )
        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer("attn_mask", self.build_causal_mask(), persistent=False)
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self) -> torch.Tensor:
        """创建因果注意力掩码，防止关注未来词元。"""
            # 延迟创建因果注意力掩码，令牌之间使用完整注意力
        # PyTorch 使用加性注意力掩码；使用 -inf 填充
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 将下三角区域置零
        return mask

    def forward(self, text: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """执行文本 Transformer 的前向传播，返回池化输出，并可选返回词元嵌入。"""
        seq_len = text.shape[1]
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled


class VETextEncoder(nn.Module):
    """用于视觉编码器（VE）模型的文本编码器，由文本 Transformer 和线性调整器组成。"""

    def __init__(
        self,
        d_model: int,
        tokenizer: Callable,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: str | None = None,
        use_act_checkpoint: bool = True,
    ):
        """初始化 VE 文本编码器，并使用线性调整器匹配解码器维度。"""
        super().__init__()
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer

        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            # 需要令牌序列，而不仅是池化输出
            output_tokens=True,
            use_ln_post=use_ln_post,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
        )
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def forward(
        self, text: list[str] | tuple[torch.Tensor, torch.Tensor, dict], input_boxes: list | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码文本输入（原始字符串或预编码张量），并调整到解码器维度。"""
        if isinstance(text[0], str):
            # 当前没有使用场景
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            # 编码文本
            tokenized = self.tokenizer(text, context_length=self.context_length).to(
                self.resizer.weight.device
            )  # [b, seq_len]
            text_attention_mask = (tokenized != 0).bool()

            # 手动嵌入令牌
            inputs_embeds = self.encoder.token_embedding(tokenized)  # [b, seq_len, d=1024]
            _, text_memory = self.encoder(tokenized)  # [b, seq_len, d=1024]

            assert text_memory.shape[1] == inputs_embeds.shape[1]
            # 反转注意力掩码，因为它与 PyTorch Transformer 中的定义相反
            text_attention_mask = text_attention_mask.ne(1)
            # 转置记忆特征，因为 PyTorch 注意力模块要求序列维在前
            text_memory = text_memory.transpose(0, 1)
            # 将编码器隐藏状态调整为与解码器相同的 d_model
            text_memory_resized = self.resizer(text_memory)
        else:
            # 文本已经编码，直接使用。
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert input_boxes is None or len(input_boxes) == 0, "Can't replace boxes in text if it's already encoded"

        # 注意，input_embeds 按 PyTorch 约定返回（序列优先）
        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )
