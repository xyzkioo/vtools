# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils.checks import check_requirements


class LLM:
    """兼容 OpenAI 接口的大语言模型类。

    属性：
        model (str): 每次请求发送的模型名称。
        api (str): API 格式，可选 "responses" 或 "chat.completions"。
        base_url (str | None): 可选的兼容 OpenAI API 基础 URL。
        prompt (str | None): 添加在文本或图像输入前的可选提示词。
        overrides (dict): 传递给每次请求的默认参数。
        client (OpenAI | None): 延迟初始化的同步客户端。
        async_client (AsyncOpenAI | None): 延迟初始化的异步客户端。

    方法：
        __call__: 执行同步推理。
        async_call: 执行异步推理。

    示例：
        >>> from ultralytics import LLM
        >>> model = LLM("gpt-5.6-luna")
        >>> response = model("What is YOLO?")

        Analyze an 图像:
        >>> response = model("Describe this image", image="bus.jpg")

        Use the Chat Completions API:
        >>> model = LLM("gpt-5.6-luna", api="chat.completions")
        >>> response = model("What is YOLO?")
    """

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        api: str = "responses",
        base_url: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化兼容 OpenAI 接口的 LLM。

        参数：
            model (str): 模型名称。
            api (str): API 格式，可选 "responses" 或 "chat.completions"。
            base_url (str, 可选): 兼容 OpenAI API 的基础 URL。
            api_key (str, 可选): API 密钥。默认读取 OPENAI_API_KEY 环境变量。
            prompt (str, 可选): 添加在文本或图像输入前的提示词。
            **kwargs (Any): 传递给每次 API 请求的默认参数。
        """
        if api not in {"responses", "chat.completions"}:
            raise ValueError(f"Unsupported API format {api!r}. Use 'responses' or 'chat.completions'.")

        self.model = model
        self.api = api
        self.base_url = base_url
        self.prompt = prompt
        self.overrides = kwargs
        self.client = None
        self.async_client = None
        self._api_key = api_key

    def __call__(self, source: Any = None, image: Any = None, **kwargs: Any) -> Any:
        """使用已配置的模型执行推理。"""
        return self._call(self._prepare(source, image), kwargs)

    def _call(self, source: Any, kwargs: dict[str, Any]) -> Any:
        """通过同步客户端发送准备好的输入。"""
        request = self._request(source, kwargs)
        client = self._get_client()
        return (
            client.responses.create(**request) if self.api == "responses" else client.chat.completions.create(**request)
        )

    async def async_call(self, source: Any = None, image: Any = None, **kwargs: Any) -> Any:
        """使用已配置的模型执行异步推理。"""
        return await self._async_call(self._prepare(source, image), kwargs)

    async def _async_call(self, source: Any, kwargs: dict[str, Any]) -> Any:
        """通过异步客户端发送准备好的输入。"""
        request = self._request(source, kwargs)
        client = self._get_async_client()
        return (
            await client.responses.create(**request)
            if self.api == "responses"
            else await client.chat.completions.create(**request)
        )

    def _request(self, source: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """构建 Responses 或 Chat Completions 请求。

        参数：
            source (Any, 可选): Responses 输入或聊天消息。字符串会转换为 Chat Completions 的用户消息。
            kwargs (dict): 覆盖构造函数默认值的请求参数。

        返回：
            (dict): 原生 OpenAI SDK 请求参数。
        """
        request = {"model": self.model, **self.overrides, **kwargs}
        if self.api == "responses":
            if source is not None:
                request["input"] = source
        elif source is not None:
            request["messages"] = [{"role": "user", "content": source}] if isinstance(source, str) else source
        return request

    def _prepare(self, source: Any, image: Any = None) -> Any:
        """规范化文本或图像输入，同时保留原生消息载荷。"""
        if image is None:
            if source is None:
                return self.prompt
            if isinstance(source, (list, tuple, dict)):
                return source
            if isinstance(source, str):
                return f"{self.prompt}\n\n{source}" if self.prompt else source
            image = source
            prompt = self.prompt or "Describe the image."
        else:
            prompt = source or "Describe the image."
            if self.prompt:
                prompt = f"{self.prompt}\n\n{source}" if source else self.prompt
        image_url = self._image_url(image)
        if self.api == "responses":
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

    @staticmethod
    def _image_url(source: Any) -> str:
        """将图像 URL、路径或数组转换为 OpenAI 图像 URL。"""
        if isinstance(source, str) and source.startswith(("http://", "https://", "data:image/")):
            return source
        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
        else:
            image = (
                cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
                if isinstance(source, Image.Image)
                else np.asarray(source)
            )
        if image is None:
            raise ValueError(f"Unable to read image source {source!r}.")
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Unable to encode image source as JPEG.")
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

    def _get_client(self) -> Any:
        """首次推理时创建 OpenAI 客户端。"""
        if self.client is None:
            check_requirements("openai>=2.0.0")
            from openai import OpenAI

            kwargs = {k: v for k, v in {"api_key": self._api_key, "base_url": self.base_url}.items() if v is not None}
            self.client = OpenAI(**kwargs)
        return self.client

    def _get_async_client(self) -> Any:
        """首次推理时创建异步 OpenAI 客户端。"""
        if self.async_client is None:
            check_requirements("openai>=2.0.0")
            from openai import AsyncOpenAI

            kwargs = {k: v for k, v in {"api_key": self._api_key, "base_url": self.base_url}.items() if v is not None}
            self.async_client = AsyncOpenAI(**kwargs)
        return self.async_client
