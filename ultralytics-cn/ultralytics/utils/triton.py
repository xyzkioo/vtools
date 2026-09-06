# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
from urllib.parse import urlsplit

import numpy as np


class TritonRemoteModel:
    """与远程 Triton Inference Server 模型交互的客户端。

    此类提供向 Triton Inference Server 发送推理请求并处理响应的便捷接口，支持 HTTP 和 gRPC 通信协议。

    属性：
        endpoint (str): Triton 服务器上的模型名称。
        url (str): Triton 服务器 URL。
        triton_client: Triton 客户端（HTTP 或 gRPC）。
        InferInput: Triton 客户端的输入类。
        InferRequestedOutput: Triton 客户端的输出请求类。
        input_formats (列表[str]): 模型输入的数据类型。
        np_input_formats (列表[type]): 模型输入的 NumPy 数据类型。
        input_names (列表[str]): 模型输入名称。
        output_names (列表[str]): 模型输出名称。
        metadata: 与模型关联的元数据。

    方法：
        __call__: 使用给定输入调用模型并返回输出。

    示例：
        使用 HTTP 初始化 Triton 客户端
        >>> model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8", scheme="http")

        使用 NumPy 数组执行推理
        >>> outputs = model(np.random.rand(1, 3, 640, 640).astype(np.float32))
    """

    def __init__(self, url: str, endpoint: str = "", scheme: str = ""):
        """初始化 TritonRemoteModel，以便与远程 Triton Inference Server 交互。

        参数可以单独提供，也可以从如下形式的完整 'url' 参数中解析：
        <scheme>://<netloc>/<endpoint>/<task_name>

        参数：
            url (str): Triton 服务器 URL。
            endpoint (str, 可选): Triton 服务器上的模型名称。
            scheme (str, 可选): 通信协议（'http' 或 'grpc'）。
        """
        if not endpoint and not scheme:  # 从 URL 字符串解析所有参数
            splits = urlsplit(url)
            endpoint = splits.path.strip("/").split("/", 1)[0]
            scheme = splits.scheme
            url = splits.netloc

        self.endpoint = endpoint
        self.url = url

        # 根据通信协议选择 Triton 客户端
        if scheme == "http":
            import tritonclient.http as client

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint)
        else:
            import tritonclient.grpc as client

            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint, as_json=True)["config"]

        # 按字母顺序排列输出名称，例如 'output0'、'output1' 等
        config["output"] = sorted(config["output"], key=lambda x: x.get("name"))

        # 定义模型属性
        type_map = {"TYPE_FP32": np.float32, "TYPE_FP16": np.float16, "TYPE_UINT8": np.uint8}
        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput
        self.input_formats = [x["data_type"] for x in config["input"]]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
        self.metadata = ast.literal_eval(config.get("parameters", {}).get("metadata", {}).get("string_value", "None"))

    def __call__(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """使用给定输入调用模型并返回推理结果。

        参数：
            *inputs (np.ndarray): 输入模型的数据。每个数组都应符合对应模型输入的预期形状和类型。

        返回：
            (列表[np.ndarray]): 转换为第一个输入数据类型的模型输出。列表中的每个元素对应一个模型输出张量。

        示例：
            >>> model = TritonRemoteModel(url="localhost:8000", endpoint="yolov8", scheme="http")
            >>> outputs = model(np.random.rand(1, 3, 640, 640).astype(np.float32))
        """
        infer_inputs = []
        input_format = inputs[0].dtype
        for i, x in enumerate(inputs):
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            infer_input = self.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace("TYPE_", ""))
            infer_input.set_data_from_numpy(x)
            infer_inputs.append(infer_input)

        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)

        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]
