# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import TORCH_2_4, select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 避免部分系统中的 OpenMP 冲突


class VisualAISearch:
    """利用 OpenAI CLIP 生成高质量图像和文本嵌入，并使用 NumPy 余弦相似度快速检索语义图像的搜索系统。

    此类将图像和文本嵌入对齐到共享的语义空间，使用户能够通过自然语言查询快速且准确地搜索大量图像。

    属性：
        data (str): 包含图像的目录。
        device (str): 计算设备，例如 'cpu' 或 'cuda'。
        index_path (str): 保存图像嵌入的 NumPy 文件路径。
        data_path_npy (str): 保存图像路径的 NumPy 文件路径。
        data_dir (Path): 数据目录的 Path 对象。
        model: 已加载的 CLIP 模型。
        index (np.ndarray): 用于余弦相似度搜索的 L2 归一化图像嵌入。
        image_paths (列表[str]): 图像文件路径列表。

    方法：
        extract_image_feature: 从图像中提取 CLIP 嵌入。
        extract_text_feature: 从文本中提取 CLIP 嵌入。
        load_or_build_index: 加载已有嵌入，或根据图像构建嵌入索引。
        search: 搜索与查询相似的图像。

    示例：
        初始化对象并搜索图像
        >>> searcher = VisualAISearch(data="path/to/images", device="cuda")
        >>> results = searcher.search("a cat sitting on a chair", k=10)
    """

    def __init__(self, **kwargs: Any) -> None:
        """使用嵌入索引和 CLIP 模型初始化 VisualAISearch 类。"""
        assert TORCH_2_4, f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})"
        from ultralytics.nn.text_model import build_text_model

        self.index_path = "embeddings.npy"
        self.data_path_npy = "paths.npy"
        self.data_dir = Path(kwargs.get("data", "images"))
        self.device = select_device(kwargs.get("device", "cpu"))

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            LOGGER.warning(f"{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")

        self.model = build_text_model("clip:ViT-B/32", device=self.device)

        self.index = None
        self.image_paths = []

        self.load_or_build_index()

    def extract_image_feature(self, path: Path) -> np.ndarray:
        """从给定图像路径提取 CLIP 图像嵌入。"""
        return self.model.encode_image(Image.open(path)).detach().cpu().numpy()

    def extract_text_feature(self, text: str) -> np.ndarray:
        """从给定文本查询中提取 CLIP 文本嵌入。"""
        return self.model.encode_text(self.model.tokenize([text])).detach().cpu().numpy()

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """对 `x` 的每一行执行 L2 归一化，使内积等于余弦相似度。

        参数：
            x (np.ndarray): 形状为 (N, D) 的特征数组。

        返回：
            (np.ndarray): 按行进行 L2 归一化且形状与输入相同的数组。
        """
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def load_or_build_index(self) -> None:
        """加载已有图像嵌入，或根据图像目录构建嵌入索引。

        检查磁盘上是否存在嵌入文件和图像路径文件。若存在则直接加载，否则从数据目录中的所有图像提取特征，
        执行 L2 归一化，并保存嵌入和图像路径供后续使用。
        """
        # 检查嵌入文件和对应的图像路径文件是否已经存在
        if Path(self.index_path).exists() and Path(self.data_path_npy).exists():
            LOGGER.info("正在加载已有嵌入...")
            self.index = np.load(self.index_path)  # 从磁盘加载 L2 归一化嵌入
            self.image_paths = np.load(self.data_path_npy)  # 加载已保存的图像路径列表
            return  # 索引已成功加载，退出函数

        # 如果嵌入不存在，则从头开始构建
        LOGGER.info("正在根据图像构建嵌入...")
        vectors = []  # 保存图像特征向量的列表

        # 遍历数据目录中的所有图像文件
        for file in self.data_dir.iterdir():
            # 跳过不是有效图像格式的文件
            if file.suffix.lower().lstrip(".") not in IMG_FORMATS:
                continue
            try:
                # 提取图像特征向量并添加到列表
                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)  # 保存对应的图像名称
            except Exception as e:
                LOGGER.warning(f"Skipping {file.name}: {e}")

        # 如果没有成功创建任何向量，则抛出错误
        if not vectors:
            raise RuntimeError("No image embeddings could be generated.")

        vectors = np.vstack(vectors).astype("float32")  # 将所有向量堆叠为 NumPy 数组并转换为 float32
        self.index = self._normalize(vectors)  # 执行 L2 归一化，使内积等于余弦相似度
        np.save(self.index_path, self.index)  # 将嵌入保存到磁盘
        np.save(self.data_path_npy, np.array(self.image_paths))  # 将图像路径列表保存到磁盘

        LOGGER.info(f"Indexed {len(self.image_paths)} images.")

    def search(self, query: str, k: int = 30, similarity_thresh: float = 0.1) -> list[str]:
        """返回与给定查询语义最相似的前 k 个图像。

        参数：
            query (str): 用于搜索的自然语言文本查询。
            k (int, 可选): 要返回的最大结果数量。
            similarity_thresh (float, 可选): 过滤结果时使用的最小相似度阈值。

        返回：
            (列表[str]): 按相似度分数排序的图像文件名列表。

        示例：
            搜索与查询匹配的图像
            >>> searcher = VisualAISearch(data="images")
            >>> results = searcher.search("red car", k=5, similarity_thresh=0.2)
        """
        text_feat = self._normalize(self.extract_text_feature(query).astype("float32"))
        scores = self.index @ text_feat[0]  # 计算余弦相似度（嵌入已进行 L2 归一化）
        top_k = np.argsort(scores)[::-1][: max(k, 0)]
        results = [(self.image_paths[i], float(scores[i])) for i in top_k if scores[i] >= similarity_thresh]
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.info("\n排序后的结果：")
        for name, score in results:
            LOGGER.info(f"  - {name} | 相似度：{score:.4f}")

        return [r[0] for r in results]

    def __call__(self, query: str) -> list[str]:
        """搜索函数的直接调用接口。"""
        return self.search(query)


class SearchApp:
    """基于 Flask 的语义图像搜索网页界面，支持自然语言查询。

    此类提供简洁、响应迅速的前端，用户可以输入自然语言查询，并立即查看从索引数据库中检索到的最相关图像。

    属性：
        render_template: Flask 模板渲染函数。
        request: Flask 请求对象。
        searcher (VisualAISearch): VisualAISearch 类实例。
        app (Flask): Flask 应用实例。

    方法：
        index: 处理用户查询并显示搜索结果。
        run: 启动 Flask 网页应用。

    示例：
        Start a search application
        >>> app = SearchApp(data="path/to/images", device="cuda")
        >>> app.run(debug=True)
    """

    def __init__(self, data: str = "images", device: str | None = None) -> None:
        """使用 VisualAISearch 后端初始化 SearchApp。

        参数：
            data (str, 可选): 用于建立索引和搜索的图像目录路径。
            device (str, 可选): 执行推理的设备（例如 'cpu'、'cuda'）。
        """
        check_requirements("flask>=3.0.1")
        from flask import Flask, render_template, request

        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=Path(data).resolve(),  # 用于提供图像的绝对路径
            static_url_path="/images",  # 图像 URL 前缀
        )
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self) -> str:
        """处理用户查询，并在网页界面中显示搜索结果。"""
        results = []
        if self.request.method == "POST":
            query = self.request.form.get("query", "").strip()
            results = self.searcher(query)
        return self.render_template("similarity-search.html", results=results)

    def run(self, debug: bool = False) -> None:
        """启动 Flask 网页应用服务器。"""
        self.app.run(debug=debug)
