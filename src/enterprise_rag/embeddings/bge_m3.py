"""BGE-M3 Dense Embedding 服务。"""

from collections.abc import Sequence

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


class BGEEmbeddingService:
    """
    对 BGE-M3 Dense Embedding 进行统一封装。

    当前 V1 仅启用 Dense Embedding。

    暂不启用：
    - Sparse Embedding
    - ColBERT Multi-vector

    当前已通过独立检查脚本实测：
    BGE-M3 Dense Embedding 维度为 1024。
    """

    EXPECTED_DIMENSION = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda:0",
        use_fp16: bool = True,
        max_length: int = 512,
    ) -> None:
        """
        初始化 BGE-M3 Embedding 服务。

        参数：
            model_name:
                Hugging Face 模型名称。

            device:
                模型推理设备。
                当前开发环境默认使用 cuda:0。

            use_fp16:
                是否使用 FP16 推理。
                RTX 4060 推理场景下开启，可以降低显存占用。

            max_length:
                单条输入允许的最大 Token 长度。
                当前先设为 512，
                后续根据真实 Chunk Token 统计结果确认。
        """

        # 如果明确配置 CUDA，
        # 就不允许 GPU 不可用时静默退回 CPU。
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "配置要求使用 CUDA，"
                    "但当前 PyTorch 无法访问 CUDA。"
                )

        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.max_length = max_length

        # 真正加载 BGE-M3 模型。
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            devices=[device],
        )

    @property
    def dimension(self) -> int:
        """
        返回 Dense Embedding 向量维度。

        当前 BGE-M3 已通过 check_bge_m3.py
        实际验证为 1024 维。
        """

        return self.EXPECTED_DIMENSION

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        批量生成文档 Dense Embedding。

        参数：
            texts:
                多条待向量化文本。

            batch_size:
                一次送入 GPU 的文本数量。
                当前 8GB 显存先采用较保守值。

        返回：
            NumPy 二维数组。

            shape:
                (文档数量, 1024)
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        output = self._model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=self.max_length,

            # V1 只需要 Dense。
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vectors = output["dense_vecs"]

        # 防止模型配置变化后静默产生错误维度。
        if dense_vectors.ndim != 2:
            raise RuntimeError(
                "文档 Embedding 输出维度异常："
                f"{dense_vectors.shape}"
            )

        if dense_vectors.shape[1] != self.dimension:
            raise RuntimeError(
                "文档 Embedding 向量维度异常："
                f"{dense_vectors.shape}"
            )

        return dense_vectors

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        为单条用户 Query 生成 Dense Embedding。

        返回：
            NumPy 一维数组。

            shape:
                (1024,)
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        output = self._model.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,

            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        query_vector = output[
            "dense_vecs"
        ][0]

        # 同样检查 Query 向量维度。
        if query_vector.shape != (
            self.dimension,
        ):
            raise RuntimeError(
                "Query Embedding 向量维度异常："
                f"{query_vector.shape}"
            )

        return query_vector