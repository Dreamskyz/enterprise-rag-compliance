"""BGE Reranker 服务封装。"""

from collections.abc import Sequence

import torch
from FlagEmbedding import FlagReranker


class BGERerankerService:
    """
    BGE Cross-Encoder Reranker。

    当前模型：
        BAAI/bge-reranker-v2-m3

    当前职责：
        Query + 多个 Passage
            ↓
        Cross-Encoder
            ↓
        relevance scores
    """

    def __init__(
        self,
        model_name: str = (
            "BAAI/bge-reranker-v2-m3"
        ),
        device: str = "cuda:0",
        use_fp16: bool = True,
    ) -> None:
        """
        初始化 Reranker。

        参数：
            model_name:
                Hugging Face 模型名称。

            device:
                当前开发环境默认 cuda:0。

            use_fp16:
                GPU 推理使用 FP16，
                降低显存并提升推理速度。
        """

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "配置要求使用 CUDA，"
                    "但当前 CUDA 不可用。"
                )

        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16

        # FlagReranker 官方支持指定 devices。
        # 当前只有一张 GPU，因此明确使用 cuda:0，
        # 避免不必要的多设备初始化。
        self._reranker = FlagReranker(
            model_name,
            use_fp16=use_fp16,
            devices=device,
        )

    def compute_scores(
        self,
        query: str,
        passages: Sequence[str],
    ) -> list[float]:
        """
        对 Query 与多个 Passage 计算相关性分数。

        参数：
            query:
                用户问题。

            passages:
                候选文档文本。

        返回：
            与 passages 顺序完全对应的
            rerank score 列表。

        注意：
            score 越大，代表 Query 与 Passage
            越相关。
        """

        if not query.strip():
            raise ValueError(
                "query 不能为空"
            )

        if not passages:
            return []

        pairs = [
            [
                query,
                passage,
            ]
            for passage in passages
        ]

        scores = (
            self._reranker.compute_score(
                pairs,
                normalize=False,
            )
        )

        # FlagEmbedding 在只有一个 pair 时，
        # 某些接口场景可能返回单个 float。
        #
        # 这里统一转换成 list[float]，
        # 让上层接口保持稳定。
        if isinstance(
            scores,
            (int, float),
        ):
            return [
                float(scores)
            ]

        return [
            float(score)
            for score in scores
        ]