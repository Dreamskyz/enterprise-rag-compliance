"""基于 Rerank Score 的 Evidence Gate。"""

import math
from collections.abc import Sequence

from enterprise_rag.evidence.models import (
    EvidenceDecisionReason,
    EvidenceGateDecision,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
)


class EvidenceGate:
    """
    判断当前 Retrieval Evidence
    是否足以进入 LLM Generation。

    V1 Baseline：

        Reranked Results
              ↓
        读取 Top1 Rerank Score
              ↓
        与 calibrated threshold 比较
              ↓
          pass / reject

    注意：

    threshold 不在这里写死。

    正式值必须以后通过
    Answerable / Unanswerable
    Evaluation Set 标定。
    """

    def __init__(
        self,
        min_top_score: float,
    ) -> None:
        """
        初始化 Evidence Gate。

        参数：
            min_top_score:
                Top1 Rerank Score
                至少达到多少，
                才允许进入 Generation。

        当前不提供默认值，
        防止未经评测就偷偷形成
        “拍脑袋阈值”。
        """

        if not math.isfinite(
            min_top_score
        ):
            raise ValueError(
                "min_top_score 必须是有限数值"
            )

        self.min_top_score = float(
            min_top_score
        )

    def evaluate(
        self,
        results: Sequence[
            RerankedSearchResult
        ],
    ) -> EvidenceGateDecision:
        """
        判断当前证据是否足够。

        当前 V1 规则：

        1. 没有 Retrieval Result
           → Reject

        2. Top1 Rerank Score
           < threshold
           → Reject

        3. Top1 Rerank Score
           >= threshold
           → Pass

        注意：

        这里假设输入 results 已经由
        RerankedRetriever 按 rerank_score
        从高到低排序。
        """

        # --------------------------------------------------
        # Case 1:
        # 完全没有证据。
        # --------------------------------------------------

        if not results:
            return EvidenceGateDecision(
                passed=False,
                reason=(
                    EvidenceDecisionReason.NO_RESULTS
                ),
                top_score=None,
                threshold=self.min_top_score,
            )

        # --------------------------------------------------
        # Case 2:
        # 使用最强证据 Top1 Score。
        # --------------------------------------------------

        top_score = float(
            results[0].rerank_score
        )

        if not math.isfinite(
            top_score
        ):
            raise ValueError(
                "Top1 rerank_score "
                "必须是有限数值"
            )

        # --------------------------------------------------
        # Case 3:
        # 最强 Evidence 仍不足。
        # --------------------------------------------------

        if (
            top_score
            < self.min_top_score
        ):
            return EvidenceGateDecision(
                passed=False,
                reason=(
                    EvidenceDecisionReason
                    .BELOW_THRESHOLD
                ),
                top_score=top_score,
                threshold=self.min_top_score,
            )

        # --------------------------------------------------
        # Case 4:
        # Evidence 达到要求。
        # --------------------------------------------------

        return EvidenceGateDecision(
            passed=True,
            reason=(
                EvidenceDecisionReason.PASSED
            ),
            top_score=top_score,
            threshold=self.min_top_score,
        )