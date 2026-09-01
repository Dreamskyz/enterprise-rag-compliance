"""Evidence Gate 相关数据模型。"""

from dataclasses import dataclass
from enum import StrEnum


class EvidenceDecisionReason(StrEnum):
    """
    Evidence Gate 的判定原因。

    PASSED:
        当前检索证据达到阈值，
        可以继续进入生成阶段。

    NO_RESULTS:
        Retrieval 没有任何候选。

    BELOW_THRESHOLD:
        有候选，
        但最强证据分数仍低于阈值。
    """

    PASSED = "passed"

    NO_RESULTS = "no_results"

    BELOW_THRESHOLD = "below_threshold"


@dataclass(frozen=True)
class EvidenceGateDecision:
    """
    Evidence Gate 的结构化判定结果。

    passed:
        是否允许进入 LLM Generation。

    reason:
        本次判定原因。

    top_score:
        当前 Top1 Rerank Score。

        如果 Retrieval 没结果，
        则为 None。

    threshold:
        当前 Gate 使用的阈值。
    """

    passed: bool

    reason: EvidenceDecisionReason

    top_score: float | None

    threshold: float