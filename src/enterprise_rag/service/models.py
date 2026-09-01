"""QueryService 对外返回的数据模型。"""

from dataclasses import dataclass

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.generation.models import (
    Citation,
)


@dataclass(frozen=True)
class QueryResult:
    """
    一次完整 RAG Query 的业务结果。

    query:
        原始用户问题。

    role:
        当前请求实际使用的角色。

    answerable:
        系统最终是否认为当前知识库
        足以支持回答。

    answer:
        可回答时为最终文本。

        拒答时为 None。

    reason:
        回答 / 拒答原因。

    citations:
        经过程序校验后的确定性引用。

    retrieval_count:
        Reranker 最终返回的 Candidate 数量。

    top_rerank_score:
        当前最强 Evidence 的 rerank score。

        如果没有任何 Retrieval Result，
        则为 None。

    gate_reason:
        程序化 Coarse Relevance Gate
        的判定原因：

            passed
            no_results
            below_threshold

        注意：

        gate_reason == "passed"
        只代表通过粗粒度相关性检查，

        不代表最终一定 answerable。
    """

    query: str

    role: UserRole

    answerable: bool

    answer: str | None

    reason: str

    citations: tuple[Citation, ...]

    retrieval_count: int

    top_rerank_score: float | None

    gate_reason: str