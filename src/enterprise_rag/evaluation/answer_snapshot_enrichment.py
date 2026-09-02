"""将最新 Evaluation Annotation 合并到旧 Full-RAG Snapshot。"""

from collections.abc import Sequence
from dataclasses import replace

from enterprise_rag.evaluation.answer_metrics import (
    AnswerEvalCaseResult,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
)


def enrich_answer_eval_results(
    *,
    results: Sequence[
        AnswerEvalCaseResult
    ],
    cases: Sequence[
        RetrievalEvalCase
    ],
) -> list[
    AnswerEvalCaseResult
]:
    """
    使用最新 Dataset Annotation 升级旧 Snapshot。

    这里只允许更新：

        citation_gold_chunk_ids
        strict_citation_eval

    不允许修改真实模型输出：

        answer
        cited_chunk_ids
        actual_answerable
        gate_reason
        reason
        retrieval_count
        top_rerank_score
        latency_ms

    所以该过程：

        不需要 GPU
        不需要 Qdrant
        不需要 LLM
    """

    if not results:
        raise ValueError(
            "results 不能为空"
        )

    if not cases:
        raise ValueError(
            "cases 不能为空"
        )

    case_by_query_id = {
        case.query_id: case
        for case in cases
    }

    enriched_results: list[
        AnswerEvalCaseResult
    ] = []

    for result in results:
        case = case_by_query_id.get(
            result.query_id
        )

        if case is None:
            raise ValueError(
                "Snapshot 中存在 Dataset "
                "无法匹配的 query_id："
                f"{result.query_id}"
            )

        # --------------------------------------------------
        # Query 文本也必须完全一致。
        #
        # 如果 query_id 一样，但问题文本已经被改过，
        # 就不能把旧模型输出和新 Gold Annotation
        # 强行拼在一起。
        # --------------------------------------------------

        if (
            result.query
            != case.query
        ):
            raise ValueError(
                f"{result.query_id} "
                "Snapshot Query 与 Dataset 不一致"
            )

        enriched_results.append(
            replace(
                result,
                citation_gold_chunk_ids=(
                    case.citation_gold_chunk_ids
                ),
                strict_citation_eval=(
                    case.strict_citation_eval
                ),
            )
        )

    return enriched_results