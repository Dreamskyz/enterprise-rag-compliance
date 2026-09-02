"""Retrieval Ablation Failure Case 分析。"""

from collections.abc import Sequence
from dataclasses import dataclass

from enterprise_rag.evaluation.retrieval_runner import (
    RetrievalMethod,
    RetrievalMethodEvalResult,
    RetrievalQueryEvalResult,
)


@dataclass(frozen=True)
class RetrievalComparisonCase:
    """
    两种 Retrieval Method 在同一 Query 上的对比结果。
    """

    query_id: str

    query: str

    left_method: RetrievalMethod

    right_method: RetrievalMethod

    left_recall_at_1: float

    right_recall_at_1: float

    left_reciprocal_rank: float

    right_reciprocal_rank: float

    left_top_chunks: tuple[str, ...]

    right_top_chunks: tuple[str, ...]

    gold_chunk_ids: tuple[str, ...]


def get_top1_failure_cases(
    result: RetrievalMethodEvalResult,
) -> list[RetrievalQueryEvalResult]:
    """
    找出 Recall@1 < 1.0 的 Query。

    注意：

    多 Gold Query 只命中部分 Gold 时，
    Recall@1 也会小于 1。

    因此这里叫 Top1 Failure Case，
    但不等于“Top1 完全无相关结果”。

    后续要结合 MRR / Gold 一起看。
    """

    failures: list[
        RetrievalQueryEvalResult
    ] = []

    for query_result in (
        result.query_results
    ):
        metrics = (
            query_result.metrics_by_k[1]
        )

        if metrics.recall_at_k < 1.0:
            failures.append(
                query_result
            )

    return failures


def compare_methods(
    left: RetrievalMethodEvalResult,
    right: RetrievalMethodEvalResult,
    metric_k: int = 10,
) -> list[RetrievalComparisonCase]:
    """
    按 query_id 对齐两种 Method，
    返回逐 Query 对比结果。
    """

    left_map = {
        result.query_id: result
        for result in left.query_results
    }

    right_map = {
        result.query_id: result
        for result in right.query_results
    }

    if set(left_map) != set(right_map):
        raise ValueError(
            "两种 Retrieval Method "
            "评测的 query_id 集合不一致"
        )

    comparisons: list[
        RetrievalComparisonCase
    ] = []

    for query_id in sorted(
        left_map
    ):
        left_result = left_map[
            query_id
        ]

        right_result = right_map[
            query_id
        ]

        if metric_k not in left_result.metrics_by_k:
            raise ValueError(
                f"left result 不包含 k={metric_k} 指标"
            )

        if metric_k not in right_result.metrics_by_k:
            raise ValueError(
                f"right result 不包含 k={metric_k} 指标"
            )

        left_metrics = (
            left_result.metrics_by_k[
                metric_k
            ]
        )

        right_metrics = (
            right_result.metrics_by_k[
                metric_k
            ]
        )

        comparisons.append(
            RetrievalComparisonCase(
                query_id=query_id,
                query=left_result.query,
                left_method=left.method,
                right_method=right.method,
                left_recall_at_1=(
                    left_metrics.recall_at_k
                ),
                right_recall_at_1=(
                    right_metrics.recall_at_k
                ),
                left_reciprocal_rank=(
                    left_metrics.reciprocal_rank
                ),
                right_reciprocal_rank=(
                    right_metrics.reciprocal_rank
                ),
                left_top_chunks=(
                    left_result
                    .retrieved_chunk_ids[:5]
                ),
                right_top_chunks=(
                    right_result
                    .retrieved_chunk_ids[:5]
                ),
                gold_chunk_ids=(
                    left_result.gold_chunk_ids
                ),
            )
        )

    return comparisons


def find_rank_degradations(
    left: RetrievalMethodEvalResult,
    right: RetrievalMethodEvalResult,
) -> list[RetrievalComparisonCase]:
    """
    找出从 left → right 后，
    Reciprocal Rank 下降的 Query。

    例如：

        Dense
          ↓
        Hybrid RRF

    如果 RR 下降，
    说明融合后最靠前的 Gold 排名变差。
    """

    comparisons = compare_methods(
        left=left,
        right=right,
    )

    return [
        item
        for item in comparisons
        if (
            item.right_reciprocal_rank
            <
            item.left_reciprocal_rank
        )
    ]


def find_rank_improvements(
    left: RetrievalMethodEvalResult,
    right: RetrievalMethodEvalResult,
) -> list[RetrievalComparisonCase]:
    """
    找出从 left → right 后，
    Reciprocal Rank 提升的 Query。
    """

    comparisons = compare_methods(
        left=left,
        right=right,
    )

    return [
        item
        for item in comparisons
        if (
            item.right_reciprocal_rank
            >
            item.left_reciprocal_rank
        )
    ]


def print_comparison_cases(
    title: str,
    cases: Sequence[
        RetrievalComparisonCase
    ],
) -> None:
    """
    打印逐 Query 对比结果。
    """

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if not cases:
        print(
            "No cases."
        )

        return

    for item in cases:
        print()

        print(
            item.query_id,
            "|",
            item.query,
        )

        print(
            "Gold:",
            item.gold_chunk_ids,
        )

        print(
            (
                f"{item.left_method.value}: "
                f"RR={item.left_reciprocal_rank:.4f}"
            )
        )

        print(
            "Top5:",
            item.left_top_chunks,
        )

        print(
            (
                f"{item.right_method.value}: "
                f"RR={item.right_reciprocal_rank:.4f}"
            )
        )

        print(
            "Top5:",
            item.right_top_chunks,
        )