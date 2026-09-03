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
    两种 Retrieval Method
    在同一 Query 上的对比结果。

    left_recall_at_1 / right_recall_at_1:

        真正保存 Recall@1。

    left_reciprocal_rank / right_reciprocal_rank:

        使用 compare_methods() 指定的 metric_k
        对应的 Reciprocal Rank。

        项目当前通常使用：

            metric_k = 10

        因此这里实际表示：

            RR@10

    注意：

        Recall@1 和 Reciprocal Rank
        描述的是不同问题。

        对 multi-gold Query：

            Gold = A, B
            Top1 = A

        此时：

            Recall@1 = 0.5
            RR = 1.0

        因此不能使用 Recall@1
        判断“第一个 Gold 是否位于 Rank1”。
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


def _validate_metric_k(
    result: RetrievalMethodEvalResult,
    metric_k: int,
) -> None:
    """
    检查指定 metric_k
    是否存在于 Result 中。

    为什么单独封装：

    后面的多个 Failure Inspector
    都依赖 metrics_by_k。

    如果误传一个没有评测过的 K，
    应该显式失败，
    而不是静默产生不完整结果。
    """

    if metric_k <= 0:
        raise ValueError(
            "metric_k 必须大于 0"
        )

    for query_result in (
        result.query_results
    ):
        if (
            metric_k
            not in query_result.metrics_by_k
        ):
            raise ValueError(
                f"{result.method.value} "
                f"结果不包含 k={metric_k} 指标"
            )


def get_top1_failure_cases(
    result: RetrievalMethodEvalResult,
) -> list[RetrievalQueryEvalResult]:
    """
    找出 Recall@1 < 1.0 的 Query。

    注意：

    这个函数保留原来的语义：

        Recall@1 < 1.0

    它适合检查：

        Top1 是否覆盖了全部 Retrieval Gold。

    但它不等于：

        “Top1 不是 Gold”。

    例如 multi-gold Query：

        Gold = A, B
        Top1 = A

    此时：

        Recall@1 = 0.5

    所以这个 Query 会进入当前列表，
    但它实际上已经有 Gold 位于 Rank1。

    如果想检查：

        “第一个 Gold 是否位于 Rank1”

    应使用：

        find_non_top1_gold_cases()
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


def find_gold_miss_cases(
    result: RetrievalMethodEvalResult,
    metric_k: int = 10,
) -> list[RetrievalQueryEvalResult]:
    """
    找出 TopK 中完全没有任何 Gold 的 Query。

    判断依据：

        reciprocal_rank == 0.0

    为什么不用：

        recall_at_k < 1.0

    因为 multi-gold Query
    可能只召回部分 Gold：

        Gold = A, B
        Top10 中找到 A

    此时：

        Recall@10 = 0.5

    但不能说 Retriever
    “完全 miss”。

    真正的完全 miss 是：

        TopK 内一个 Gold 都没有

    这时 Reciprocal Rank 才会是：

        0.0
    """

    _validate_metric_k(
        result=result,
        metric_k=metric_k,
    )

    misses: list[
        RetrievalQueryEvalResult
    ] = []

    for query_result in (
        result.query_results
    ):
        metrics = (
            query_result.metrics_by_k[
                metric_k
            ]
        )

        if (
            metrics.reciprocal_rank
            == 0.0
        ):
            misses.append(
                query_result
            )

    return misses


def find_non_top1_gold_cases(
    result: RetrievalMethodEvalResult,
    metric_k: int = 10,
) -> list[RetrievalQueryEvalResult]:
    """
    找出“第一个 Gold 不在 Rank1”的 Query。

    判断依据：

        reciprocal_rank < 1.0

    包括两种情况：

    1. Gold 在 Rank2 / Rank3 / ...
       RR > 0，但小于 1；

    2. TopK 中完全没有 Gold
       RR = 0。

    这个 Inspector
    比 Recall@1 更适合分析：

        Retriever 的第一个核心证据
        是否真正排在第一位。

    对 multi-gold Query：

        Gold = A, B
        Rank1 = A

    即使：

        Recall@1 = 0.5

    仍然有：

        RR = 1.0

    因此不会被误判成
    “Gold 没有排第一”。
    """

    _validate_metric_k(
        result=result,
        metric_k=metric_k,
    )

    failures: list[
        RetrievalQueryEvalResult
    ] = []

    for query_result in (
        result.query_results
    ):
        metrics = (
            query_result.metrics_by_k[
                metric_k
            ]
        )

        if (
            metrics.reciprocal_rank
            < 1.0
        ):
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

    Recall@1：

        始终从 metrics_by_k[1]
        读取真正 Recall@1。

    Reciprocal Rank：

        从 metric_k 对应的指标读取。

        当前默认：

            metric_k = 10

        即项目当前主要观察的 MRR@10
        所使用的 Query-level RR。
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

        # ----------------------------------------------
        # Recall@1 必须真的从 k=1 读取。
        # ----------------------------------------------

        if 1 not in left_result.metrics_by_k:
            raise ValueError(
                "left result 不包含 k=1 指标"
            )

        if 1 not in right_result.metrics_by_k:
            raise ValueError(
                "right result 不包含 k=1 指标"
            )

        # ----------------------------------------------
        # RR 则使用调用方指定的 metric_k。
        # ----------------------------------------------

        if (
            metric_k
            not in left_result.metrics_by_k
        ):
            raise ValueError(
                "left result 不包含 "
                f"k={metric_k} 指标"
            )

        if (
            metric_k
            not in right_result.metrics_by_k
        ):
            raise ValueError(
                "right result 不包含 "
                f"k={metric_k} 指标"
            )

        left_top1_metrics = (
            left_result.metrics_by_k[1]
        )

        right_top1_metrics = (
            right_result.metrics_by_k[1]
        )

        left_rank_metrics = (
            left_result.metrics_by_k[
                metric_k
            ]
        )

        right_rank_metrics = (
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
                    left_top1_metrics.recall_at_k
                ),
                right_recall_at_1=(
                    right_top1_metrics.recall_at_k
                ),
                left_reciprocal_rank=(
                    left_rank_metrics.reciprocal_rank
                ),
                right_reciprocal_rank=(
                    right_rank_metrics.reciprocal_rank
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
    打印两种 Method 的逐 Query 对比结果。
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
                f"RR="
                f"{item.left_reciprocal_rank:.4f}"
            )
        )

        print(
            "Top5:",
            item.left_top_chunks,
        )

        print(
            (
                f"{item.right_method.value}: "
                f"RR="
                f"{item.right_reciprocal_rank:.4f}"
            )
        )

        print(
            "Top5:",
            item.right_top_chunks,
        )


def print_method_failure_cases(
    *,
    title: str,
    result: RetrievalMethodEvalResult,
    cases: Sequence[
        RetrievalQueryEvalResult
    ],
    metric_k: int = 10,
    top_n: int = 5,
) -> None:
    """
    打印单个 Retrieval Method 的 Failure Cases。

    用于：

        BM25 Gold Misses @10
        Dense Non-Top1 Cases
        Hybrid RRF Non-Top1 Cases
        Rerank Non-Top1 Cases

    输出：

        Query ID
        Query
        Role
        Gold
        RR
        Recall@K
        TopN

    这样 Failure Analysis
    不再只能比较两个 Method，
    也可以独立审计某个 Retriever。
    """

    if top_n <= 0:
        raise ValueError(
            "top_n 必须大于 0"
        )

    _validate_metric_k(
        result=result,
        metric_k=metric_k,
    )

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if not cases:
        print(
            "No cases."
        )

        return

    for query_result in cases:
        metrics = (
            query_result.metrics_by_k[
                metric_k
            ]
        )

        print()

        print(
            query_result.query_id,
            "|",
            query_result.query,
        )

        print(
            "Role:",
            query_result.role.value,
        )

        print(
            "Gold:",
            query_result.gold_chunk_ids,
        )

        print(
            (
                f"{result.method.value}: "
                f"RR={metrics.reciprocal_rank:.4f}, "
                f"Recall@{metric_k}="
                f"{metrics.recall_at_k:.4f}"
            )
        )

        print(
            f"Top{top_n}:",
            query_result
            .retrieved_chunk_ids[:top_n],
        )