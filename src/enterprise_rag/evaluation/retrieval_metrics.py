"""Retrieval Evaluation Metrics。"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetricResult:
    """
    单条 Query 的 Retrieval 指标结果。
    """

    recall_at_k: float

    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalAggregateMetrics:
    """
    Dataset 级 Retrieval 指标。
    """

    query_count: int

    mean_recall: float

    mrr: float


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """
    计算单条 Query 的 Recall@K。

    Recall@K
    =
    TopK 中命中的 Gold Evidence 数
    /
    Gold Evidence 总数
    """

    if k <= 0:
        raise ValueError(
            "k 必须大于 0"
        )

    if not gold_chunk_ids:
        raise ValueError(
            "gold_chunk_ids 不能为空"
        )

    gold_set = set(
        gold_chunk_ids
    )

    top_k_ids = (
        retrieved_chunk_ids[:k]
    )

    matched_gold_ids = (
        gold_set.intersection(
            top_k_ids
        )
    )

    return (
        len(matched_gold_ids)
        / len(gold_set)
    )


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    k: int | None = None,
) -> float:
    """
    计算单条 Query 的 Reciprocal Rank。

    找到第一个 Gold Evidence 的排名：

        RR = 1 / rank

    如果 TopK 内没有命中：

        RR = 0
    """

    if not gold_chunk_ids:
        raise ValueError(
            "gold_chunk_ids 不能为空"
        )

    if (
        k is not None
        and k <= 0
    ):
        raise ValueError(
            "k 必须大于 0"
        )

    gold_set = set(
        gold_chunk_ids
    )

    if k is None:
        candidates = (
            retrieved_chunk_ids
        )
    else:
        candidates = (
            retrieved_chunk_ids[:k]
        )

    for rank, chunk_id in enumerate(
        candidates,
        start=1,
    ):
        if chunk_id in gold_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval_case(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    k: int,
) -> RetrievalMetricResult:
    """
    同时计算单条 Query 的：

        Recall@K
        Reciprocal Rank
    """

    return RetrievalMetricResult(
        recall_at_k=recall_at_k(
            retrieved_chunk_ids=(
                retrieved_chunk_ids
            ),
            gold_chunk_ids=(
                gold_chunk_ids
            ),
            k=k,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_chunk_ids=(
                retrieved_chunk_ids
            ),
            gold_chunk_ids=(
                gold_chunk_ids
            ),
            k=k,
        ),
    )


def aggregate_retrieval_metrics(
    results: Sequence[
        RetrievalMetricResult
    ],
) -> RetrievalAggregateMetrics:
    """
    汇总多个 Query 的：

        Mean Recall@K
        MRR
    """

    if not results:
        raise ValueError(
            "results 不能为空"
        )

    mean_recall = (
        sum(
            result.recall_at_k
            for result in results
        )
        / len(results)
    )

    mrr = (
        sum(
            result.reciprocal_rank
            for result in results
        )
        / len(results)
    )

    return RetrievalAggregateMetrics(
        query_count=len(results),
        mean_recall=mean_recall,
        mrr=mrr,
    )