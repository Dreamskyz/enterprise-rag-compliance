"""测试 Retrieval Evaluation Metrics。"""

import pytest

from enterprise_rag.evaluation.retrieval_metrics import (
    RetrievalMetricResult,
    aggregate_retrieval_metrics,
    evaluate_retrieval_case,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_single_gold_hit() -> None:
    """单 Gold 在 Top3 中命中。"""

    score = recall_at_k(
        retrieved_chunk_ids=[
            "a",
            "b",
            "gold",
            "c",
        ],
        gold_chunk_ids=[
            "gold"
        ],
        k=3,
    )

    assert score == 1.0


def test_recall_at_k_single_gold_miss() -> None:
    """Gold 不在 TopK 中时 Recall=0。"""

    score = recall_at_k(
        retrieved_chunk_ids=[
            "a",
            "b",
            "gold",
        ],
        gold_chunk_ids=[
            "gold"
        ],
        k=2,
    )

    assert score == 0.0


def test_recall_at_k_multiple_gold_partial_hit() -> None:
    """
    两个 Gold 只命中一个：

        Recall = 1 / 2
    """

    score = recall_at_k(
        retrieved_chunk_ids=[
            "gold-a",
            "x",
            "y",
        ],
        gold_chunk_ids=[
            "gold-a",
            "gold-b",
        ],
        k=3,
    )

    assert score == 0.5


def test_reciprocal_rank_first() -> None:
    """Gold 排第 1，RR=1。"""

    score = reciprocal_rank(
        retrieved_chunk_ids=[
            "gold",
            "x",
        ],
        gold_chunk_ids=[
            "gold"
        ],
    )

    assert score == 1.0


def test_reciprocal_rank_second() -> None:
    """Gold 排第 2，RR=0.5。"""

    score = reciprocal_rank(
        retrieved_chunk_ids=[
            "x",
            "gold",
        ],
        gold_chunk_ids=[
            "gold"
        ],
    )

    assert score == 0.5


def test_reciprocal_rank_respects_k() -> None:
    """
    Gold 在 rank=3，
    但只评 Top2，应返回 0。
    """

    score = reciprocal_rank(
        retrieved_chunk_ids=[
            "a",
            "b",
            "gold",
        ],
        gold_chunk_ids=[
            "gold"
        ],
        k=2,
    )

    assert score == 0.0


def test_reciprocal_rank_multiple_gold_uses_first() -> None:
    """
    多 Gold 时取最先出现的那个。
    """

    score = reciprocal_rank(
        retrieved_chunk_ids=[
            "x",
            "gold-b",
            "gold-a",
        ],
        gold_chunk_ids=[
            "gold-a",
            "gold-b",
        ],
    )

    assert score == 0.5


def test_evaluate_retrieval_case() -> None:
    """单 Case 应同时返回 Recall 和 RR。"""

    result = evaluate_retrieval_case(
        retrieved_chunk_ids=[
            "x",
            "gold",
        ],
        gold_chunk_ids=[
            "gold"
        ],
        k=5,
    )

    assert (
        result.recall_at_k
        == 1.0
    )

    assert (
        result.reciprocal_rank
        == 0.5
    )


def test_aggregate_retrieval_metrics() -> None:
    """Dataset 级指标应正确求平均。"""

    metrics = (
        aggregate_retrieval_metrics(
            [
                RetrievalMetricResult(
                    recall_at_k=1.0,
                    reciprocal_rank=1.0,
                ),
                RetrievalMetricResult(
                    recall_at_k=0.5,
                    reciprocal_rank=0.5,
                ),
                RetrievalMetricResult(
                    recall_at_k=0.0,
                    reciprocal_rank=0.0,
                ),
            ]
        )
    )

    assert (
        metrics.query_count
        == 3
    )

    assert metrics.mean_recall == (
        pytest.approx(0.5)
    )

    assert metrics.mrr == (
        pytest.approx(0.5)
    )


def test_recall_rejects_empty_gold() -> None:
    """无 Gold Query 不应参与 Recall。"""

    with pytest.raises(
        ValueError,
        match="gold_chunk_ids",
    ):
        recall_at_k(
            retrieved_chunk_ids=[],
            gold_chunk_ids=[],
            k=5,
        )


def test_reciprocal_rank_rejects_empty_gold() -> None:
    """无 Gold Query 不应参与 RR。"""

    with pytest.raises(
        ValueError,
        match="gold_chunk_ids",
    ):
        reciprocal_rank(
            retrieved_chunk_ids=[],
            gold_chunk_ids=[],
        )


def test_recall_rejects_invalid_k() -> None:
    """K 必须大于 0。"""

    with pytest.raises(
        ValueError,
        match="k",
    ):
        recall_at_k(
            retrieved_chunk_ids=[
                "gold"
            ],
            gold_chunk_ids=[
                "gold"
            ],
            k=0,
        )