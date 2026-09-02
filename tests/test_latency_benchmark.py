"""测试 Retrieval Latency Benchmark。"""

import pytest

from enterprise_rag.acl.models import (
    AccessContext,
)
from enterprise_rag.evaluation.latency_benchmark import (
    benchmark_retrieval_method,
    percentile,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalCategory,
)
from enterprise_rag.evaluation.retrieval_runner import (
    RetrievalMethod,
)


def make_case(
    query_id: str,
) -> RetrievalEvalCase:
    """构造 Benchmark 测试 Case。"""

    return RetrievalEvalCase(
        query_id=query_id,
        query=f"测试问题 {query_id}",
        gold_chunk_ids=(
            "gold",
        ),
        category=(
            RetrievalEvalCategory.DIRECT
        ),
        answerable=True,
        note="",
    )


def test_percentile_single_value() -> None:
    """单个样本的任意百分位都等于自身。"""

    assert percentile(
        [10.0],
        95.0,
    ) == 10.0


def test_percentile_p50() -> None:
    """P50 应正确计算。"""

    value = percentile(
        [
            10.0,
            20.0,
            30.0,
            40.0,
        ],
        50.0,
    )

    assert value == (
        pytest.approx(
            25.0
        )
    )


def test_percentile_rejects_empty_values() -> None:
    """空样本不允许计算百分位。"""

    with pytest.raises(
        ValueError,
        match="values",
    ):
        percentile(
            [],
            95.0,
        )


def test_percentile_rejects_invalid_percentile() -> None:
    """百分位范围必须为 0~100。"""

    with pytest.raises(
        ValueError,
        match="0~100",
    ):
        percentile(
            [1.0],
            101.0,
        )


def test_benchmark_generates_expected_sample_count() -> None:
    """
    3 个 Query × 2 rounds
    应产生 6 个正式样本。

    Warmup 不进入 sample_count。
    """

    cases = [
        make_case("R001"),
        make_case("R002"),
        make_case("R003"),
    ]

    call_count = 0

    def fake_retrieve(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        nonlocal call_count

        call_count += 1

        return []

    result = benchmark_retrieval_method(
        method=(
            RetrievalMethod.BM25
        ),
        cases=cases,
        retrieve_fn=(
            fake_retrieve
        ),
        retrieval_top_k=5,
        rounds=2,
        warmup_count=2,
    )

    assert (
        result.sample_count
        == 6
    )

    assert (
        result.warmup_count
        == 2
    )

    # 2 warmup + 6 measured
    assert call_count == 8


def test_benchmark_calls_synchronize_when_provided() -> None:
    """
    GPU Benchmark 应支持计时前后同步。
    """

    cases = [
        make_case("R001")
    ]

    sync_count = 0

    def fake_sync() -> None:
        nonlocal sync_count

        sync_count += 1

    def fake_retrieve(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        return []

    benchmark_retrieval_method(
        method=(
            RetrievalMethod.DENSE
        ),
        cases=cases,
        retrieve_fn=(
            fake_retrieve
        ),
        retrieval_top_k=5,
        rounds=2,
        warmup_count=1,
        synchronize_fn=(
            fake_sync
        ),
    )

    # Warmup：
    # 前后同步 = 2 次
    #
    # 正式 2 次：
    # 每次前后同步 = 4 次
    #
    # 总计 = 6
    assert sync_count == 6


def test_benchmark_rejects_invalid_rounds() -> None:
    """rounds 必须大于 0。"""

    with pytest.raises(
        ValueError,
        match="rounds",
    ):
        benchmark_retrieval_method(
            method=(
                RetrievalMethod.DENSE
            ),
            cases=[
                make_case("R001")
            ],
            retrieve_fn=(
                lambda query,
                top_k,
                access_context: []
            ),
            retrieval_top_k=5,
            rounds=0,
        )