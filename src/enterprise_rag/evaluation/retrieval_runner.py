"""统一 Retrieval Evaluation Runner。"""

from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import TypeAlias

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
)
from enterprise_rag.evaluation.retrieval_metrics import (
    RetrievalAggregateMetrics,
    RetrievalMetricResult,
    aggregate_retrieval_metrics,
    evaluate_retrieval_case,
)
from enterprise_rag.retrieval.models import (
    BM25SearchResult,
    DenseSearchResult,
    HybridSearchResult,
    RerankedSearchResult,
)


class RetrievalMethod(StrEnum):
    """
    Retrieval Ablation 中比较的方法。
    """

    DENSE = "dense"

    BM25 = "bm25"

    HYBRID_RRF = "hybrid_rrf"

    HYBRID_RERANK = "hybrid_rerank"


SearchResult: TypeAlias = (
    DenseSearchResult
    | BM25SearchResult
    | HybridSearchResult
    | RerankedSearchResult
)


RetrieverCallable = Callable[
    [
        str,
        int,
        AccessContext,
    ],
    Sequence[SearchResult],
]


@dataclass(frozen=True)
class RetrievalQueryEvalResult:
    """
    单条 Query、单种 Method 的评测结果。
    """

    query_id: str

    query: str

    method: RetrievalMethod

    retrieved_chunk_ids: tuple[
        str,
        ...,
    ]

    gold_chunk_ids: tuple[
        str,
        ...,
    ]

    metrics_by_k: dict[
        int,
        RetrievalMetricResult,
    ]

    latency_ms: float


@dataclass(frozen=True)
class RetrievalMethodEvalResult:
    """
    单种 Retrieval Method 的 Dataset 级结果。
    """

    method: RetrievalMethod

    query_count: int

    aggregate_by_k: dict[
        int,
        RetrievalAggregateMetrics,
    ]

    mean_latency_ms: float

    query_results: tuple[
        RetrievalQueryEvalResult,
        ...,
    ]


def extract_chunk_ids(
    results: Sequence[
        SearchResult
    ],
) -> list[str]:
    """
    从不同 Retrieval Result 类型中
    统一提取排名后的 chunk_id。
    """

    return [
        result.candidate.chunk_id
        for result in results
    ]


def evaluate_retrieval_method(
    *,
    method: RetrievalMethod,
    cases: Sequence[
        RetrievalEvalCase
    ],
    retrieve_fn: RetrieverCallable,
    evaluation_ks: Sequence[int],
    retrieval_top_k: int,
    access_context: AccessContext | None = None,
) -> RetrievalMethodEvalResult:
    """
    使用统一 Dataset / Metrics
    评估一种 Retrieval Method。

    当前只统计 answerable=true 的 Case。
    """

    if retrieval_top_k <= 0:
        raise ValueError(
            "retrieval_top_k 必须大于 0"
        )

    if not evaluation_ks:
        raise ValueError(
            "evaluation_ks 不能为空"
        )

    normalized_ks = sorted(
        set(
            int(k)
            for k in evaluation_ks
        )
    )

    if any(
        k <= 0
        for k in normalized_ks
    ):
        raise ValueError(
            "所有 evaluation k 必须大于 0"
        )

    if max(
        normalized_ks
    ) > retrieval_top_k:
        raise ValueError(
            "最大的 evaluation k "
            "不能超过 retrieval_top_k"
        )

    if access_context is None:
        access_context = AccessContext(
            role=UserRole.GUEST
        )

    answerable_cases = [
        case
        for case in cases
        if case.answerable
    ]

    if not answerable_cases:
        raise ValueError(
            "没有可用于 Retrieval Evaluation "
            "的 answerable case"
        )

    query_results: list[
        RetrievalQueryEvalResult
    ] = []

    for case in answerable_cases:
        started_at = (
            perf_counter()
        )

        results = retrieve_fn(
            case.query,
            retrieval_top_k,
            access_context,
        )

        latency_ms = (
            perf_counter()
            - started_at
        ) * 1000.0

        retrieved_chunk_ids = (
            extract_chunk_ids(
                results
            )
        )

        metrics_by_k: dict[
            int,
            RetrievalMetricResult,
        ] = {}

        for k in normalized_ks:
            metrics_by_k[k] = (
                evaluate_retrieval_case(
                    retrieved_chunk_ids=(
                        retrieved_chunk_ids
                    ),
                    gold_chunk_ids=(
                        case.gold_chunk_ids
                    ),
                    k=k,
                )
            )

        query_results.append(
            RetrievalQueryEvalResult(
                query_id=(
                    case.query_id
                ),
                query=(
                    case.query
                ),
                method=method,
                retrieved_chunk_ids=tuple(
                    retrieved_chunk_ids
                ),
                gold_chunk_ids=(
                    case.gold_chunk_ids
                ),
                metrics_by_k=(
                    metrics_by_k
                ),
                latency_ms=(
                    latency_ms
                ),
            )
        )

    aggregate_by_k: dict[
        int,
        RetrievalAggregateMetrics,
    ] = {}

    for k in normalized_ks:
        metric_results = [
            result.metrics_by_k[k]
            for result in query_results
        ]

        aggregate_by_k[k] = (
            aggregate_retrieval_metrics(
                metric_results
            )
        )

    mean_latency_ms = (
        sum(
            result.latency_ms
            for result in query_results
        )
        / len(query_results)
    )

    return RetrievalMethodEvalResult(
        method=method,
        query_count=len(
            query_results
        ),
        aggregate_by_k=(
            aggregate_by_k
        ),
        mean_latency_ms=(
            mean_latency_ms
        ),
        query_results=tuple(
            query_results
        ),
    )