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

    role:

        表示这次 Retrieval
        实际使用的用户角色。

        记录这个字段非常重要，
        因为同一个 Query：

            guest

        和：

            developer

        实际搜索的是不同的授权 Candidate Space。

        如果结果对象不记录 role，
        后续 Failure Analysis 很容易丢失
        这层关键实验上下文。
    """

    query_id: str

    query: str

    role: UserRole

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

    当前只统计：

        answerable=true

    的 Case。

    ------------------------------------------------------
    Role-aware Evaluation
    ------------------------------------------------------

    默认情况下：

        每条 RetrievalEvalCase
        使用自己的：

            case.role

    构造：

        AccessContext(role=case.role)

    因此 V2 Dataset 可以同时包含：

        guest case
        developer case
        admin case

    同一个评测运行中，
    每条 Query 都可以在自己的
    ACL Candidate Space 内执行。

    ------------------------------------------------------
    access_context 参数
    ------------------------------------------------------

    这个参数保留下来作为：

        explicit global override

    即：

        如果 access_context is not None

    则所有 Case 都使用该 Context，
    暂时忽略 Dataset 中的 case.role。

    这样可以兼容项目里已有的：

        测试
        Debug Script
        临时实验

    也避免一次 schema evolution
    破坏原有 Runner API。

    正式 Dataset Evaluation
    默认不传 access_context，
    让 Dataset 自身成为 role 的事实来源。
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

        # --------------------------------------------------
        # 1. 为当前 Case 决定实际 AccessContext。
        # --------------------------------------------------
        #
        # 默认：
        #
        #     Dataset role
        #
        # 显式传入 access_context 时：
        #
        #     global override
        #
        # 这使正式评测能够 role-aware，
        # 同时保持旧 API 的兼容性。
        if access_context is None:
            case_access_context = AccessContext(
                role=case.role
            )
        else:
            case_access_context = access_context

        # --------------------------------------------------
        # 2. 执行 Retrieval。
        # --------------------------------------------------

        started_at = (
            perf_counter()
        )

        results = retrieve_fn(
            case.query,
            retrieval_top_k,
            case_access_context,
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

        # --------------------------------------------------
        # 3. 计算不同 K 下 Retrieval Metrics。
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 4. 保存 Query-level Evaluation Result。
        # --------------------------------------------------

        query_results.append(
            RetrievalQueryEvalResult(
                query_id=(
                    case.query_id
                ),
                query=(
                    case.query
                ),
                role=(
                    case_access_context.role
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

    # ------------------------------------------------------
    # 5. Aggregate Metrics。
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # 6. 当前保留 Query-level mean latency。
    #
    # 注意：
    #
    # 这个字段只属于 Retrieval Runner
    # 的粗粒度诊断信息。
    #
    # 项目正式 latency 结论来自：
    #
    # Query-level Interleaved Benchmark
    #
    # 通过随机化不同 Method 的执行顺序，
    # 控制：
    #
    # GPU Warm-State
    # Method Order Bias
    #
    # 因此这里的 mean_latency_ms
    # 不应该替代正式 latency benchmark。
    # ------------------------------------------------------

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