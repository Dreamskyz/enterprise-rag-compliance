"""完整 RAG Answer / Refusal Evaluation Runner。"""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.evaluation.answer_metrics import (
    AnswerAggregateMetrics,
    AnswerEvalCaseResult,
    aggregate_answer_metrics,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
)
from enterprise_rag.service.query_service import (
    QueryService,
)


@dataclass(frozen=True)
class AnswerEvalRunResult:
    """
    一次完整 Full-RAG Evaluation Run。

    metrics:
        Dataset 级指标。

    case_results:
        每条 Query 的原始评测结果。

    total_latency_ms:
        整次评测总耗时。

    mean_latency_ms:
        平均每条 Query 的端到端耗时。

    注意：
        这里的 latency 不是正式 API Benchmark，
        只是 Full-RAG Evaluation 的运行记录。
    """

    metrics: AnswerAggregateMetrics

    case_results: tuple[
        AnswerEvalCaseResult,
        ...,
    ]

    total_latency_ms: float

    mean_latency_ms: float


def run_answer_evaluation(
    *,
    cases: Sequence[
        RetrievalEvalCase
    ],
    query_service: QueryService,
    access_context: AccessContext | None = None,
) -> AnswerEvalRunResult:
    """
    使用真实 QueryService 执行完整 Full-RAG Evaluation。

    链路：

        Query
        ↓
        ACL-aware Retrieval
        ↓
        Rerank
        ↓
        Coarse Relevance Gate
        ↓
        Evidence-Constrained Generation
        或 Programmatic Refusal
        ↓
        Citation
    """

    if not cases:
        raise ValueError(
            "cases 不能为空"
        )

    if access_context is None:
        access_context = AccessContext(
            role=UserRole.GUEST
        )

    case_results: list[
        AnswerEvalCaseResult
    ] = []

    run_started_at = (
        perf_counter()
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        # --------------------------------------------------
        # 打印进度。
        #
        # Full-RAG Evaluation 会真实调用：
        #
        # Retrieval
        # GPU Reranker
        # SiliconFlow LLM
        #
        # 所以需要知道当前运行到哪一条。
        # --------------------------------------------------

        print(
            f"[{index}/{len(cases)}] "
            f"{case.query_id} | "
            f"{case.query}"
        )

        case_started_at = (
            perf_counter()
        )

        # --------------------------------------------------
        # 这里必须调用真实 QueryService。
        #
        # 参数名必须与生产接口保持一致：
        #
        #     access_context=
        #
        # 不要写成 context=。
        # --------------------------------------------------

        result = query_service.ask(
            query=case.query,
            access_context=(
                access_context
            ),
        )

        case_latency_ms = (
            perf_counter()
            - case_started_at
        ) * 1000.0

        # --------------------------------------------------
        # QueryResult 中 Citation 已经是程序验证后的
        # 结构化 Citation。
        #
        # Evaluation 这里只提取 chunk_id。
        # --------------------------------------------------

        cited_chunk_ids = tuple(
            citation.chunk_id
            for citation
            in result.citations
        )

        # --------------------------------------------------
        # 这里同时保存：
        #
        # 1. Retrieval Gold
        # 2. Citation Gold
        # 3. strict citation annotation
        # 4. 真实模型输出
        #
        # 这样 Snapshot 后续可以独立离线分析。
        # --------------------------------------------------

        case_results.append(
            AnswerEvalCaseResult(
                query_id=(
                    case.query_id
                ),
                query=(
                    case.query
                ),
                category=(
                    case.category
                ),
                expected_answerable=(
                    case.answerable
                ),

                # Retrieval Gold。
                gold_chunk_ids=(
                    case.gold_chunk_ids
                ),

                # Citation Gold。
                citation_gold_chunk_ids=(
                    case.citation_gold_chunk_ids
                ),

                # 是否进入 Strict Citation Metrics。
                strict_citation_eval=(
                    case.strict_citation_eval
                ),

                actual_answerable=(
                    result.answerable
                ),

                answer=(
                    result.answer
                ),

                cited_chunk_ids=(
                    cited_chunk_ids
                ),

                gate_reason=(
                    result.gate_reason
                ),

                reason=(
                    result.reason
                ),

                retrieval_count=(
                    result.retrieval_count
                ),

                top_rerank_score=(
                    result.top_rerank_score
                ),

                latency_ms=(
                    case_latency_ms
                ),
            )
        )

    total_latency_ms = (
        perf_counter()
        - run_started_at
    ) * 1000.0

    metrics = (
        aggregate_answer_metrics(
            case_results
        )
    )

    return AnswerEvalRunResult(
        metrics=metrics,
        case_results=tuple(
            case_results
        ),
        total_latency_ms=(
            total_latency_ms
        ),
        mean_latency_ms=(
            total_latency_ms
            / len(case_results)
        ),
    )