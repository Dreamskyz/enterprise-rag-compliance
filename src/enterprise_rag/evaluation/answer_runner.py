"""完整 RAG Answer / Refusal Evaluation Runner。"""

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from enterprise_rag.acl.models import (
    AccessContext,
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
    使用真实 QueryService
    执行完整 Full-RAG Evaluation。

    链路：

        Query
          ↓
        Case Role / Override AccessContext
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

    ------------------------------------------------------
    Role-aware Evaluation
    ------------------------------------------------------

    默认情况下：

        access_context is None

    每一条 Case 都必须使用 Dataset 中自己的：

        case.role

    例如：

        R042
        role = developer

        R045
        role = guest

    即使两条 Query 完全相同，
    也必须进入不同的 ACL Candidate Space。

    ------------------------------------------------------
    Explicit Override
    ------------------------------------------------------

    如果调用者显式传入：

        access_context

    则认为调用者希望对整批 Case
    使用同一个 AccessContext。

    这种能力主要用于：

        Debug
        特殊对照实验
        向后兼容

    正式 Role-aware Evaluation
    不应传入全局 access_context。
    """

    if not cases:
        raise ValueError(
            "cases 不能为空"
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
        # 1. 打印进度。
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
        # 2. 构造当前 Case 的 AccessContext。
        # --------------------------------------------------
        #
        # 这是 Role-aware Evaluation
        # 最关键的一步。
        #
        # 默认：
        #
        #     case.role
        #
        # 例如：
        #
        #     guest
        #     developer
        #     admin
        #
        # 不能在 Evaluation Runner
        # 中把所有 Case 偷偷固定成 guest。
        #
        # 如果调用者显式传入 access_context，
        # 才使用全局 Override。
        # --------------------------------------------------

        if access_context is None:
            case_access_context = (
                AccessContext(
                    role=case.role
                )
            )
        else:
            case_access_context = (
                access_context
            )

        # --------------------------------------------------
        # 3. 调用真实 QueryService。
        #
        # 参数名必须与生产接口保持一致：
        #
        #     access_context=
        #
        # 不能写成：
        #
        #     context=
        # --------------------------------------------------

        result = query_service.ask(
            query=case.query,
            access_context=(
                case_access_context
            ),
        )

        case_latency_ms = (
            perf_counter()
            - case_started_at
        ) * 1000.0

        # --------------------------------------------------
        # 4. 提取程序验证后的 Citation Chunk ID。
        #
        # QueryResult 中 Citation
        # 已经是生产链路最终的
        # 结构化 Citation。
        #
        # Evaluation 层这里只读取：
        #
        #     citation.chunk_id
        # --------------------------------------------------

        cited_chunk_ids = tuple(
            citation.chunk_id
            for citation
            in result.citations
        )

        # --------------------------------------------------
        # 5. 保存当前 Case 的完整 Raw Result。
        #
        # 同时保留：
        #
        # 1. Retrieval Gold
        # 2. Citation Gold
        # 3. Strict Citation Annotation
        # 4. Answer / Refusal Decision
        # 5. Gate Reason
        # 6. Top Rerank Score
        # 7. Latency
        #
        # 这样 Snapshot
        # 后续可以脱离在线 Runtime
        # 进行离线 Failure Analysis。
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

                # ------------------------------------------
                # Retrieval Gold。
                # ------------------------------------------

                gold_chunk_ids=(
                    case.gold_chunk_ids
                ),

                # ------------------------------------------
                # Citation Gold。
                # ------------------------------------------

                citation_gold_chunk_ids=(
                    case.citation_gold_chunk_ids
                ),

                # ------------------------------------------
                # 是否进入 Strict Citation Metrics。
                # ------------------------------------------

                strict_citation_eval=(
                    case.strict_citation_eval
                ),

                # ------------------------------------------
                # 真实 Full-RAG 输出。
                # ------------------------------------------

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

    # ------------------------------------------------------
    # 6. 整次 Evaluation 总耗时。
    # ------------------------------------------------------

    total_latency_ms = (
        perf_counter()
        - run_started_at
    ) * 1000.0

    # ------------------------------------------------------
    # 7. Aggregate Metrics。
    # ------------------------------------------------------

    metrics = (
        aggregate_answer_metrics(
            case_results
        )
    )

    # ------------------------------------------------------
    # 8. 返回完整 Evaluation Result。
    # ------------------------------------------------------

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