"""测试 Answer Evaluation Runner。"""

from enterprise_rag.acl.models import (
    AccessContext,
)
from enterprise_rag.evaluation.answer_runner import (
    run_answer_evaluation,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalCategory,
)
from enterprise_rag.service.models import (
    QueryResult,
)


class FakeQueryService:
    """
    不调用真实 Retrieval / LLM 的 Fake QueryService。

    注意：

    ask() 参数名故意和真实 QueryService 保持一致：

        access_context

    防止再次出现 Test Double 与生产接口漂移。
    """

    def ask(
        self,
        query: str,
        access_context: AccessContext,
    ) -> QueryResult:
        if query == "可回答":
            return QueryResult(
                query=query,
                role=(
                    access_context.role
                ),
                answerable=True,
                answer="测试回答",
                reason="证据充分",
                citations=(),
                retrieval_count=1,
                top_rerank_score=5.0,
                gate_reason="passed",
            )

        return QueryResult(
            query=query,
            role=(
                access_context.role
            ),
            answerable=False,
            answer=None,
            reason="证据不足",
            citations=(),
            retrieval_count=1,
            top_rerank_score=-5.0,
            gate_reason=(
                "below_threshold"
            ),
        )


def test_answer_runner() -> None:
    """
    Runner 应：

        1. 执行全部 Case；
        2. 正确统计 Answer / Refusal；
        3. 将 Retrieval Gold 和 Citation Gold
           一起带入 Snapshot Result。
    """

    cases = [
        RetrievalEvalCase(
            query_id="R001",
            query="可回答",

            # Retrieval Gold。
            gold_chunk_ids=(
                "retrieval-a",
            ),

            category=(
                RetrievalEvalCategory.DIRECT
            ),

            answerable=True,

            note="",

            # Citation Gold。
            citation_gold_chunk_ids=(
                "citation-a",
            ),

            strict_citation_eval=True,
        ),

        RetrievalEvalCase(
            query_id="R002",
            query="不可回答",

            gold_chunk_ids=(),

            category=(
                RetrievalEvalCategory.OUT_OF_DOMAIN
            ),

            answerable=False,

            note="",

            citation_gold_chunk_ids=(),

            strict_citation_eval=False,
        ),
    ]

    result = run_answer_evaluation(
        cases=cases,

        # Fake 满足真实 ask Contract。
        query_service=(
            FakeQueryService()
        ),  # type: ignore[arg-type]
    )

    assert (
        len(
            result.case_results
        )
        == 2
    )

    assert (
        result.metrics.true_positive
        == 1
    )

    assert (
        result.metrics.true_negative
        == 1
    )

    assert (
        result.metrics.false_positive
        == 0
    )

    assert (
        result.metrics.false_negative
        == 0
    )

    assert (
        result
        .metrics
        .overall_decision_accuracy
        == 1.0
    )

    # ------------------------------------------------------
    # 验证 Annotation 被 Runner 正确传播。
    # ------------------------------------------------------

    first = result.case_results[0]

    assert (
        first.gold_chunk_ids
        == (
            "retrieval-a",
        )
    )

    assert (
        first.citation_gold_chunk_ids
        == (
            "citation-a",
        )
    )

    assert (
        first.strict_citation_eval
        is True
    )

    second = result.case_results[1]

    assert (
        second.citation_gold_chunk_ids
        == ()
    )

    assert (
        second.strict_citation_eval
        is False
    )


def test_answer_runner_rejects_empty_cases() -> None:
    """空 Dataset 不允许执行 Evaluation。"""

    try:
        run_answer_evaluation(
            cases=[],
            query_service=(
                FakeQueryService()
            ),  # type: ignore[arg-type]
        )

    except ValueError as exc:
        assert (
            "cases"
            in str(exc)
        )

    else:
        raise AssertionError(
            "空 cases 应抛出 ValueError"
        )