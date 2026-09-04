"""测试 Answer Evaluation Runner。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
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

    防止再次出现 Test Double
    与生产接口漂移。
    """

    def ask(
        self,
        query: str,
        access_context: AccessContext,
    ) -> QueryResult:
        """
        根据 query 返回固定结果。

        当前 Fake 只负责：

        1. 模拟可回答；
        2. 模拟不可回答；
        3. 保留传入的 role。

        不执行任何真实 Retrieval / LLM。
        """

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


class RecordingFakeQueryService:
    """
    专门用于检查 Runner
    是否正确传播 AccessContext。

    每次 ask() 被调用时，
    都把收到的 role 保存到：

        received_roles

    这样测试可以直接验证：

        guest case
            ↓
        guest

        developer case
            ↓
        developer

    而不是只通过最终 Answer Metrics
    间接猜测权限是否正确。
    """

    def __init__(self) -> None:
        self.received_roles: list[
            UserRole
        ] = []

    def ask(
        self,
        query: str,
        access_context: AccessContext,
    ) -> QueryResult:
        """
        记录当前真实收到的 role，
        然后返回一个固定的可回答结果。
        """

        self.received_roles.append(
            access_context.role
        )

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

            # 当前基础 Case 显式声明 guest，
            # 避免测试依赖模型默认值。
            role=UserRole.GUEST,
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

            role=UserRole.GUEST,
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


def test_answer_runner_uses_case_role() -> None:
    """
    默认情况下，
    Runner 必须使用每条 Dataset Case
    自己的 role。

    这个测试专门防止曾经出现过的 Bug：

        Dataset:
            guest
            developer

        Runner:
            全部错误地变成 guest

    这是 Full-RAG ACL Evaluation
    最重要的 Regression Test 之一。
    """

    service = (
        RecordingFakeQueryService()
    )

    cases = [
        RetrievalEvalCase(
            query_id="R101",
            query="guest query",

            gold_chunk_ids=(
                "public-chunk",
            ),

            category=(
                RetrievalEvalCategory.DIRECT
            ),

            answerable=True,

            note="guest role case",

            citation_gold_chunk_ids=(
                "public-chunk",
            ),

            strict_citation_eval=True,

            role=UserRole.GUEST,
        ),

        RetrievalEvalCase(
            query_id="R102",
            query="developer query",

            gold_chunk_ids=(
                "developer-chunk",
            ),

            category=(
                RetrievalEvalCategory.DIRECT
            ),

            answerable=True,

            note="developer role case",

            citation_gold_chunk_ids=(
                "developer-chunk",
            ),

            strict_citation_eval=True,

            role=UserRole.DEVELOPER,
        ),
    ]

    run_answer_evaluation(
        cases=cases,
        query_service=(
            service
        ),  # type: ignore[arg-type]
    )

    # ------------------------------------------------------
    # 核心断言：
    #
    # Runner 必须按 Case 顺序传播：
    #
    #     guest
    #     developer
    #
    # 如果未来又错误地在 Runner 中写成：
    #
    #     AccessContext(
    #         role=UserRole.GUEST
    #     )
    #
    # 那这里会直接失败。
    # ------------------------------------------------------

    assert (
        service.received_roles
        == [
            UserRole.GUEST,
            UserRole.DEVELOPER,
        ]
    )


def test_answer_runner_allows_explicit_access_context_override() -> None:
    """
    如果调用者显式传入 access_context，
    则允许对整批 Evaluation Case
    使用同一个 AccessContext。

    这个能力主要用于：

        Debug
        特殊对照实验

    正式 V3 Role-aware Evaluation
    不使用这个 Override。

    本测试用于锁定 Runner 当前定义好的语义：

        默认
            → case.role

        显式 access_context
            → global override
    """

    service = (
        RecordingFakeQueryService()
    )

    cases = [
        RetrievalEvalCase(
            query_id="R201",
            query="guest query",

            gold_chunk_ids=(
                "public-chunk",
            ),

            category=(
                RetrievalEvalCategory.DIRECT
            ),

            answerable=True,

            note="guest role case",

            citation_gold_chunk_ids=(
                "public-chunk",
            ),

            strict_citation_eval=True,

            role=UserRole.GUEST,
        ),

        RetrievalEvalCase(
            query_id="R202",
            query="developer query",

            gold_chunk_ids=(
                "developer-chunk",
            ),

            category=(
                RetrievalEvalCategory.DIRECT
            ),

            answerable=True,

            note="developer role case",

            citation_gold_chunk_ids=(
                "developer-chunk",
            ),

            strict_citation_eval=True,

            role=UserRole.DEVELOPER,
        ),
    ]

    # ------------------------------------------------------
    # 显式指定 ADMIN，
    # 用于测试全局 Override。
    # ------------------------------------------------------

    override_context = AccessContext(
        role=UserRole.ADMIN
    )

    run_answer_evaluation(
        cases=cases,
        query_service=(
            service
        ),  # type: ignore[arg-type]
        access_context=(
            override_context
        ),
    )

    # ------------------------------------------------------
    # 即使 Dataset 中分别是：
    #
    #     guest
    #     developer
    #
    # 显式 Override 后都应该收到 admin。
    # ------------------------------------------------------

    assert (
        service.received_roles
        == [
            UserRole.ADMIN,
            UserRole.ADMIN,
        ]
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