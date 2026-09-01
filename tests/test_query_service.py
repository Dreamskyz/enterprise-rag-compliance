"""测试 QueryService 的业务编排逻辑。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.evidence.gate import (
    EvidenceGate,
)
from enterprise_rag.generation.models import (
    Citation,
    GroundedAnswer,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
    RetrievalCandidate,
)
from enterprise_rag.service.query_service import (
    QueryService,
)


def make_result(
    score: float,
) -> RerankedSearchResult:
    """构造测试 Retrieval Result。"""

    candidate = RetrievalCandidate(
        chunk_id="test_chunk",
        document_id="test_doc",
        title="测试法规",
        document_type="regulation",
        language="zh-CN",
        version="1",
        chapter_number="第一章",
        chapter_title="测试章节",
        article_number="第一条",
        content="测试正文。",
        retrieval_text="测试检索文本。",
        source_url=(
            "https://example.com"
        ),
        access_level="public",
        chunk_index=0,
        content_hash="test-hash",
    )

    return RerankedSearchResult(
        candidate=candidate,
        rerank_score=score,
        original_rank=1,
        rrf_score=0.03,
        dense_rank=1,
        bm25_rank=1,
    )


class FakeRetriever:
    """模拟最终 Reranked Retriever。"""

    def __init__(
        self,
        results: list[
            RerankedSearchResult
        ],
    ) -> None:
        self.results = results

        self.last_access_context: (
            AccessContext | None
        ) = None

    def search(
        self,
        query: str,
        top_k: int,
        access_context: AccessContext,
    ) -> list[
        RerankedSearchResult
    ]:
        self.last_access_context = (
            access_context
        )

        return self.results[
            :top_k
        ]


class FakeAnswerer:
    """模拟 EvidenceGroundedAnswerer。"""

    def __init__(
        self,
        answer: GroundedAnswer,
    ) -> None:
        self.answer_value = answer

        self.call_count = 0

    def answer(
        self,
        query: str,
        results,
    ) -> GroundedAnswer:
        self.call_count += 1

        return self.answer_value


def build_answerable_grounded_answer(
) -> GroundedAnswer:
    """构造正常回答。"""

    return GroundedAnswer(
        answerable=True,
        answer="这是有依据的回答。",
        reason="证据明确支持回答。",
        citations=(
            Citation(
                evidence_id="E1",
                chunk_id="test_chunk",
                title="测试法规",
                article_number="第一条",
                source_url=(
                    "https://example.com"
                ),
            ),
        ),
    )


def build_refusal_grounded_answer(
) -> GroundedAnswer:
    """构造 LLM Evidence Sufficiency 拒答。"""

    return GroundedAnswer(
        answerable=False,
        answer=None,
        reason=(
            "证据相关，但缺少问题"
            "要求的具体事实。"
        ),
        citations=(),
    )


def test_query_service_defaults_to_guest() -> None:
    """
    未提供 AccessContext 时，
    必须使用 guest。
    """

    retriever = FakeRetriever(
        [
            make_result(5.0),
        ]
    )

    answerer = FakeAnswerer(
        build_answerable_grounded_answer()
    )

    service = QueryService(
        retriever=retriever,
        evidence_gate=EvidenceGate(
            min_top_score=0.0
        ),
        answerer=answerer,
    )

    service.ask(
        query="测试问题"
    )

    assert (
        retriever.last_access_context
        is not None
    )

    assert (
        retriever
        .last_access_context
        .role
        == UserRole.GUEST
    )


def test_query_service_propagates_access_context() -> None:
    """developer 权限必须继续传给 Retriever。"""

    retriever = FakeRetriever(
        [
            make_result(5.0),
        ]
    )

    answerer = FakeAnswerer(
        build_answerable_grounded_answer()
    )

    service = QueryService(
        retriever=retriever,
        evidence_gate=EvidenceGate(
            min_top_score=0.0
        ),
        answerer=answerer,
    )

    context = AccessContext(
        role=UserRole.DEVELOPER
    )

    service.ask(
        query="测试问题",
        access_context=context,
    )

    assert (
        retriever.last_access_context
        == context
    )


def test_query_service_rejects_no_results_without_llm() -> None:
    """
    无 Retrieval Result 时：
        直接拒答
        不调用 Answerer
    """

    retriever = FakeRetriever(
        []
    )

    answerer = FakeAnswerer(
        build_answerable_grounded_answer()
    )

    service = QueryService(
        retriever=retriever,
        evidence_gate=EvidenceGate(
            min_top_score=0.0
        ),
        answerer=answerer,
    )

    result = service.ask(
        query="完全无证据问题"
    )

    assert result.answerable is False

    assert result.answer is None

    assert (
        result.gate_reason
        == "no_results"
    )

    assert answerer.call_count == 0


def test_query_service_rejects_below_gate_without_llm() -> None:
    """
    低于 Coarse Relevance Gate：
        程序拒答
        不调用 LLM Answerer。
    """

    retriever = FakeRetriever(
        [
            make_result(-8.0),
        ]
    )

    answerer = FakeAnswerer(
        build_answerable_grounded_answer()
    )

    service = QueryService(
        retriever=retriever,
        evidence_gate=EvidenceGate(
            min_top_score=-3.0
        ),
        answerer=answerer,
    )

    result = service.ask(
        query="明显无关问题"
    )

    assert result.answerable is False

    assert (
        result.gate_reason
        == "below_threshold"
    )

    assert answerer.call_count == 0


def test_query_service_returns_grounded_answer() -> None:
    """通过 Gate 后应返回 Answerer 的结果。"""

    retriever = FakeRetriever(
        [
            make_result(5.0),
        ]
    )

    answerer = FakeAnswerer(
        build_answerable_grounded_answer()
    )

    service = QueryService(
        retriever=retriever,
        evidence_gate=EvidenceGate(
            min_top_score=-3.0
        ),
        answerer=answerer,
    )

    result = service.ask(
        query="可回答问题"
    )

    assert result.answerable is True

    assert result.answer == (
        "这是有依据的回答。"
    )

    assert len(
        result.citations
    ) == 1

    assert result.gate_reason == (
        "passed"
    )

    assert answerer.call_count == 1


def test_query_service_allows_llm_structured_refusal() -> None:
    """
    即使通过 Relevance Gate，
    LLM 仍然可以因为 Evidence
    不足而拒答。
    """

    retriever = FakeRetriever(
        [
            make_result(4.5),
        ]
    )

    answerer = FakeAnswerer(
        build_refusal_grounded_answer()
    )

    service = QueryService(
        retriever=retriever,
        evidence_gate=EvidenceGate(
            min_top_score=-3.0
        ),
        answerer=answerer,
    )

    result = service.ask(
        query=(
            "发现违法内容后"
            "必须几小时处理？"
        )
    )

    assert result.gate_reason == (
        "passed"
    )

    assert result.answerable is False

    assert result.answer is None

    assert result.citations == ()

    assert answerer.call_count == 1