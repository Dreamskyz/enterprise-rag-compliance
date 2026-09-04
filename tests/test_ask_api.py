"""测试 Ask API。"""

from fastapi.testclient import (
    TestClient,
)

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.api.app import (
    create_app,
)
from enterprise_rag.generation.models import (
    Citation,
)
from enterprise_rag.service.models import (
    QueryResult,
)


class FakeQueryService:
    """模拟真实 QueryService。"""

    def __init__(
        self,
        result: QueryResult,
    ) -> None:
        self.result = result

        self.last_query: str | None = None

        self.last_access_context: (
            AccessContext | None
        ) = None

    def ask(
        self,
        query: str,
        access_context: AccessContext,
    ) -> QueryResult:
        """记录调用参数并返回预设结果。"""

        self.last_query = query

        self.last_access_context = (
            access_context
        )

        return self.result


def build_answerable_result(
) -> QueryResult:
    """构造法规类可回答 QueryResult。"""

    return QueryResult(
        query="测试问题",
        role=UserRole.GUEST,
        answerable=True,
        answer="这是有依据的回答。",
        reason="证据充分。",
        citations=(
            Citation(
                evidence_id="E1",
                chunk_id="chunk-1",
                title="测试法规",
                article_number="第一条",
                source_url=(
                    "https://example.com"
                ),
            ),
        ),
        retrieval_count=5,
        top_rerank_score=5.5,
        gate_reason="passed",
    )


def build_answerable_technical_result(
) -> QueryResult:
    """构造技术文档类可回答 QueryResult。

    这个 Case 专门用于锁定曾经真实发生的 Regression：

        FastAPI / Qdrant 等技术文档
        没有法规 article_number

    因此：

        article_number = None

    必须能够合法通过：

        Domain Model
        ↓
        AskResponse
        ↓
        CitationResponse
        ↓
        HTTP Response Validation

    而不能因为 API Schema 错误要求：

        article_number: str

    导致 HTTP 500。
    """

    return QueryResult(
        query="Qdrant Payload Filter 怎么使用？",
        role=UserRole.DEVELOPER,
        answerable=True,
        answer=(
            "Qdrant 可以在搜索时"
            "使用 Payload Filter 过滤结果。"
        ),
        reason=(
            "技术文档证据明确说明"
            "搜索过程中可以使用 payload filter。"
        ),
        citations=(
            Citation(
                evidence_id="E1",
                chunk_id="qdrant-filtering-1",
                title="Qdrant Filtering",
                article_number=None,
                source_url=(
                    "https://qdrant.tech/"
                    "documentation/search/filtering/"
                ),
            ),
        ),
        retrieval_count=5,
        top_rerank_score=4.4219,
        gate_reason="passed",
    )


def build_refusal_result(
) -> QueryResult:
    """构造拒答 QueryResult。"""

    return QueryResult(
        query="测试问题",
        role=UserRole.GUEST,
        answerable=False,
        answer=None,
        reason=(
            "证据未提供问题要求的"
            "具体事实。"
        ),
        citations=(),
        retrieval_count=5,
        top_rerank_score=4.5,
        gate_reason="passed",
    )


def build_client(
    result: QueryResult,
) -> tuple[
    TestClient,
    FakeQueryService,
]:
    """创建带 Fake QueryService 的 App。"""

    app = create_app(
        use_lifespan=False
    )

    fake_service = FakeQueryService(
        result=result
    )

    app.state.query_service = (
        fake_service
    )

    return (
        TestClient(app),
        fake_service,
    )


def test_ask_returns_answerable_response() -> None:
    """可回答法规问题应返回 Answer + Citation。"""

    client, _ = build_client(
        build_answerable_result()
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "测试问题",
            "role": "guest",
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["answerable"]
        is True
    )

    assert payload["answer"] == (
        "这是有依据的回答。"
    )

    assert (
        payload["gate_reason"]
        == "passed"
    )

    assert len(
        payload["citations"]
    ) == 1

    citation = (
        payload["citations"][0]
    )

    assert (
        citation["chunk_id"]
        == "chunk-1"
    )

    assert (
        citation["article_number"]
        == "第一条"
    )


def test_ask_supports_technical_citation_without_article_number(
) -> None:
    """技术文档 Citation 的 article_number=None 应正常返回 200。

    这是针对真实 HTTP 500 Bug 增加的 Regression Test。

    技术文档：

        OWASP
        FastAPI
        Qdrant

    并不存在法规式 article_number。

    因此 API 必须保留：

        str | None

    的真实领域语义。
    """

    client, _ = build_client(
        build_answerable_technical_result()
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": (
                "Qdrant Payload Filter "
                "怎么使用？"
            ),
            "role": "developer",
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["answerable"]
        is True
    )

    assert (
        payload["role"]
        == "developer"
    )

    assert len(
        payload["citations"]
    ) == 1

    citation = (
        payload["citations"][0]
    )

    assert (
        citation["chunk_id"]
        == "qdrant-filtering-1"
    )

    # ------------------------------------------------------
    # 核心 Regression Assertion：
    #
    # None 必须作为合法 JSON null 返回，
    # 而不是触发 FastAPI ResponseValidationError。
    # ------------------------------------------------------

    assert (
        citation["article_number"]
        is None
    )


def test_ask_returns_structured_refusal() -> None:
    """LLM Evidence Sufficiency 拒答应正确映射成 HTTP Response。"""

    client, _ = build_client(
        build_refusal_result()
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "测试问题",
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["answerable"]
        is False
    )

    assert (
        payload["answer"]
        is None
    )

    assert (
        payload["citations"]
        == []
    )


def test_ask_defaults_to_guest() -> None:
    """未提供 role 时默认 guest。"""

    client, service = build_client(
        build_answerable_result()
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "测试问题"
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        service.last_access_context
        is not None
    )

    assert (
        service
        .last_access_context
        .role
        == UserRole.GUEST
    )


def test_ask_propagates_admin_role() -> None:
    """admin role 应正确传给 QueryService。"""

    result = build_answerable_result()

    client, service = build_client(
        result
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "测试问题",
            "role": "admin",
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        service.last_access_context
        is not None
    )

    assert (
        service
        .last_access_context
        .role
        == UserRole.ADMIN
    )


def test_ask_rejects_invalid_role() -> None:
    """非法 role 应返回 422。"""

    client, _ = build_client(
        build_answerable_result()
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "测试问题",
            "role": "superuser",
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_ask_returns_503_when_service_missing() -> None:
    """QueryService 未初始化时应返回 503。"""

    app = create_app(
        use_lifespan=False
    )

    client = TestClient(
        app
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "测试问题"
        },
    )

    assert (
        response.status_code
        == 503
    )