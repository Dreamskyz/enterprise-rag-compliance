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
    """
    模拟真实 QueryService。
    """

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
        self.last_query = query

        self.last_access_context = (
            access_context
        )

        return self.result


def build_answerable_result(
) -> QueryResult:
    """构造可回答 QueryResult。"""

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
    """
    创建带 Fake QueryService 的 App。
    """

    app = create_app(
        use_lifespan=False
    )

    fake_service = (
        FakeQueryService(
            result=result
        )
    )

    app.state.query_service = (
        fake_service
    )

    return (
        TestClient(app),
        fake_service,
    )


def test_ask_returns_answerable_response() -> None:
    """可回答问题应返回 Answer + Citation。"""

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


def test_ask_returns_structured_refusal() -> None:
    """
    LLM Evidence Sufficiency 拒答
    应正确映射成 HTTP Response。
    """

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
    """
    QueryService 未初始化时，
    应返回 503。
    """

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