"""测试 Retrieval API。"""

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
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
    RetrievalCandidate,
)


def make_result(
    access_level: str = "public",
) -> RerankedSearchResult:
    """构造一条法规 Retrieval Result。"""

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
        content="这是测试法规正文。",
        retrieval_text=(
            "测试法规 第一条 "
            "这是测试法规正文。"
        ),
        source_url=(
            "https://example.com"
        ),
        access_level=access_level,
        chunk_index=0,
        content_hash="test-hash",
    )

    return RerankedSearchResult(
        candidate=candidate,
        rerank_score=5.5,
        original_rank=1,
        rrf_score=0.032,
        dense_rank=1,
        bm25_rank=2,
    )


def make_technical_result(
) -> RerankedSearchResult:
    """构造技术文档 Retrieval Result。

    技术文档没有法规 Article Number。

    因此：

        article_number = None

    这个 Fixture 专门用于 Regression Test，
    防止 RetrieveResponse Schema 再次错误收窄为：

        article_number: str

    从而导致 FastAPI ResponseValidationError。
    """

    candidate = RetrievalCandidate(
        chunk_id="qdrant_filtering_1",
        document_id="qdrant_filtering",
        title="Qdrant Filtering",
        document_type=(
            "technical_documentation"
        ),
        language="en",
        version="1",
        chapter_number=None,
        chapter_title=None,
        article_number=None,
        content=(
            "Qdrant supports filtering search "
            "results using payload conditions."
        ),
        retrieval_text=(
            "Qdrant Filtering "
            "payload conditions search results"
        ),
        source_url=(
            "https://qdrant.tech/"
            "documentation/search/filtering/"
        ),
        access_level="developer",
        chunk_index=0,
        content_hash=(
            "technical-test-hash"
        ),
    )

    return RerankedSearchResult(
        candidate=candidate,
        rerank_score=4.4219,
        original_rank=1,
        rrf_score=0.0164,
        dense_rank=1,
        bm25_rank=None,
    )


class FakeRetriever:
    """模拟真实 RerankedRetriever。"""

    def __init__(
        self,
        result: RerankedSearchResult | None = None,
    ) -> None:
        self.result = (
            result
            if result is not None
            else make_result()
        )

        self.last_query: str | None = None

        self.last_top_k: int | None = None

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
        """记录调用参数并返回预设 Result。"""

        self.last_query = query

        self.last_top_k = top_k

        self.last_access_context = (
            access_context
        )

        return [
            self.result
        ]


def build_client(
    result: RerankedSearchResult | None = None,
) -> tuple[
    TestClient,
    FakeRetriever,
]:
    """创建带 Fake Retriever 的测试 App。"""

    app = create_app(
        use_lifespan=False
    )

    fake_retriever = FakeRetriever(
        result=result
    )

    app.state.reranked_retriever = (
        fake_retriever
    )

    client = TestClient(
        app
    )

    return (
        client,
        fake_retriever,
    )


def test_retrieve_endpoint_returns_results() -> None:
    """/api/v1/retrieve 应返回结构化法规 Retrieval Result。"""

    client, _ = build_client()

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "测试问题",
            "role": "guest",
            "top_k": 5,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert payload[
        "query"
    ] == "测试问题"

    assert payload[
        "role"
    ] == "guest"

    assert payload[
        "result_count"
    ] == 1

    assert len(
        payload["results"]
    ) == 1

    first = (
        payload["results"][0]
    )

    assert first[
        "chunk_id"
    ] == "test_chunk"

    assert first[
        "title"
    ] == "测试法规"

    assert first[
        "article_number"
    ] == "第一条"

    assert first[
        "rerank_score"
    ] == 5.5


def test_retrieve_supports_technical_document_without_article_number(
) -> None:
    """技术文档 article_number=None 应合法返回 HTTP 200。

    这是针对真实 Demo Contract Audit 中发现的：

        technical document
        ↓
        article_number=None
        ↓
        API Schema 错误要求 str
        ↓
        HTTP 500

    增加的 Regression Test。
    """

    client, _ = build_client(
        result=make_technical_result()
    )

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": (
                "Qdrant 中 Payload Filter "
                "是如何过滤结果的？"
            ),
            "role": "developer",
            "top_k": 5,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["role"]
        == "developer"
    )

    assert (
        payload["result_count"]
        == 1
    )

    first = (
        payload["results"][0]
    )

    assert (
        first["chunk_id"]
        == "qdrant_filtering_1"
    )

    assert (
        first["title"]
        == "Qdrant Filtering"
    )

    assert (
        first["access_level"]
        == "developer"
    )

    # ------------------------------------------------------
    # 核心 Regression Assertion：
    #
    # 技术文档没有法规 Article Number，
    # None 应序列化成 JSON null，
    # 不能触发 ResponseValidationError。
    # ------------------------------------------------------

    assert (
        first["article_number"]
        is None
    )

    assert (
        first["rerank_score"]
        == 4.4219
    )

    assert (
        first["dense_rank"]
        == 1
    )

    assert (
        first["bm25_rank"]
        is None
    )


def test_retrieve_defaults_to_guest() -> None:
    """Request 未传 role 时 API Schema 应默认 guest。"""

    client, retriever = (
        build_client()
    )

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "测试问题"
        },
    )

    assert (
        response.status_code
        == 200
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


def test_retrieve_propagates_developer_role() -> None:
    """developer 应正确转成 AccessContext 并传入 Retriever。"""

    client, retriever = (
        build_client()
    )

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "测试问题",
            "role": "developer",
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        retriever.last_access_context
        is not None
    )

    assert (
        retriever
        .last_access_context
        .role
        == UserRole.DEVELOPER
    )


def test_retrieve_propagates_top_k() -> None:
    """top_k 应正确传给 Retriever。"""

    client, retriever = (
        build_client()
    )

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "测试问题",
            "top_k": 7,
        },
    )

    assert (
        response.status_code
        == 200
    )

    assert (
        retriever.last_top_k
        == 7
    )


def test_retrieve_rejects_invalid_role() -> None:
    """非法 role 应由 Pydantic 在 HTTP 边界直接拒绝。"""

    client, _ = build_client()

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "测试问题",
            "role": "superuser",
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_retrieve_rejects_invalid_top_k() -> None:
    """top_k > 20 应直接返回 422。"""

    client, _ = build_client()

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "测试问题",
            "top_k": 100,
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_retrieve_returns_503_when_not_initialized() -> None:
    """Retrieval Service 未初始化时应返回 HTTP 503。"""

    app = create_app(
        use_lifespan=False
    )

    client = TestClient(
        app
    )

    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "测试问题"
        },
    )

    assert (
        response.status_code
        == 503
    )