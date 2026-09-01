"""测试 RAG Readiness API。"""

from types import SimpleNamespace

from fastapi.testclient import (
    TestClient,
)

from enterprise_rag.api.app import (
    create_app,
)


def test_ready_returns_503_without_runtime() -> None:
    """
    没有 RAG Runtime 时，
    /health 可以活着，
    但 /ready 必须返回 503。
    """

    app = create_app(
        use_lifespan=False
    )

    client = TestClient(
        app
    )

    health_response = client.get(
        "/health"
    )

    ready_response = client.get(
        "/ready"
    )

    assert (
        health_response.status_code
        == 200
    )

    assert (
        ready_response.status_code
        == 503
    )

    detail = (
        ready_response.json()[
            "detail"
        ]
    )

    assert (
        detail["status"]
        == "not_ready"
    )

    assert (
        detail["runtime_ready"]
        is False
    )

    assert (
        detail["retrieval_ready"]
        is False
    )

    assert (
        detail["query_ready"]
        is False
    )


def test_ready_returns_200_when_runtime_available() -> None:
    """
    Runtime / Retriever / QueryService
    都存在时，/ready 应返回 200。
    """

    app = create_app(
        use_lifespan=False
    )

    # --------------------------------------------------
    # Readiness 这里只检查组件是否存在，
    # 不执行真实 GPU / LLM 调用。
    # --------------------------------------------------

    app.state.rag_runtime = (
        SimpleNamespace(
            chunks=[
                object(),
                object(),
            ]
        )
    )

    app.state.reranked_retriever = (
        object()
    )

    app.state.query_service = (
        object()
    )

    client = TestClient(
        app
    )

    response = client.get(
        "/ready"
    )

    assert (
        response.status_code
        == 200
    )

    assert response.json() == {
        "status": "ready",
        "runtime_ready": True,
        "retrieval_ready": True,
        "query_ready": True,
        "chunk_count": 2,
    }