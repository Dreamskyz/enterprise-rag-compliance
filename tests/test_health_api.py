"""测试 FastAPI 健康检查接口。"""

from fastapi.testclient import (
    TestClient,
)

from enterprise_rag.api.app import (
    create_app,
)


def test_health_endpoint() -> None:
    """
    /health 应返回 HTTP 200
    和稳定的健康状态。

    Unit Test 不启动真实 RAG Runtime。
    """

    app = create_app(
        use_lifespan=False
    )

    client = TestClient(
        app
    )

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    assert response.json() == {
        "status": "ok",
        "service": (
            "enterprise-rag-compliance"
        ),
        "version": "0.1.0",
    }