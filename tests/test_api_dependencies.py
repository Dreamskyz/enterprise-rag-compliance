"""测试 FastAPI Runtime Dependencies。"""

import pytest
from fastapi import (
    HTTPException,
)
from starlette.requests import (
    Request,
)

from enterprise_rag.api.dependencies import (
    get_query_service,
    get_reranked_retriever,
)


def build_request(
    app,
) -> Request:
    """
    构造最小 Starlette Request，
    用于测试 app.state Dependency。
    """

    scope = {
        "type": "http",
        "app": app,
    }

    return Request(
        scope
    )


def test_get_retriever_returns_503_when_missing() -> None:
    """Retriever 缺失应统一返回 503。"""

    class FakeApp:
        pass

    app = FakeApp()

    app.state = type(
        "State",
        (),
        {},
    )()

    request = build_request(
        app
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        get_reranked_retriever(
            request
        )

    assert (
        exc_info.value.status_code
        == 503
    )


def test_get_query_service_returns_503_when_missing() -> None:
    """QueryService 缺失应统一返回 503。"""

    class FakeApp:
        pass

    app = FakeApp()

    app.state = type(
        "State",
        (),
        {},
    )()

    request = build_request(
        app
    )

    with pytest.raises(
        HTTPException
    ) as exc_info:
        get_query_service(
            request
        )

    assert (
        exc_info.value.status_code
        == 503
    )