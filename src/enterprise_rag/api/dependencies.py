"""FastAPI 共享依赖解析。"""

from fastapi import (
    HTTPException,
    Request,
    status,
)

from enterprise_rag.retrieval.reranked import (
    RerankedRetriever,
)
from enterprise_rag.service.query_service import (
    QueryService,
)


def get_reranked_retriever(
    request: Request,
) -> RerankedRetriever:
    """
    从 Application State 获取共享 Retriever。

    Retriever 应在 FastAPI Lifespan Startup
    阶段初始化一次。

    如果不存在，说明：

        HTTP 服务进程仍然存活，
        但 Retrieval Runtime 尚不可用。

    此时统一返回 HTTP 503。
    """

    retriever = getattr(
        request.app.state,
        "reranked_retriever",
        None,
    )

    if retriever is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Retrieval service "
                "is not ready"
            ),
        )

    return retriever


def get_query_service(
    request: Request,
) -> QueryService:
    """
    从 Application State 获取共享 QueryService。

    如果 QueryService 尚未初始化，
    统一返回 HTTP 503。
    """

    query_service = getattr(
        request.app.state,
        "query_service",
        None,
    )

    if query_service is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Query service "
                "is not ready"
            ),
        )

    return query_service