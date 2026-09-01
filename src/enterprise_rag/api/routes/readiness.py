"""RAG Runtime Readiness 接口。"""

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)

from enterprise_rag.api.schemas.readiness import (
    ReadinessResponse,
)


router = APIRouter(
    tags=["health"],
)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="RAG Runtime 就绪检查",
)
def readiness_check(
    request: Request,
) -> ReadinessResponse:
    """
    判断完整 RAG Runtime 是否已经准备好。

    /health：
        只检查 FastAPI 服务进程是否存活。

    /ready：
        检查核心 RAG 组件是否已经注入并可使用。
    """

    runtime = getattr(
        request.app.state,
        "rag_runtime",
        None,
    )

    retriever = getattr(
        request.app.state,
        "reranked_retriever",
        None,
    )

    query_service = getattr(
        request.app.state,
        "query_service",
        None,
    )

    runtime_ready = (
        runtime is not None
    )

    retrieval_ready = (
        retriever is not None
    )

    query_ready = (
        query_service is not None
    )

    if not (
        runtime_ready
        and retrieval_ready
        and query_ready
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "status": "not_ready",
                "runtime_ready": (
                    runtime_ready
                ),
                "retrieval_ready": (
                    retrieval_ready
                ),
                "query_ready": (
                    query_ready
                ),
            },
        )

    return ReadinessResponse(
        status="ready",
        runtime_ready=True,
        retrieval_ready=True,
        query_ready=True,
        chunk_count=len(
            runtime.chunks
        ),
    )