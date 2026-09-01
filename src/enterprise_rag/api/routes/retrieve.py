"""Retrieval API 路由。"""

from fastapi import (
    APIRouter,
    Request,
)

from enterprise_rag.acl.models import (
    AccessContext,
)
from enterprise_rag.api.dependencies import (
    get_reranked_retriever,
)
from enterprise_rag.api.schemas.retrieval import (
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResultItem,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["retrieval"],
)


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="执行 ACL-aware 混合检索",
)
def retrieve(
    payload: RetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    """
    执行完整 Retrieval Pipeline：

        AccessContext
            ↓
        ACL-aware Dense
            +
        ACL-aware BM25
            ↓
        RRF
            ↓
        Reranker

    本接口不调用 LLM。
    """

    access_context = AccessContext(
        role=payload.role
    )

    retriever = (
        get_reranked_retriever(
            request
        )
    )

    results = retriever.search(
        query=payload.query.strip(),
        top_k=payload.top_k,
        access_context=access_context,
    )

    response_items: list[
        RetrieveResultItem
    ] = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        candidate = (
            result.candidate
        )

        response_items.append(
            RetrieveResultItem(
                rank=rank,
                chunk_id=(
                    candidate.chunk_id
                ),
                title=candidate.title,
                article_number=(
                    candidate.article_number
                ),
                content=(
                    candidate.content
                ),
                source_url=(
                    candidate.source_url
                ),
                access_level=(
                    candidate.access_level
                ),
                rerank_score=float(
                    result.rerank_score
                ),
                rrf_score=float(
                    result.rrf_score
                ),
                dense_rank=(
                    result.dense_rank
                ),
                bm25_rank=(
                    result.bm25_rank
                ),
            )
        )

    return RetrieveResponse(
        query=payload.query.strip(),
        role=payload.role,
        result_count=len(
            response_items
        ),
        results=response_items,
    )