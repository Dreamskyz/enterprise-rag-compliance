"""完整 RAG 问答 API 路由。"""

from fastapi import (
    APIRouter,
    Request,
)

from enterprise_rag.acl.models import (
    AccessContext,
)
from enterprise_rag.api.dependencies import (
    get_query_service,
)
from enterprise_rag.api.schemas.ask import (
    AskRequest,
    AskResponse,
    CitationResponse,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["ask"],
)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="执行企业合规 RAG 问答",
)
def ask(
    payload: AskRequest,
    request: Request,
) -> AskResponse:
    """
    执行完整在线 RAG Query：

        HTTP Request
            ↓
        AccessContext
            ↓
        ACL-aware Retrieval
            ↓
        Dense + BM25
            ↓
        RRF
            ↓
        Reranker
            ↓
        Coarse Relevance Gate
            ↓
        Evidence-Constrained Generation
            ↓
        Answer / Refusal
            ↓
        Deterministic Citation
    """

    access_context = AccessContext(
        role=payload.role
    )

    query_service = (
        get_query_service(
            request
        )
    )

    result = query_service.ask(
        query=payload.query.strip(),
        access_context=(
            access_context
        ),
    )

    citations = [
        CitationResponse(
            evidence_id=(
                citation.evidence_id
            ),
            chunk_id=(
                citation.chunk_id
            ),
            title=citation.title,
            article_number=(
                citation.article_number
            ),
            source_url=(
                citation.source_url
            ),
        )
        for citation in result.citations
    ]

    return AskResponse(
        query=result.query,
        role=result.role,
        answerable=(
            result.answerable
        ),
        answer=result.answer,
        reason=result.reason,
        citations=citations,
        retrieval_count=(
            result.retrieval_count
        ),
        top_rerank_score=(
            result.top_rerank_score
        ),
        gate_reason=(
            result.gate_reason
        ),
    )