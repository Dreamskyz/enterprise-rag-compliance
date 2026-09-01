"""Readiness API Schema。"""

from pydantic import BaseModel


class ReadinessResponse(BaseModel):
    """
    RAG Runtime 就绪状态。

    status:
        ready

    runtime_ready:
        完整 RAGRuntime 是否存在。

    retrieval_ready:
        RerankedRetriever 是否可用。

    query_ready:
        QueryService 是否可用。

    chunk_count:
        当前 Runtime 中加载的 Chunk 数量。
    """

    status: str

    runtime_ready: bool

    retrieval_ready: bool

    query_ready: bool

    chunk_count: int | None