"""Retrieval API 的请求与响应 Schema。"""

from pydantic import (
    BaseModel,
    Field,
)

from enterprise_rag.acl.models import (
    UserRole,
)


class RetrieveRequest(BaseModel):
    """
    /api/v1/retrieve 请求体。

    query:
        用户检索问题。

    role:
        当前请求使用的访问角色。

        当前为了 ACL Demo
        由请求显式传入。

        生产环境应由认证系统
        从 JWT / SSO Claims 中解析。

    top_k:
        返回最终 Reranked Result 的数量。
    """

    query: str = Field(
        min_length=1,
        description="检索问题",
    )

    role: UserRole = Field(
        default=UserRole.GUEST,
        description=(
            "访问角色："
            "guest / developer / admin"
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "返回的 Reranked Result 数量"
        ),
    )


class RetrieveResultItem(BaseModel):
    """
    一条最终 Retrieval Result。

    当前 API 暴露的是：

        Final Reranked Result

    不直接暴露底层：

        Qdrant SDK Object
        Retriever Dataclass

    article_number:

        Regulation Chunk
            → 可能存在，例如“第十四条”。

        OWASP / FastAPI / Qdrant
            → 没有法规 Article Number，
              因此允许为 None。

    这与统一 KnowledgeChunk
    的 Domain Schema 保持一致。
    """

    rank: int

    chunk_id: str

    title: str

    # ------------------------------------------------------
    # 技术文档没有法规条文编号，
    # 因此这里不能强制要求 str。
    # ------------------------------------------------------

    article_number: str | None

    content: str

    source_url: str

    access_level: str

    rerank_score: float

    rrf_score: float

    dense_rank: int | None

    bm25_rank: int | None


class RetrieveResponse(BaseModel):
    """
    /api/v1/retrieve 响应体。
    """

    query: str

    role: UserRole

    result_count: int

    results: list[
        RetrieveResultItem
    ]