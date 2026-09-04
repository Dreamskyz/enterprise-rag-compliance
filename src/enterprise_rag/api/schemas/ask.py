"""Ask API 的请求与响应 Schema。"""

from pydantic import (
    BaseModel,
    Field,
)

from enterprise_rag.acl.models import (
    UserRole,
)


class AskRequest(BaseModel):
    """
    /api/v1/ask 请求体。

    query:
        用户问题。

    role:
        当前请求使用的演示角色。

        当前为了 ACL Demo
        由请求显式传入。

        生产环境中应从可信认证系统：

            JWT
            SSO
            Session Claims

        中解析。
    """

    query: str = Field(
        min_length=1,
        description="用户问题",
    )

    role: UserRole = Field(
        default=UserRole.GUEST,
        description=(
            "访问角色："
            "guest / developer / admin"
        ),
    )


class CitationResponse(BaseModel):
    """
    API 返回的一条确定性 Citation。

    Citation Metadata 来自程序内部：

        Evidence ID
        ↓
        Evidence Mapping
        ↓
        KnowledgeChunk

    而不是直接相信 LLM 输出。

    article_number:

        法规 Chunk：
            例如“第十一条”。

        OWASP / FastAPI / Qdrant
        等通用技术文档：
            没有法规条文编号，
            因此允许为 None。

    这里必须保留 Domain Model
    中的真实可空语义，
    不能为了 API 输出方便
    强制转换成空字符串。
    """

    evidence_id: str

    chunk_id: str

    title: str

    # ------------------------------------------------------
    # 法规存在 Article Number，
    # 技术文档不存在。
    #
    # 因此必须允许 None。
    # ------------------------------------------------------

    article_number: str | None

    source_url: str


class AskResponse(BaseModel):
    """
    /api/v1/ask 响应体。

    answerable:
        最终是否能够依据当前知识库回答。

    answer:
        可回答时为最终回答；
        拒答时为 None。

    reason:
        回答 / 拒答原因。

    citations:
        经过程序确定性映射和验证后的 Citation。

    retrieval_count:
        Reranker 最终参与 QueryService 的
        Candidate 数量。

    top_rerank_score:
        Top1 Cross-Encoder relevance score。

        注意：
        这是可观测 relevance signal，
        不是 answerability probability。

    gate_reason:
        Coarse Evidence Gate 结果。
    """

    query: str

    role: UserRole

    answerable: bool

    answer: str | None

    reason: str

    citations: list[
        CitationResponse
    ]

    retrieval_count: int

    top_rerank_score: float | None

    gate_reason: str