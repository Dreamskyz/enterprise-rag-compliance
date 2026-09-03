"""Ingestion 阶段使用的数据模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedDocument:
    """
    已完成正文抽取和标准化的一整篇知识库文档。
    """

    document_id: str
    title: str

    document_type: str
    language: str
    version: str

    text: str

    source_url: str
    access_level: str


@dataclass(frozen=True)
class RegulationArticle:
    """
    表示法规中的一条。

    例如：

        第七条
        生成式人工智能服务提供者应当……
    """

    article_number: str
    content: str


@dataclass(frozen=True)
class RegulationChapter:
    """
    表示法规中的一章。

    一章下面包含若干 RegulationArticle。
    """

    chapter_number: str
    title: str
    articles: list[RegulationArticle]


@dataclass(frozen=True)
class GenericSection:
    """
    表示非法规文档中的一个结构化 Section。

    主要用于：

        OWASP 安全规范
        FastAPI 官方文档
        Qdrant 官方文档

    等采用标题层级组织内容的文档。

    title：
        当前 Section 自己的标题。

        例如：

            Classes as Dependencies

    level：
        原始 Markdown Heading Level。

        例如：

            # Title
                → level=1

            ## Dependencies
                → level=2

            ### Classes as Dependencies
                → level=3

        level 主要用于 Parser 恢复标题层级，
        暂时不进入最终 KnowledgeChunk。

    path：
        从祖先标题到当前标题组成的完整路径。

        例如：

            FastAPI > Dependencies > Classes as Dependencies

    content：
        只属于当前 Section 自己的正文。

        子 Section 的正文不会自动复制到父 Section，
        从而避免后续 Chunk 出现大量重复内容。
    """

    title: str
    level: int
    path: str
    content: str


@dataclass(frozen=True)
class KnowledgeChunk:
    """
    最终进入知识库检索层的统一标准知识单元。

    不同类型文档在 Ingestion 上游可以拥有不同的结构：

        regulation
            → chapter / article

        security_guideline
            → section hierarchy

        technical_documentation
            → section hierarchy

    但最终都会统一转换为 KnowledgeChunk，
    从而让下游：

        BM25
        Embedding
        Qdrant
        Rerank
        ACL
        Evaluation

    不需要关心原始文档使用的是哪一种 Parser。

    content：
        原始正文，用于回答、引用和展示。

    retrieval_text：
        加入标题以及结构上下文后的检索文本，
        用于后续 BM25、Embedding 和 Rerank。

    chapter_number / chapter_title / article_number：
        主要供法规类文档使用。

        技术文档或安全规范没有对应法规结构时，
        可以显式传入 None。

    section_title：
        当前 Chunk 直接所属的 Section 标题。

        例如：

            Classes as Dependencies

    section_path：
        当前 Chunk 的完整标题层级路径。

        例如：

            Tutorial > Dependencies > Classes as Dependencies

        主要用于 OWASP、FastAPI、Qdrant
        等非法规结构文档。
    """

    chunk_id: str

    # ======================================================
    # 文档身份。
    # ======================================================

    document_id: str
    title: str

    # ======================================================
    # 文档级元数据。
    # ======================================================

    document_type: str
    language: str
    version: str

    # ======================================================
    # 法规结构。
    #
    # 过去这里只有法规，所以三个字段都是 str。
    #
    # Day 6 开始引入：
    #
    #   OWASP
    #   FastAPI
    #   Qdrant
    #
    # 这些文档没有“第几章 / 第几条”的概念，
    # 因此类型升级为 str | None。
    #
    # 注意：
    # 这里仍然没有给默认值，
    # 是为了保持原有构造函数参数 Contract，
    # 避免已有代码在 Schema Evolution 中静默变化。
    # ======================================================

    chapter_number: str | None
    chapter_title: str | None
    article_number: str | None

    # ======================================================
    # 文本。
    # ======================================================

    content: str
    retrieval_text: str

    # ======================================================
    # 来源与权限。
    # ======================================================

    source_url: str
    access_level: str

    # ======================================================
    # Chunk 内部信息。
    # ======================================================

    chunk_index: int
    content_hash: str

    # ======================================================
    # 通用 Section 结构。
    #
    # 这两个字段是 Day 6 新增字段。
    #
    # 放在所有无默认值字段之后非常重要：
    #
    # Python dataclass 不允许：
    #
    #   有默认值字段
    #   ↓
    #   无默认值字段
    #
    # 否则会产生：
    #
    #   TypeError:
    #   non-default argument follows default argument
    #
    # 给默认值 None 的另一个好处是：
    #
    # 原有 Regulation Chunker 不需要立刻修改。
    # ======================================================

    section_title: str | None = None
    section_path: str | None = None