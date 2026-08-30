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
class KnowledgeChunk:
    """
    最终进入知识库检索层的标准知识单元。

    content：
        原始条款正文，用于引用和展示。

    retrieval_text：
        加入标题、章节、条款上下文后的文本，
        用于后续 BM25、Embedding 和 Rerank。
    """

    chunk_id: str

    # 文档身份
    document_id: str
    title: str

    # 文档级元数据
    document_type: str
    language: str
    version: str

    # 法规结构
    chapter_number: str
    chapter_title: str
    article_number: str

    # 文本
    content: str
    retrieval_text: str

    # 来源与权限
    source_url: str
    access_level: str

    # Chunk 内部信息
    chunk_index: int
    content_hash: str