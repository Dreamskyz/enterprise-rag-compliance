"""Ingestion 阶段使用的数据模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedDocument:
    """
    表示一篇已经完成正文抽取和文本标准化的文档。

    注意：
    此时仍然是“整篇文档”，还没有切成 Chunk。
    """

    document_id: str
    title: str
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