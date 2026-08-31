"""检索结果数据模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseSearchResult:
    """
    Dense Retriever 返回的一条检索结果。

    该对象用于隔离上层业务代码与 Qdrant SDK。

    也就是说：
    上层只需要认识 DenseSearchResult，
    不需要认识 Qdrant 的 ScoredPoint。
    """

    chunk_id: str

    document_id: str

    title: str

    document_type: str

    language: str

    version: str

    chapter_number: str

    chapter_title: str

    article_number: str

    content: str

    retrieval_text: str

    source_url: str

    access_level: str

    chunk_index: int

    content_hash: str

    # Dense Retrieval 原始相似度分数。
    #
    # 当前 Qdrant Collection 使用 Cosine Distance，
    # 所以值越大表示语义越相关。
    score: float