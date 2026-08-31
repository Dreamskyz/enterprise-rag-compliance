"""检索结果数据模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCandidate:
    """
    统一检索候选对象。

    Dense、BM25、RRF、Reranker 后续都围绕同一份
    Chunk 基础信息工作。

    这样可以避免每个检索阶段都复制一套
    title/content/source_url 等字段。
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


@dataclass(frozen=True)
class DenseSearchResult:
    """
    Dense Retriever 返回结果。
    """

    candidate: RetrievalCandidate

    # Qdrant Cosine 相似度。
    score: float


@dataclass(frozen=True)
class BM25SearchResult:
    """
    BM25 Retriever 返回结果。
    """

    candidate: RetrievalCandidate

    # BM25 原始相关性分数。
    score: float


@dataclass(frozen=True)
class HybridSearchResult:
    """
    RRF 融合后的检索结果。
    """

    candidate: RetrievalCandidate

    # RRF 最终分数。
    rrf_score: float

    # 如果进入 Dense Top-K，
    # 则记录 Dense 排名；否则为 None。
    dense_rank: int | None

    # 如果进入 BM25 Top-K，
    # 则记录 BM25 排名；否则为 None。
    bm25_rank: int | None

    # 原始分数仅用于 Debug / 分析。
    # Hybrid 不直接使用这些数值做加法。
    dense_score: float | None

    bm25_score: float | None


@dataclass(frozen=True)
class RerankedSearchResult:
    """
    Reranker 精排后的结果。

    candidate:
        原始知识 Chunk。

    rerank_score:
        Cross-Encoder 对 Query + Passage
        直接计算得到的相关性分数。

    original_rank:
        该候选在 RRF Hybrid 结果中的原始排名。

    rrf_score:
        保留原始 RRF 分数用于 Debug 和评测。
    """

    candidate: RetrievalCandidate

    rerank_score: float

    original_rank: int

    rrf_score: float

    dense_rank: int | None

    bm25_rank: int | None