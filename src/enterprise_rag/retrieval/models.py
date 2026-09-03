"""检索结果数据模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCandidate:
    """
    统一检索候选对象。

    Dense、BM25、RRF、Reranker 后续都围绕同一份
    Chunk 基础信息工作。

    这样可以避免每个检索阶段都复制一套
    title / content / source_url 等字段定义。

    Day 6 开始，知识库从纯法规扩展为：

        regulation
        security_guideline
        technical_documentation

    因此 RetrievalCandidate 同样需要支持两类结构元数据：

        法规结构：
            chapter_number
            chapter_title
            article_number

        通用 Section 结构：
            section_title
            section_path

    注意：

    RetrievalCandidate 是 Retrieval 层的统一 Contract。

    KnowledgeChunk 中能够用于检索结果展示、
    Citation 和 Debug 的结构信息，
    进入 RetrievalCandidate 后不能被丢失。
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
    # 法规文档：
    #
    #   chapter_number="第二章"
    #   chapter_title="技术发展与治理"
    #   article_number="第七条"
    #
    # 技术文档 / 安全规范：
    #
    #   chapter_number=None
    #   chapter_title=None
    #   article_number=None
    #
    # 因此这里必须允许 None。
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
    # Day 6 新增。
    #
    # 对旧法规 Candidate 来说默认都是 None，
    # 因而保持向后兼容。
    #
    # 示例：
    #
    # section_title:
    #     "Classes as Dependencies"
    #
    # section_path:
    #     "Tutorial > Dependencies > "
    #     "Classes as Dependencies"
    # ======================================================

    section_title: str | None = None

    section_path: str | None = None


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