"""构建 RAG 应用运行时。"""

from pathlib import Path

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.evidence.gate import (
    EvidenceGate,
)
from enterprise_rag.generation.answerer import (
    EvidenceGroundedAnswerer,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
)
from enterprise_rag.llm.siliconflow import (
    SiliconFlowLLMService,
)
from enterprise_rag.reranking.bge_reranker import (
    BGERerankerService,
)
from enterprise_rag.retrieval.bm25 import (
    BM25Retriever,
)
from enterprise_rag.retrieval.dense import (
    DenseRetriever,
)
from enterprise_rag.retrieval.hybrid import (
    HybridRetriever,
)
from enterprise_rag.retrieval.reranked import (
    RerankedRetriever,
)
from enterprise_rag.runtime.models import (
    RAGRuntime,
)
from enterprise_rag.service.query_service import (
    QueryService,
)


DEFAULT_CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# 当前仅用于开发阶段的 Coarse Relevance Gate。
#
# Part 5.4 中：
#
# Hard Positive min ≈ -2.09
# 明显 OOD 多在 -6 ~ -8。
#
# 因此暂时使用 -3.0：
#
# - 尽量避免提前误杀 Hard Positive；
# - 明显 OOD 仍可便宜拒绝；
# - 真正 Evidence Sufficiency
#   由 Grounded Generation 判断。
#
# 正式值必须在后续 Evaluation 阶段重新标定。
# ==========================================================

DEMO_COARSE_RELEVANCE_THRESHOLD = -3.0


def build_rag_runtime(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> RAGRuntime:
    """
    构建完整在线 RAG Runtime。

    初始化顺序：

        chunks.jsonl
            ↓
        BGE-M3
            ↓
        DenseRetriever

        chunks
            ↓
        BM25Retriever

        Dense + BM25
            ↓
        HybridRetriever
            ↓
        Reranker
            ↓
        RerankedRetriever
            ↓
        Coarse Evidence Gate
            ↓
        SiliconFlow LLM
            ↓
        EvidenceGroundedAnswerer
            ↓
        QueryService

    此函数只应在 Application Startup 阶段调用。
    """

    # --------------------------------------------------
    # 1. 检查知识库文件。
    # --------------------------------------------------

    if not chunks_path.exists():
        raise FileNotFoundError(
            "知识库 Chunk 文件不存在："
            f"{chunks_path}"
        )

    # --------------------------------------------------
    # 2. 读取 Chunk Corpus。
    # --------------------------------------------------

    chunks = read_chunks_jsonl(
        chunks_path
    )

    if not chunks:
        raise RuntimeError(
            "知识库 Chunk 为空，"
            "无法启动 RAG Runtime"
        )

    # --------------------------------------------------
    # 3. BGE-M3 Embedding。
    # --------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    # --------------------------------------------------
    # 4. Dense Retriever。
    # --------------------------------------------------

    dense_retriever = DenseRetriever(
        embedding_service=(
            embedding_service
        )
    )

    # --------------------------------------------------
    # 5. BM25 Retriever。
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    # --------------------------------------------------
    # 6. Hybrid Retriever。
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=(
            dense_retriever
        ),
        bm25_retriever=(
            bm25_retriever
        ),
        dense_top_k=20,
        bm25_top_k=20,
        rrf_k=60,
    )

    # --------------------------------------------------
    # 7. Cross-Encoder Reranker。
    # --------------------------------------------------

    reranker_service = (
        BGERerankerService()
    )

    reranked_retriever = (
        RerankedRetriever(
            hybrid_retriever=(
                hybrid_retriever
            ),
            reranker_service=(
                reranker_service
            ),
            candidate_top_k=20,
        )
    )

    # --------------------------------------------------
    # 8. Stage 1:
    #    Coarse Relevance Gate。
    # --------------------------------------------------

    evidence_gate = EvidenceGate(
        min_top_score=(
            DEMO_COARSE_RELEVANCE_THRESHOLD
        )
    )

    # --------------------------------------------------
    # 9. SiliconFlow LLM。
    #
    # API Key 从 .env / 环境变量读取。
    # --------------------------------------------------

    llm_service = (
        SiliconFlowLLMService()
    )

    # --------------------------------------------------
    # 10. Stage 2:
    #     Evidence-Constrained Generation。
    # --------------------------------------------------

    grounded_answerer = (
        EvidenceGroundedAnswerer(
            llm_service=llm_service,
            max_evidence=5,
        )
    )

    # --------------------------------------------------
    # 11. Application Service。
    # --------------------------------------------------

    query_service = QueryService(
        retriever=(
            reranked_retriever
        ),
        evidence_gate=(
            evidence_gate
        ),
        answerer=(
            grounded_answerer
        ),
        retrieval_top_k=5,
    )

    return RAGRuntime(
        chunks=chunks,
        embedding_service=(
            embedding_service
        ),
        dense_retriever=(
            dense_retriever
        ),
        bm25_retriever=(
            bm25_retriever
        ),
        hybrid_retriever=(
            hybrid_retriever
        ),
        reranker_service=(
            reranker_service
        ),
        reranked_retriever=(
            reranked_retriever
        ),
        evidence_gate=(
            evidence_gate
        ),
        llm_service=(
            llm_service
        ),
        grounded_answerer=(
            grounded_answerer
        ),
        query_service=(
            query_service
        ),
    )