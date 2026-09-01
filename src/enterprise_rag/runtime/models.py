"""RAG Runtime 数据模型。"""

from dataclasses import dataclass

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.evidence.gate import (
    EvidenceGate,
)
from enterprise_rag.generation.answerer import (
    EvidenceGroundedAnswerer,
)
from enterprise_rag.ingestion.models import (
    KnowledgeChunk,
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
from enterprise_rag.service.query_service import (
    QueryService,
)


@dataclass
class RAGRuntime:
    """
    FastAPI 运行期间复用的完整 RAG Runtime。

    Retrieval 组件：
        chunks
        embedding_service
        dense_retriever
        bm25_retriever
        hybrid_retriever
        reranker_service
        reranked_retriever

    Generation / Application 组件：
        evidence_gate
        llm_service
        grounded_answerer
        query_service

    Runtime 在 Application Startup 时创建一次，
    后续 HTTP Request 全部复用。
    """

    chunks: list[KnowledgeChunk]

    embedding_service: BGEEmbeddingService

    dense_retriever: DenseRetriever

    bm25_retriever: BM25Retriever

    hybrid_retriever: HybridRetriever

    reranker_service: BGERerankerService

    reranked_retriever: RerankedRetriever

    evidence_gate: EvidenceGate

    llm_service: SiliconFlowLLMService

    grounded_answerer: EvidenceGroundedAnswerer

    query_service: QueryService