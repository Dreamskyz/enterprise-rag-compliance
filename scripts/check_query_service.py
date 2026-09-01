"""检查完整 QueryService 在线业务链。"""

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
from enterprise_rag.service.query_service import (
    QueryService,
)


CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# 仅用于当前开发 Smoke Test。
#
# 不是正式生产阈值。
#
# Part 5.4 preliminary inspection 中：
#
# Hard Positive min ≈ -2.09
# 明显 OOD 多在 -6 ~ -8。
#
# 因此这里暂时使用 -3.0，
# 让 Stage 1 偏宽松，降低 False Reject。
#
# 正式值后续必须通过 Evaluation 标定。
# ==========================================================

DEMO_COARSE_RELEVANCE_THRESHOLD = -3.0


TEST_CASES = [
    (
        "ANSWERABLE",
        (
            "生成式人工智能服务处理训练数据"
            "需要遵守什么规定？"
        ),
    ),
    (
        "HARD_NEGATIVE",
        (
            "生成式人工智能服务管理暂行办法"
            "规定发现违法内容后"
            "必须在几小时内处理？"
        ),
    ),
    (
        "OUT_OF_DOMAIN",
        "南京明天会下雨吗？",
    ),
]


def main() -> None:
    """
    验证完整 QueryService：

        Access
          ↓
        Retrieval
          ↓
        Rerank
          ↓
        Coarse Gate
          ↓
        Grounded Generation
          ↓
        Answer / Refusal
    """

    print("=" * 100)
    print(
        "QueryService Full Pipeline Check"
    )
    print("=" * 100)

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    print(
        "Chunk count:",
        len(chunks),
    )

    print(
        "Demo coarse threshold:",
        DEMO_COARSE_RELEVANCE_THRESHOLD,
    )

    # --------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    # --------------------------------------------------
    # 2. Dense
    # --------------------------------------------------

    dense_retriever = DenseRetriever(
        embedding_service=(
            embedding_service
        )
    )

    # --------------------------------------------------
    # 3. BM25
    # --------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    # --------------------------------------------------
    # 4. Hybrid
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
    # 5. Reranker
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
    # 6. Stage 1:
    #    Coarse Relevance Gate
    # --------------------------------------------------

    evidence_gate = EvidenceGate(
        min_top_score=(
            DEMO_COARSE_RELEVANCE_THRESHOLD
        )
    )

    # --------------------------------------------------
    # 7. Stage 2:
    #    Evidence-Constrained Generation
    # --------------------------------------------------

    llm_service = (
        SiliconFlowLLMService()
    )

    answerer = (
        EvidenceGroundedAnswerer(
            llm_service=llm_service,
            max_evidence=5,
        )
    )

    # --------------------------------------------------
    # 8. Application Service
    # --------------------------------------------------

    query_service = QueryService(
        retriever=(
            reranked_retriever
        ),
        evidence_gate=(
            evidence_gate
        ),
        answerer=answerer,
        retrieval_top_k=5,
    )

    # --------------------------------------------------
    # 9. 三类真实问题。
    # --------------------------------------------------

    for label, query in TEST_CASES:
        print()
        print("=" * 100)

        print(
            "Case:",
            label,
        )

        print(
            "Query:",
            query,
        )

        print("=" * 100)

        result = query_service.ask(
            query=query
        )

        print(
            "Role:",
            result.role.value,
        )

        print(
            "Retrieval Count:",
            result.retrieval_count,
        )

        print(
            "Top Rerank Score:",
            result.top_rerank_score,
        )

        print(
            "Gate Reason:",
            result.gate_reason,
        )

        print(
            "Answerable:",
            result.answerable,
        )

        print(
            "Answer:",
            result.answer,
        )

        print(
            "Reason:",
            result.reason,
        )

        print(
            "Citations:"
        )

        for citation in (
            result.citations
        ):
            print(
                "  -",
                citation.evidence_id,
                "|",
                citation.title,
                citation.article_number,
                "|",
                citation.chunk_id,
            )

        # --------------------------------------------------
        # 行为断言
        # --------------------------------------------------

        if label == "ANSWERABLE":
            assert (
                result.gate_reason
                == "passed"
            )

            assert (
                result.answerable
                is True
            )

            assert (
                result.answer
                is not None
            )

            assert (
                len(
                    result.citations
                )
                >= 1
            )

        elif label == "HARD_NEGATIVE":
            # 高相关，所以 Stage 1 应通过。
            assert (
                result.gate_reason
                == "passed"
            )

            # 但 Evidence 不足，
            # Stage 2 应拒答。
            assert (
                result.answerable
                is False
            )

            assert (
                result.answer
                is None
            )

        elif label == "OUT_OF_DOMAIN":
            # 明显 OOD 应在 Stage 1
            # 被程序直接拒绝。
            assert (
                result.gate_reason
                == "below_threshold"
            )

            assert (
                result.answerable
                is False
            )

            assert (
                result.answer
                is None
            )

            assert (
                result.citations
                == ()
            )

    print()
    print("=" * 100)

    print(
        "✅ QueryService 完整在线业务链"
        "验证通过"
    )

    print()

    print(
        "⚠ 当前 -3.0 仅为开发阶段 "
        "Coarse Relevance Gate "
        "Smoke-Test Threshold。"
    )

    print(
        "⚠ 正式阈值将在 Evaluation "
        "阶段重新标定。"
    )


if __name__ == "__main__":
    main()