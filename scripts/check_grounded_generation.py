"""检查 Evidence-Constrained Generation 的真实行为。"""

from pathlib import Path

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
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


CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


TEST_QUERIES = [
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
]


def main() -> None:
    """
    验证两种最核心行为：

    1. 有明确证据
       → 回答 + Citation

    2. Evidence 相关但缺少具体事实
       → Structured Refusal
    """

    print("=" * 100)
    print(
        "Grounded Generation Check"
    )
    print("=" * 100)

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    print(
        "Chunk count:",
        len(chunks),
    )

    # --------------------------------------------------
    # Retrieval Pipeline
    # --------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    dense_retriever = DenseRetriever(
        embedding_service=(
            embedding_service
        )
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

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

    reranker_service = (
        BGERerankerService()
    )

    retriever = RerankedRetriever(
        hybrid_retriever=(
            hybrid_retriever
        ),
        reranker_service=(
            reranker_service
        ),
        candidate_top_k=20,
    )

    # --------------------------------------------------
    # LLM + Grounded Generation
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

    for label, query in TEST_QUERIES:
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

        retrieval_results = (
            retriever.search(
                query=query,
                top_k=5,
            )
        )

        print()
        print("Top Evidence:")

        for rank, result in enumerate(
            retrieval_results,
            start=1,
        ):
            print(
                f"{rank}. "
                f"{result.candidate.title} "
                f"{result.candidate.article_number} "
                f"| rerank="
                f"{result.rerank_score:.4f}"
            )

        grounded_answer = (
            answerer.answer(
                query=query,
                results=(
                    retrieval_results
                ),
            )
        )

        print()
        print(
            "Answerable:",
            grounded_answer.answerable,
        )

        print(
            "Answer:",
            grounded_answer.answer,
        )

        print(
            "Reason:",
            grounded_answer.reason,
        )

        print(
            "Citations:"
        )

        for citation in (
            grounded_answer.citations
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
        # 最关键的行为断言。
        # --------------------------------------------------

        if label == "ANSWERABLE":
            assert (
                grounded_answer.answerable
                is True
            )

            assert (
                grounded_answer.answer
                is not None
            )

            assert (
                len(
                    grounded_answer.citations
                )
                >= 1
            )

        elif label == "HARD_NEGATIVE":
            assert (
                grounded_answer.answerable
                is False
            )

            assert (
                grounded_answer.answer
                is None
            )

            assert (
                grounded_answer.citations
                == ()
            )

    print()
    print("=" * 100)

    print(
        "✅ Evidence-Constrained Generation "
        "真实验证通过"
    )


if __name__ == "__main__":
    main()