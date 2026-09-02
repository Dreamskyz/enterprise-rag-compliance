"""运行 Retrieval Ablation Evaluation。"""

from pathlib import Path

from enterprise_rag.acl.models import (
    AccessContext,
)
from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.evaluation.retrieval_analysis import (
    find_rank_degradations,
    find_rank_improvements,
    print_comparison_cases,
)
from enterprise_rag.evaluation.retrieval_report import (
    print_retrieval_summary,
)
from enterprise_rag.evaluation.retrieval_runner import (
    RetrievalMethod,
    evaluate_retrieval_method,
)
from enterprise_rag.ingestion.chunk_store import (
    read_chunks_jsonl,
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


# ==========================================================
# Evaluation Dataset。
# ==========================================================

EVAL_PATH = Path(
    "data/eval/retrieval_eval_v1.jsonl"
)


# ==========================================================
# 当前真实 KnowledgeChunk 数据。
# ==========================================================

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# 所有 Retrieval Method 使用完全相同的 Evaluation K。
#
# 后续统一比较：
#
# Recall@1
# Recall@3
# Recall@5
# Recall@10
# MRR@10
# ==========================================================

EVALUATION_KS = [
    1,
    3,
    5,
    10,
]


# ==========================================================
# 每种 Retriever 最终至少返回 Top10，
# 这样才能公平计算 Recall@10 / MRR@10。
#
# 注意：
#
# Hybrid 内部仍然可以使用：
#
# Dense Top20
# BM25 Top20
# Hybrid Top20
#
# 这里只控制 Evaluation 最终观察的 Ranked Results 数量。
# ==========================================================

RETRIEVAL_TOP_K = 10


def main() -> None:
    """
    在统一 Dataset 上运行 Retrieval Ablation：

        Dense
        BM25
        Hybrid RRF
        Hybrid + Rerank

    当前 Retrieval Recall / MRR
    只评估：

        answerable = true

    Unanswerable Query 后续进入：
        Refusal Evaluation。
    """

    print("=" * 100)
    print(
        "Retrieval Ablation Evaluation"
    )
    print("=" * 100)

    # ======================================================
    # 1. 读取统一 Evaluation Dataset。
    # ======================================================

    cases = (
        read_retrieval_eval_jsonl(
            EVAL_PATH
        )
    )

    # ======================================================
    # 2. 读取当前真实 KnowledgeChunk Corpus。
    # ======================================================

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    answerable_count = sum(
        1
        for case in cases
        if case.answerable
    )

    print(
        "Eval cases:",
        len(cases),
    )

    print(
        "Answerable cases:",
        answerable_count,
    )

    print(
        "Knowledge chunks:",
        len(chunks),
    )

    # ======================================================
    # 3. 初始化 Retrieval Runtime。
    #
    # 非常重要：
    #
    # 以下组件只初始化一次。
    #
    # 不能为 Dense / Hybrid / Rerank
    # 分别重新加载 BGE-M3。
    # ======================================================

    print()
    print(
        "Initializing shared "
        "retrieval runtime..."
    )

    # ------------------------------------------------------
    # BGE-M3：
    #
    # Dense 和 Hybrid 共用同一实例。
    # ------------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    # ------------------------------------------------------
    # Dense Retriever。
    # ------------------------------------------------------

    dense_retriever = DenseRetriever(
        embedding_service=(
            embedding_service
        )
    )

    # ------------------------------------------------------
    # BM25：
    #
    # 只根据当前 Chunk Corpus 构建一次。
    # ------------------------------------------------------

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

    # ------------------------------------------------------
    # Hybrid：
    #
    # Dense Top20
    # +
    # BM25 Top20
    # ↓
    # RRF(k=60)
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Cross-Encoder Reranker：
    #
    # 只加载一次。
    # ------------------------------------------------------

    reranker_service = (
        BGERerankerService()
    )

    # ------------------------------------------------------
    # Hybrid + Rerank。
    # ------------------------------------------------------

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

    print(
        "Shared retrieval runtime "
        "initialized."
    )

    # ======================================================
    # 4. Adapter Layer。
    #
    # Evaluation Runner 只认识：
    #
    #     (query, top_k, access_context)
    #
    # 但是生产 Retriever 的 search()
    # 参数形式可能略有不同。
    #
    # 因此这里只在 Evaluation 层适配，
    # 不修改已经稳定的生产 Retriever API。
    # ======================================================

    def dense_adapter(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        """
        DenseRetriever Evaluation Adapter。
        """

        return dense_retriever.search(
            query=query,
            top_k=top_k,
            role=access_context.role,
        )

    def bm25_adapter(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        """
        BM25Retriever Evaluation Adapter。
        """

        return bm25_retriever.search(
            query=query,
            top_k=top_k,
            role=access_context.role,
        )

    def hybrid_adapter(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        """
        HybridRetriever Evaluation Adapter。
        """

        return hybrid_retriever.search(
            query=query,
            top_k=top_k,
            access_context=(
                access_context
            ),
        )

    def reranked_adapter(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        """
        RerankedRetriever Evaluation Adapter。
        """

        return reranked_retriever.search(
            query=query,
            top_k=top_k,
            access_context=(
                access_context
            ),
        )

    # ======================================================
    # 5. 使用同一 Dataset / Metrics
    #    分别评估四种 Retrieval Method。
    # ======================================================

    results = []

    # ------------------------------------------------------
    # Dense。
    # ------------------------------------------------------

    print()
    print(
        "[1/4] Evaluating Dense..."
    )

    dense_result = (
        evaluate_retrieval_method(
            method=(
                RetrievalMethod.DENSE
            ),
            cases=cases,
            retrieve_fn=(
                dense_adapter
            ),
            evaluation_ks=(
                EVALUATION_KS
            ),
            retrieval_top_k=(
                RETRIEVAL_TOP_K
            ),
        )
    )

    results.append(
        dense_result
    )

    # ------------------------------------------------------
    # BM25。
    # ------------------------------------------------------

    print(
        "[2/4] Evaluating BM25..."
    )

    bm25_result = (
        evaluate_retrieval_method(
            method=(
                RetrievalMethod.BM25
            ),
            cases=cases,
            retrieve_fn=(
                bm25_adapter
            ),
            evaluation_ks=(
                EVALUATION_KS
            ),
            retrieval_top_k=(
                RETRIEVAL_TOP_K
            ),
        )
    )

    results.append(
        bm25_result
    )

    # ------------------------------------------------------
    # Hybrid RRF。
    # ------------------------------------------------------

    print(
        "[3/4] Evaluating Hybrid RRF..."
    )

    hybrid_result = (
        evaluate_retrieval_method(
            method=(
                RetrievalMethod.HYBRID_RRF
            ),
            cases=cases,
            retrieve_fn=(
                hybrid_adapter
            ),
            evaluation_ks=(
                EVALUATION_KS
            ),
            retrieval_top_k=(
                RETRIEVAL_TOP_K
            ),
        )
    )

    results.append(
        hybrid_result
    )

    # ------------------------------------------------------
    # Hybrid + Rerank。
    # ------------------------------------------------------

    print(
        "[4/4] Evaluating "
        "Hybrid + Rerank..."
    )

    rerank_result = (
        evaluate_retrieval_method(
            method=(
                RetrievalMethod.HYBRID_RERANK
            ),
            cases=cases,
            retrieve_fn=(
                reranked_adapter
            ),
            evaluation_ks=(
                EVALUATION_KS
            ),
            retrieval_top_k=(
                RETRIEVAL_TOP_K
            ),
        )
    )

    results.append(
        rerank_result
    )

    # ======================================================
    # 6. Aggregate Summary。
    # ======================================================

    print()
    print("=" * 100)
    print(
        "Retrieval Ablation Summary"
    )
    print("=" * 100)

    print_retrieval_summary(
        results=results,
        evaluation_ks=(
            EVALUATION_KS
        ),
    )

    # ======================================================
    # 7. 建立 Method → Result Mapping。
    #
    # 后面的 Failure Analysis
    # 不依赖 list index。
    # ======================================================

    results_by_method = {
        result.method: result
        for result in results
    }

    dense_result = (
        results_by_method[
            RetrievalMethod.DENSE
        ]
    )

    hybrid_result = (
        results_by_method[
            RetrievalMethod.HYBRID_RRF
        ]
    )

    rerank_result = (
        results_by_method[
            RetrievalMethod.HYBRID_RERANK
        ]
    )

    # ======================================================
    # 8. Failure Analysis：
    #
    # Dense
    # ↓
    # Hybrid RRF
    #
    # 找出融合后第一个 Gold 排名变差的 Query。
    # ======================================================

    dense_to_hybrid_degradations = (
        find_rank_degradations(
            left=dense_result,
            right=hybrid_result,
        )
    )

    print_comparison_cases(
        title=(
            "Dense -> Hybrid RRF "
            "Rank Degradations"
        ),
        cases=(
            dense_to_hybrid_degradations
        ),
    )

    # ======================================================
    # 9. Failure Recovery Analysis：
    #
    # Hybrid RRF
    # ↓
    # Reranker
    #
    # 找出 Cross-Encoder 重新提升
    # 第一个 Gold 排名的 Query。
    # ======================================================

    hybrid_to_rerank_improvements = (
        find_rank_improvements(
            left=hybrid_result,
            right=rerank_result,
        )
    )

    print_comparison_cases(
        title=(
            "Hybrid RRF -> Rerank "
            "Rank Improvements"
        ),
        cases=(
            hybrid_to_rerank_improvements
        ),
    )

    # ======================================================
    # 10. 当前实验边界提示。
    # ======================================================

    print()
    print("=" * 100)

    print(
        "⚠ 当前为 Seed Dataset 上的 "
        "Preliminary Retrieval Ablation。"
    )

    print(
        "⚠ 当前 Answerable Case 数量：",
        answerable_count,
    )

    print(
        "⚠ 当前 latency 仍为单次 Query "
        "mean latency，"
        "尚未进行 warmup / repeated runs / "
        "p50 / p95 正式统计。"
    )

    print(
        "⚠ 当前结果不用于调 RRF 参数；"
        "先进行 Failure Case 分析，"
        "避免 Evaluation Leakage。"
    )


if __name__ == "__main__":
    main()