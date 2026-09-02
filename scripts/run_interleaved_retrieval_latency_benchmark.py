"""运行 Interleaved Retrieval Latency Benchmark。"""

from pathlib import Path

import torch

from enterprise_rag.acl.models import (
    AccessContext,
)
from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.evaluation.latency_benchmark import (
    InterleavedMethodConfig,
    LatencyBenchmarkResult,
    benchmark_interleaved_methods,
)
from enterprise_rag.evaluation.retrieval_runner import (
    RetrievalMethod,
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
# Dataset / Corpus。
# ==========================================================

EVAL_PATH = Path(
    "data/eval/retrieval_eval_v1.jsonl"
)

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# Benchmark Configuration。
# ==========================================================

RETRIEVAL_TOP_K = 10

WARMUP_COUNT = 2

MEASURED_ROUNDS = 5

RANDOM_SEED = 42


def print_latency_table(
    results: list[
        LatencyBenchmarkResult
    ],
) -> None:
    """
    打印 Interleaved Benchmark Summary。
    """

    print()
    print("=" * 120)
    print(
        "Interleaved Retrieval "
        "Steady-State Latency Summary"
    )
    print("=" * 120)

    print(
        "Method | Samples | Mean(ms) | "
        "P50(ms) | P95(ms) | Min(ms) | Max(ms)"
    )

    print(
        "--- | --- | --- | --- | --- | --- | ---"
    )

    for result in results:
        print(
            f"{result.method.value} | "
            f"{result.sample_count} | "
            f"{result.mean_ms:.2f} | "
            f"{result.p50_ms:.2f} | "
            f"{result.p95_ms:.2f} | "
            f"{result.min_ms:.2f} | "
            f"{result.max_ms:.2f}"
        )


def main() -> None:
    """
    使用交错 Method 顺序重新测量：

        Dense
        BM25
        Hybrid RRF
        Hybrid + Rerank

    目的：

        减少旧版 Blocked Benchmark 中：

            Dense × 100
            BM25 × 100
            Hybrid × 100
            Rerank × 100

        引入的 Method Order Bias。
    """

    print("=" * 120)
    print(
        "Interleaved Retrieval "
        "Steady-State Latency Benchmark"
    )
    print("=" * 120)

    # ======================================================
    # 1. Dataset。
    #
    # Latency 使用全部 Query。
    # ======================================================

    cases = (
        read_retrieval_eval_jsonl(
            EVAL_PATH
        )
    )

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    print(
        "Benchmark queries:",
        len(cases),
    )

    print(
        "Knowledge chunks:",
        len(chunks),
    )

    print(
        "Warmup count / method:",
        WARMUP_COUNT,
    )

    print(
        "Measured rounds:",
        MEASURED_ROUNDS,
    )

    print(
        "Expected samples / method:",
        len(cases)
        * MEASURED_ROUNDS,
    )

    # ======================================================
    # 2. Shared Runtime。
    #
    # 所有 Method 共用同一组模型实例。
    # ======================================================

    print()
    print(
        "Initializing shared "
        "retrieval runtime..."
    )

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
    # 3. Evaluation Adapter。
    # ======================================================

    def dense_adapter(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
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
        return reranked_retriever.search(
            query=query,
            top_k=top_k,
            access_context=(
                access_context
            ),
        )

    # ======================================================
    # 4. CUDA Sync。
    # ======================================================

    synchronize_fn = (
        torch.cuda.synchronize
        if torch.cuda.is_available()
        else None
    )

    print()

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "CUDA device:",
            torch.cuda.get_device_name(
                0
            ),
        )

    # ======================================================
    # 5. Interleaved Benchmark。
    # ======================================================

    method_configs = [
        InterleavedMethodConfig(
            method=(
                RetrievalMethod.DENSE
            ),
            retrieve_fn=(
                dense_adapter
            ),
            synchronize_fn=(
                synchronize_fn
            ),
        ),
        InterleavedMethodConfig(
            method=(
                RetrievalMethod.BM25
            ),
            retrieve_fn=(
                bm25_adapter
            ),
            synchronize_fn=None,
        ),
        InterleavedMethodConfig(
            method=(
                RetrievalMethod.HYBRID_RRF
            ),
            retrieve_fn=(
                hybrid_adapter
            ),
            synchronize_fn=(
                synchronize_fn
            ),
        ),
        InterleavedMethodConfig(
            method=(
                RetrievalMethod.HYBRID_RERANK
            ),
            retrieve_fn=(
                reranked_adapter
            ),
            synchronize_fn=(
                synchronize_fn
            ),
        ),
    ]

    print()
    print(
        "Running interleaved benchmark..."
    )

    results = (
        benchmark_interleaved_methods(
            cases=cases,
            method_configs=(
                method_configs
            ),
            retrieval_top_k=(
                RETRIEVAL_TOP_K
            ),
            rounds=(
                MEASURED_ROUNDS
            ),
            warmup_count=(
                WARMUP_COUNT
            ),
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    # ======================================================
    # 6. Report。
    # ======================================================

    print_latency_table(
        results
    )

    print()
    print("=" * 120)

    print(
        "⚠ 本结果使用 Interleaved Method Order，"
        "用于降低 Blocked Benchmark 的 "
        "Method Order Bias。"
    )

    print(
        "⚠ 不包含模型初始化、FastAPI、"
        "网络传输与 LLM latency。"
    )

    print(
        "⚠ Benchmark 环境："
    )

    print(
        "   GPU:",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
    )

    print(
        "   Query count:",
        len(cases),
    )

    print(
        "   Samples / method:",
        len(cases)
        * MEASURED_ROUNDS,
    )


if __name__ == "__main__":
    main()