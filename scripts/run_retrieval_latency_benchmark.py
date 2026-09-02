"""运行 Retrieval Steady-State Latency Benchmark。"""

from pathlib import Path
from time import sleep

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
    LatencyBenchmarkResult,
    benchmark_retrieval_method,
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
#
# 20 Query
# ×
# 5 Rounds
#
# =
# 100 measured samples / Method
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
    打印 Benchmark Summary。
    """

    print()
    print("=" * 120)
    print(
        "Retrieval Steady-State "
        "Latency Summary"
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
    Benchmark：

        Dense
        BM25
        Hybrid RRF
        Hybrid + Rerank

    特点：

        - Runtime 只初始化一次；
        - 初始化时间不计入 Query Latency；
        - Benchmark 前执行 Warmup；
        - 每种 Method 运行 5 轮完整 Query Set；
        - GPU Method 使用 CUDA Synchronize；
        - 最终统计 Mean / P50 / P95。
    """

    print("=" * 120)
    print(
        "Retrieval Steady-State "
        "Latency Benchmark"
    )
    print("=" * 120)

    # ======================================================
    # 1. 读取完整 Seed Dataset。
    #
    # Latency 不区分 Answerable / Unanswerable，
    # 所以全部 Query 都参与。
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
        "Warmup count:",
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
    # 2. Shared Retrieval Runtime。
    #
    # 所有 Method 共用：
    #
    # BGE-M3
    # BM25
    # Reranker
    #
    # 初始化耗时不进入 Retrieval Query Benchmark。
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
    # 4. CUDA Synchronization。
    #
    # BGE / Reranker 当前运行在 CUDA。
    #
    # 如果未来切换到 CPU，
    # 则 synchronize_fn 自动退化为 None。
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
    # 5. Benchmark。
    # ======================================================

    results: list[
        LatencyBenchmarkResult
    ] = []

    print()
    print(
        "[1/4] Benchmarking Dense..."
    )

    dense_result = (
        benchmark_retrieval_method(
            method=(
                RetrievalMethod.DENSE
            ),
            cases=cases,
            retrieve_fn=(
                dense_adapter
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
            synchronize_fn=(
                synchronize_fn
            ),
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    results.append(
        dense_result
    )

    # ------------------------------------------------------
    # 方法之间短暂停顿一下，
    # 减少连续高负载切换带来的干扰。
    #
    # 这不是严格 thermal benchmark，
    # 只是当前本地实验的轻量稳定措施。
    # ------------------------------------------------------

    sleep(1.0)

    print(
        "[2/4] Benchmarking BM25..."
    )

    bm25_result = (
        benchmark_retrieval_method(
            method=(
                RetrievalMethod.BM25
            ),
            cases=cases,
            retrieve_fn=(
                bm25_adapter
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
            synchronize_fn=None,
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    results.append(
        bm25_result
    )

    sleep(1.0)

    print(
        "[3/4] Benchmarking "
        "Hybrid RRF..."
    )

    hybrid_result = (
        benchmark_retrieval_method(
            method=(
                RetrievalMethod.HYBRID_RRF
            ),
            cases=cases,
            retrieve_fn=(
                hybrid_adapter
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
            synchronize_fn=(
                synchronize_fn
            ),
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    results.append(
        hybrid_result
    )

    sleep(1.0)

    print(
        "[4/4] Benchmarking "
        "Hybrid + Rerank..."
    )

    rerank_result = (
        benchmark_retrieval_method(
            method=(
                RetrievalMethod.HYBRID_RERANK
            ),
            cases=cases,
            retrieve_fn=(
                reranked_adapter
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
            synchronize_fn=(
                synchronize_fn
            ),
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    results.append(
        rerank_result
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
        "⚠ 本结果为本机 warmed steady-state "
        "Retrieval Latency Benchmark。"
    )

    print(
        "⚠ 不包含模型初始化、FastAPI、"
        "网络传输和 LLM latency。"
    )

    print(
        "⚠ Hardware / software environment："
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