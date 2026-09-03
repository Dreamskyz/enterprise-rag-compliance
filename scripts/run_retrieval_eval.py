"""运行 Retrieval Ablation Evaluation。"""

import argparse
from collections import Counter
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
    find_gold_miss_cases,
    find_non_top1_gold_cases,
    find_rank_degradations,
    find_rank_improvements,
    print_comparison_cases,
    print_method_failure_cases,
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
#
# V1:
#     冻结的历史 Regression Benchmark。
#
# V2:
#     当前 Corpus V2 Capability Benchmark。
#
# 不通过修改代码切换 Dataset，
# 而是通过：
#
#     --dataset v1
#     --dataset v2
#
# 显式选择。
# ==========================================================

EVAL_PATHS: dict[
    str,
    Path,
] = {
    "v1": Path(
        "data/eval/retrieval_eval_v1.jsonl"
    ),
    "v2": Path(
        "data/eval/retrieval_eval_v2.jsonl"
    ),
}


# ==========================================================
# 当前真实 KnowledgeChunk 数据。
# ==========================================================

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# 所有 Retrieval Method 使用完全相同的 Evaluation K。
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
# 才能公平计算 Recall@10 / MRR@10。
#
# Hybrid 内部仍然使用：
#
# Dense Top20
# BM25 Top20
# Hybrid Top20
#
# 这里只控制 Evaluation
# 最终观察的 Ranked Results 数量。
# ==========================================================

RETRIEVAL_TOP_K = 10


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    默认保持：

        v1

    原因：

    在加入 --dataset 参数之前，
    当前脚本历史上一直运行的就是：

        retrieval_eval_v1.jsonl

    CLI 演化不应该偷偷改变
    旧命令的实验语义。

    Corpus V2 正式评测时显式执行：

        python scripts/run_retrieval_eval.py --dataset v2
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval ablation evaluation "
            "on a selected evaluation dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(
            EVAL_PATHS.keys()
        ),
        default="v1",
        help=(
            "Evaluation dataset version. "
            "v1=frozen regression benchmark, "
            "v2=current Corpus V2 benchmark. "
            "Default: v1."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    在统一 Dataset 上运行 Retrieval Ablation：

        Dense
        BM25
        Hybrid RRF
        Hybrid + Rerank

    Retrieval Recall / MRR
    当前只评估：

        answerable = true

    Unanswerable Query 后续进入：

        Refusal Evaluation

    ------------------------------------------------------
    Role-aware Evaluation
    ------------------------------------------------------

    Dataset 中每条 Case
    可以携带自己的：

        role

    Retrieval Runner 会根据：

        case.role

    为每条 Query
    构造自己的 AccessContext。

    因此 V2 可以在一次评测中同时包含：

        guest
        developer

    而不会绕过生产 ACL。
    """

    args = parse_args()

    dataset_version = str(
        args.dataset
    )

    eval_path = EVAL_PATHS[
        dataset_version
    ]

    print("=" * 100)
    print(
        "Retrieval Ablation Evaluation"
    )
    print("=" * 100)

    # ======================================================
    # 1. 读取指定 Evaluation Dataset。
    # ======================================================

    cases = (
        read_retrieval_eval_jsonl(
            eval_path
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

    unanswerable_count = (
        len(cases)
        - answerable_count
    )

    role_counts = Counter(
        case.role.value
        for case in cases
    )

    print(
        "Dataset version:",
        dataset_version,
    )

    print(
        "Dataset path:",
        eval_path,
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
        "Unanswerable cases:",
        unanswerable_count,
    )

    print(
        "Role distribution:",
        dict(role_counts),
    )

    print(
        "Knowledge chunks:",
        len(chunks),
    )

    # ======================================================
    # 3. 初始化 Retrieval Runtime。
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
    # BGE-M3。
    #
    # Dense 和 Hybrid
    # 共用同一 Embedding 实例。
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
    # BM25。
    #
    # 只根据当前 Chunk Corpus
    # 构建一次。
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
    # Cross-Encoder Reranker。
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
    # Evaluation Runner 统一调用：
    #
    #     (query, top_k, access_context)
    #
    # 生产 Retriever 的 search()
    # 参数形式可能略有不同。
    #
    # 因此只在 Evaluation Layer 做适配，
    # 不修改生产 Retriever API。
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
    #
    # 不传全局 access_context。
    #
    # 因此 Retrieval Runner 使用：
    #
    #     case.role
    #
    # 为每条 Case
    # 创建自己的 AccessContext。
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

    bm25_result = (
        results_by_method[
            RetrievalMethod.BM25
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
    # 8. Pairwise Failure Analysis：
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
    # 找出 Reranker
    # 把第一个 Gold 向前提升的 Query。
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
    # 10. General Failure Inspector。
    #
    # Pairwise comparison 只能告诉我们：
    #
    #     A → B
    #
    # 是变好还是变坏。
    #
    # 这里进一步检查：
    #
    #     单个 Method 自己
    #     到底在哪些 Query 上失败。
    # ======================================================

    # ------------------------------------------------------
    # Dense：
    #
    # 第一个 Gold 没有排 Rank1 的 Case。
    # ------------------------------------------------------

    dense_non_top1_cases = (
        find_non_top1_gold_cases(
            result=dense_result,
            metric_k=10,
        )
    )

    print_method_failure_cases(
        title=(
            "Dense Non-Top1 Gold Cases"
        ),
        result=dense_result,
        cases=dense_non_top1_cases,
        metric_k=10,
        top_n=5,
    )

    # ------------------------------------------------------
    # BM25：
    #
    # Top10 中完全没有任何 Gold。
    #
    # 这是我们分析中英混合 Corpus
    # lexical mismatch 的关键诊断。
    # ------------------------------------------------------

    bm25_gold_misses = (
        find_gold_miss_cases(
            result=bm25_result,
            metric_k=10,
        )
    )

    print_method_failure_cases(
        title=(
            "BM25 Gold Misses @10"
        ),
        result=bm25_result,
        cases=bm25_gold_misses,
        metric_k=10,
        top_n=5,
    )

    # ------------------------------------------------------
    # Hybrid RRF：
    #
    # 检查第一个 Gold
    # 没有排 Rank1 的 Case。
    # ------------------------------------------------------

    hybrid_non_top1_cases = (
        find_non_top1_gold_cases(
            result=hybrid_result,
            metric_k=10,
        )
    )

    print_method_failure_cases(
        title=(
            "Hybrid RRF Non-Top1 Gold Cases"
        ),
        result=hybrid_result,
        cases=hybrid_non_top1_cases,
        metric_k=10,
        top_n=5,
    )

    # ------------------------------------------------------
    # Hybrid + Rerank：
    #
    # 这是当前最值得看的最终诊断：
    #
    # Reranker 已经修复了一批 RRF failure，
    # 但 Aggregate Metrics 显示它仍没有
    # 完全达到 Dense 的水平。
    #
    # 所以这里直接列出剩余 Case。
    # ------------------------------------------------------

    rerank_non_top1_cases = (
        find_non_top1_gold_cases(
            result=rerank_result,
            metric_k=10,
        )
    )

    print_method_failure_cases(
        title=(
            "Hybrid + Rerank "
            "Non-Top1 Gold Cases"
        ),
        result=rerank_result,
        cases=rerank_non_top1_cases,
        metric_k=10,
        top_n=5,
    )

    # ======================================================
    # 11. 当前实验边界提示。
    # ======================================================

    print()
    print("=" * 100)

    print(
        "⚠ 当前 Retrieval Quality "
        "结果基于 Dataset：",
        dataset_version,
    )

    print(
        "⚠ 当前 Answerable Case 数量：",
        answerable_count,
    )

    print(
        "⚠ Summary 表中的 Mean Latency(ms) "
        "仅作为本次 Ablation Run 的附带诊断信息。"
    )

    print(
        "⚠ 项目正式 Retrieval Latency 结论 "
        "使用 Query-level Interleaved Benchmark："
        "对每个 Query 随机化四种 Method 的执行顺序，"
        "以控制 GPU Warm-State 和 Method Order Bias，"
        "并使用 repeated runs / P50 / P95 进行统计。"
    )

    print(
        "⚠ 当前 Retrieval Quality 结果 "
        "不用于根据测试结果反向修改 Gold，"
        "避免 Evaluation Leakage。"
    )


if __name__ == "__main__":
    main()