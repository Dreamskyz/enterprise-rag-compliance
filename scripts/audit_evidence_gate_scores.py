"""审计 Coarse Evidence Gate 的 Rerank Score 分布。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from enterprise_rag.acl.models import (
    AccessContext,
)
from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
)
from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.evidence.gate import (
    EvidenceGate,
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
from enterprise_rag.runtime.builder import (
    DEMO_COARSE_RELEVANCE_THRESHOLD,
)


# ==========================================================
# Audit Input。
# ==========================================================

EVAL_PATH = Path(
    "data/eval/retrieval_eval_v3.jsonl"
)

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# 与生产 Runtime 保持一致的 Retrieval 参数。
#
# 注意：
# 这里不是重新设计 Retriever，
# 而是在完全相同的生产配置下
# 观察 Evidence Gate 收到的真实分数。
# ==========================================================

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60

RERANK_CANDIDATE_TOP_K = 20

# QueryService 当前送给 Evidence Gate / Answerer
# 的最终 Evidence 数量。
FINAL_TOP_K = 5


@dataclass(frozen=True)
class GateAuditRow:
    """
    一条 Query 的 Gate Audit 结果。

    保存：

    - Query 身份；
    - Gold Answerability；
    - Role；
    - Top5 Rerank Score；
    - 当前 Gate Decision；
    - Top1 Candidate；
    - Retrieval Gold 是否进入 Top5。

    这个模型只服务于离线诊断，
    不进入生产 Runtime。
    """

    query_id: str

    query: str

    category: str

    role: str

    expected_answerable: bool

    gold_chunk_ids: tuple[str, ...]

    top_chunk_id: str | None

    top_scores: tuple[float, ...]

    gate_passed: bool

    gate_reason: str

    threshold: float

    gold_hit_top5: bool


def build_retrieval_runtime() -> tuple[
    RerankedRetriever,
    EvidenceGate,
]:
    """
    构建 Gate Audit 所需的最小 Runtime。

    这里只初始化：

        chunks
        BGE-M3
        Dense
        BM25
        Hybrid RRF
        BGE Reranker
        Evidence Gate

    特意不初始化：

        SiliconFlowLLMService
        EvidenceGroundedAnswerer
        QueryService

    原因：

    当前任务只是分析：

        rerank score
        +
        coarse gate

    完全不需要调用 LLM。

    这样可以保证本脚本：

        不消耗 SiliconFlow API
        不产生生成式模型费用
    """

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "KnowledgeChunk 文件不存在："
            f"{CHUNKS_PATH}"
        )

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    if not chunks:
        raise RuntimeError(
            "KnowledgeChunk Corpus 为空"
        )

    # --------------------------------------------------
    # 1. BGE-M3
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
    # 4. Hybrid RRF
    #
    # 参数必须与生产 Runtime 完全一致。
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=(
            dense_retriever
        ),
        bm25_retriever=(
            bm25_retriever
        ),
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        rrf_k=RRF_K,
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
            candidate_top_k=(
                RERANK_CANDIDATE_TOP_K
            ),
        )
    )

    # --------------------------------------------------
    # 6. 使用当前生产中的 -3.0 Gate。
    #
    # 注意：
    # 本脚本不会修改 threshold，
    # 只观察它当前产生的行为。
    # --------------------------------------------------

    evidence_gate = EvidenceGate(
        min_top_score=(
            DEMO_COARSE_RELEVANCE_THRESHOLD
        )
    )

    return (
        reranked_retriever,
        evidence_gate,
    )


def run_gate_audit() -> list[GateAuditRow]:
    """
    对 V3 全部 Case 执行：

        Retrieval
        ↓
        Rerank
        ↓
        Evidence Gate

    与 Retrieval Eval 不同：

    Retrieval Recall / MRR
    只统计 answerable=true。

    但 Gate Calibration 必须同时观察：

        Answerable Positive
        +
        Unanswerable Negative

    因为我们真正想判断的是：

        Top1 rerank score

    能否在两类 Query 之间
    提供稳定的区分能力。
    """

    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            "Evaluation Dataset 不存在："
            f"{EVAL_PATH}"
        )

    cases = read_retrieval_eval_jsonl(
        EVAL_PATH
    )

    (
        reranked_retriever,
        evidence_gate,
    ) = build_retrieval_runtime()

    rows: list[GateAuditRow] = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(cases):02d}] "
            f"{case.query_id} "
            f"{case.query}"
        )

        # --------------------------------------------------
        # 每条 Case 必须使用自己的 Role。
        #
        # 这样：
        #
        # R042 developer
        # 和
        # R045 guest
        #
        # 会真实经过不同 ACL Candidate Space。
        # --------------------------------------------------

        access_context = AccessContext(
            role=case.role
        )

        results = (
            reranked_retriever.search(
                query=case.query,
                top_k=FINAL_TOP_K,
                access_context=(
                    access_context
                ),
            )
        )

        gate_decision = (
            evidence_gate.evaluate(
                results
            )
        )

        top_scores = tuple(
            float(result.rerank_score)
            for result in results
        )

        top_chunk_id = None

        if results:
            top_chunk_id = (
                results[0]
                .candidate
                .chunk_id
            )

        returned_chunk_ids = {
            result.candidate.chunk_id
            for result in results
        }

        # --------------------------------------------------
        # 多 Gold Case：
        #
        # 当前这里只诊断：
        #
        #     Top5 是否至少命中一个 Gold。
        #
        # 它不是正式 Recall Metric，
        # 只是帮助区分：
        #
        # Retrieval 没找到 Gold
        #
        # 与
        #
        # Retrieval 找到了但 Gate 拒绝
        #
        # 两种 Failure。
        # --------------------------------------------------

        gold_hit_top5 = bool(
            set(case.gold_chunk_ids)
            & returned_chunk_ids
        )

        rows.append(
            GateAuditRow(
                query_id=case.query_id,
                query=case.query,
                category=(
                    case.category.value
                ),
                role=case.role.value,
                expected_answerable=(
                    case.answerable
                ),
                gold_chunk_ids=tuple(
                    case.gold_chunk_ids
                ),
                top_chunk_id=(
                    top_chunk_id
                ),
                top_scores=top_scores,
                gate_passed=(
                    gate_decision.passed
                ),
                gate_reason=(
                    gate_decision.reason.value
                ),
                threshold=(
                    gate_decision.threshold
                ),
                gold_hit_top5=(
                    gold_hit_top5
                ),
            )
        )

    return rows


def format_scores(
    scores: tuple[float, ...],
) -> str:
    """
    将 Top-K Rerank Score
    格式化为便于终端阅读的字符串。
    """

    if not scores:
        return "[]"

    return "[" + ", ".join(
        f"{score:.4f}"
        for score in scores
    ) + "]"


def print_all_cases(
    rows: list[GateAuditRow],
) -> None:
    """
    打印全部 V3 Case 的 Gate Audit。
    """

    print()
    print("=" * 120)
    print(
        "All V3 Evidence Gate Scores"
    )
    print("=" * 120)

    for row in rows:
        print()

        print(
            row.query_id,
            "|",
            row.query,
        )

        print(
            "Role:",
            row.role,
        )

        print(
            "Category:",
            row.category,
        )

        print(
            "Expected answerable:",
            row.expected_answerable,
        )

        print(
            "Gold hit Top5:",
            row.gold_hit_top5,
        )

        print(
            "Top1 chunk:",
            row.top_chunk_id,
        )

        print(
            "Top5 rerank scores:",
            format_scores(
                row.top_scores
            ),
        )

        print(
            "Threshold:",
            f"{row.threshold:.4f}",
        )

        print(
            "Gate:",
            (
                "PASS"
                if row.gate_passed
                else "REJECT"
            ),
        )

        print(
            "Gate reason:",
            row.gate_reason,
        )


def print_false_negatives(
    rows: list[GateAuditRow],
) -> None:
    """
    打印：

        Gold Answerable
        但是
        Coarse Gate Reject

    的 Case。

    这正对应当前 Full-RAG V3
    的主要 Failure Pattern。
    """

    false_negatives = [
        row
        for row in rows
        if (
            row.expected_answerable
            and not row.gate_passed
        )
    ]

    print()
    print("=" * 120)
    print(
        "Gate False Negatives"
    )
    print("=" * 120)

    if not false_negatives:
        print(
            "No Gate false negatives."
        )
        return

    # Top1 Score 从高到低排列。
    #
    # 可以直接观察哪些 Case
    # 最接近当前 -3.0 threshold。

    false_negatives = sorted(
        false_negatives,
        key=lambda row: (
            row.top_scores[0]
            if row.top_scores
            else float("-inf")
        ),
        reverse=True,
    )

    for row in false_negatives:
        top1_score = (
            row.top_scores[0]
            if row.top_scores
            else None
        )

        print()

        print(
            row.query_id,
            "|",
            row.query,
        )

        print(
            "Role:",
            row.role,
        )

        print(
            "Gold hit Top5:",
            row.gold_hit_top5,
        )

        print(
            "Top1 score:",
            (
                f"{top1_score:.4f}"
                if top1_score is not None
                else "None"
            ),
        )

        print(
            "Top5 scores:",
            format_scores(
                row.top_scores
            ),
        )

        print(
            "Top1 chunk:",
            row.top_chunk_id,
        )


def print_unanswerable_cases(
    rows: list[GateAuditRow],
) -> None:
    """
    打印所有 Gold Unanswerable Case。

    这些 Case 是分析 threshold
    是否能够安全下调时最重要的 Negative。

    注意：

    本脚本不会真的下调 threshold。

    这里只观察：

        如果 Positive 和 Negative
        分数大量重叠，

    那么单个 Top1 raw score threshold
    本身就不是理想的
    Answerability Classifier。
    """

    negatives = [
        row
        for row in rows
        if not row.expected_answerable
    ]

    print()
    print("=" * 120)
    print(
        "Gold Unanswerable Cases"
    )
    print("=" * 120)

    negatives = sorted(
        negatives,
        key=lambda row: (
            row.top_scores[0]
            if row.top_scores
            else float("-inf")
        ),
        reverse=True,
    )

    for row in negatives:
        top1_score = (
            row.top_scores[0]
            if row.top_scores
            else None
        )

        print()

        print(
            row.query_id,
            "|",
            row.query,
        )

        print(
            "Role:",
            row.role,
        )

        print(
            "Category:",
            row.category,
        )

        print(
            "Top1 score:",
            (
                f"{top1_score:.4f}"
                if top1_score is not None
                else "None"
            ),
        )

        print(
            "Top5 scores:",
            format_scores(
                row.top_scores
            ),
        )

        print(
            "Gate:",
            (
                "PASS"
                if row.gate_passed
                else "REJECT"
            ),
        )

        print(
            "Top1 chunk:",
            row.top_chunk_id,
        )


def print_score_distribution(
    rows: list[GateAuditRow],
) -> None:
    """
    输出 Answerable / Unanswerable
    Top1 Rerank Score 的简单统计。

    当前只做描述性统计：

        min
        median
        mean
        max

    不自动搜索最佳 threshold。

    原因：

    V3 是 Final Benchmark。

    如果脚本直接根据 V3
    搜索 optimal threshold，
    就会形成 Evaluation Leakage。
    """

    positive_scores = [
        row.top_scores[0]
        for row in rows
        if (
            row.expected_answerable
            and row.top_scores
        )
    ]

    negative_scores = [
        row.top_scores[0]
        for row in rows
        if (
            not row.expected_answerable
            and row.top_scores
        )
    ]

    print()
    print("=" * 120)
    print(
        "Top1 Rerank Score Distribution"
    )
    print("=" * 120)

    print(
        "Current threshold:",
        f"{DEMO_COARSE_RELEVANCE_THRESHOLD:.4f}",
    )

    print()

    print(
        "Answerable positives:"
    )

    if positive_scores:
        print(
            "  count:",
            len(positive_scores),
        )

        print(
            "  min:",
            f"{min(positive_scores):.4f}",
        )

        print(
            "  median:",
            f"{median(positive_scores):.4f}",
        )

        print(
            "  mean:",
            f"{mean(positive_scores):.4f}",
        )

        print(
            "  max:",
            f"{max(positive_scores):.4f}",
        )

    print()

    print(
        "Unanswerable negatives:"
    )

    if negative_scores:
        print(
            "  count:",
            len(negative_scores),
        )

        print(
            "  min:",
            f"{min(negative_scores):.4f}",
        )

        print(
            "  median:",
            f"{median(negative_scores):.4f}",
        )

        print(
            "  mean:",
            f"{mean(negative_scores):.4f}",
        )

        print(
            "  max:",
            f"{max(negative_scores):.4f}",
        )

    # --------------------------------------------------
    # 这里额外输出一个非常关键的诊断值：
    #
    # 最低 Positive
    # vs
    # 最高 Negative
    #
    # 如果：
    #
    # min_positive > max_negative
    #
    # 说明这批样本在 score 上存在 clean gap。
    #
    # 如果：
    #
    # min_positive <= max_negative
    #
    # 说明 Positive / Negative 分布已经重叠，
    # 不存在一个简单全局阈值
    # 可以在当前 Dataset 上完美区分两者。
    #
    # 注意：
    # 这仍然只是诊断，
    # 不是自动调参。
    # --------------------------------------------------

    if (
        positive_scores
        and negative_scores
    ):
        min_positive = min(
            positive_scores
        )

        max_negative = max(
            negative_scores
        )

        print()

        print(
            "Lowest positive Top1:",
            f"{min_positive:.4f}",
        )

        print(
            "Highest negative Top1:",
            f"{max_negative:.4f}",
        )

        print(
            "Positive / Negative overlap:",
            (
                "YES"
                if min_positive
                <= max_negative
                else "NO"
            ),
        )


def print_gate_confusion_matrix(
    rows: list[GateAuditRow],
) -> None:
    """
    只针对当前 Coarse Gate
    计算一个离线 confusion matrix。

    注意：

    这不是 Full-RAG 的最终 TP/TN/FP/FN。

    它只回答：

        仅凭当前 -3.0 Top1 Score Gate，
        会把多少 Answerable / Unanswerable
        Query 放行或拒绝？
    """

    true_positive = sum(
        1
        for row in rows
        if (
            row.expected_answerable
            and row.gate_passed
        )
    )

    true_negative = sum(
        1
        for row in rows
        if (
            not row.expected_answerable
            and not row.gate_passed
        )
    )

    false_positive = sum(
        1
        for row in rows
        if (
            not row.expected_answerable
            and row.gate_passed
        )
    )

    false_negative = sum(
        1
        for row in rows
        if (
            row.expected_answerable
            and not row.gate_passed
        )
    )

    print()
    print("=" * 120)
    print(
        "Current Coarse Gate Confusion Matrix"
    )
    print("=" * 120)

    print(
        "TP:",
        true_positive,
    )

    print(
        "TN:",
        true_negative,
    )

    print(
        "FP:",
        false_positive,
    )

    print(
        "FN:",
        false_negative,
    )

    print()

    print(
        "注意：这里评估的是 Gate 本身，"
        "不是最终 LLM Answer / Refusal Decision。"
    )


def main() -> None:
    """
    运行 Final V3 Coarse Evidence Gate Audit。

    目标：

    1. 验证 Full-RAG V3 的 9 个 FN
       是否确实来自 -3.0 Gate；

    2. 观察 Answerable 与 Unanswerable
       Query 的 Top1 Rerank Score 分布；

    3. 判断当前问题更像：

           threshold calibration

       还是：

           single raw-score gate
           architecture limitation

    本脚本：

        不修改 threshold
        不修改 Gold
        不调用 LLM
        不产生 SiliconFlow API 成本
    """

    print("=" * 120)
    print(
        "Final V3 Coarse Evidence Gate Audit"
    )
    print("=" * 120)

    print(
        "Dataset:",
        EVAL_PATH,
    )

    print(
        "Corpus:",
        CHUNKS_PATH,
    )

    print(
        "Current threshold:",
        DEMO_COARSE_RELEVANCE_THRESHOLD,
    )

    print(
        "LLM calls:",
        "DISABLED",
    )

    rows = run_gate_audit()

    print_gate_confusion_matrix(
        rows
    )

    print_score_distribution(
        rows
    )

    print_false_negatives(
        rows
    )

    print_unanswerable_cases(
        rows
    )

    print_all_cases(
        rows
    )

    print()
    print("=" * 120)

    print(
        "⚠ 本脚本仅做 Score Audit，"
        "不根据 V3 自动搜索或修改 threshold。"
    )

    print(
        "⚠ V3 仍然保持 Final Benchmark 身份，"
        "避免 Evaluation Leakage。"
    )


if __name__ == "__main__":
    main()