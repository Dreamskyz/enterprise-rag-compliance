"""运行 Prompt v2 完整 RAG Answer / Refusal Evaluation。"""

import argparse
from pathlib import Path

from enterprise_rag.evaluation.answer_metrics import (
    AnswerEvalCaseResult,
    aggregate_answer_metrics,
    citation_precision_for_case,
    citation_recall_for_case,
)
from enterprise_rag.evaluation.answer_result_store import (
    read_answer_eval_results_jsonl,
    write_answer_eval_results_jsonl,
)
from enterprise_rag.evaluation.answer_runner import (
    AnswerEvalRunResult,
    run_answer_evaluation,
)
from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
)
from enterprise_rag.runtime.builder import (
    build_rag_runtime,
)


# ==========================================================
# Evaluation Dataset Registry。
#
# V1:
#     冻结的历史 Seed Regression Benchmark。
#
# V2:
#     Corpus V2 Capability Benchmark。
#
# V3:
#     Final Corpus Capability Benchmark。
#
# 当前 V3 对应：
#
#     28 documents
#     835 KnowledgeChunks
#     46 evaluation cases
#
# Dataset 通过 CLI 显式选择：
#
#     --dataset v1
#     --dataset v2
#     --dataset v3
# ==========================================================

EVAL_PATHS: dict[str, Path] = {
    "v1": Path(
        "data/eval/retrieval_eval_v1.jsonl"
    ),
    "v2": Path(
        "data/eval/retrieval_eval_v2.jsonl"
    ),
    "v3": Path(
        "data/eval/retrieval_eval_v3.jsonl"
    ),
}


# ==========================================================
# 当前真实 KnowledgeChunk Corpus。
# ==========================================================

CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


# ==========================================================
# Snapshot Output Directory。
# ==========================================================

RESULTS_DIR = Path(
    "data/eval/results"
)


def parse_args() -> argparse.Namespace:
    """
    解析 Full-RAG Evaluation 命令行参数。

    正常启动：

        python scripts/run_answer_eval.py \
            --dataset v3 \
            --run-id 003

    从中断的 Partial Snapshot 恢复：

        python scripts/run_answer_eval.py \
            --dataset v3 \
            --run-id 003 \
            --resume

    ------------------------------------------------------
    --dataset
    ------------------------------------------------------

    指定 Evaluation Dataset。

    ------------------------------------------------------
    --run-id
    ------------------------------------------------------

    指定本次实验编号。

    例如：

        001
        002
        003

    最终形成：

        answer_eval_v3_run_003.jsonl

    ------------------------------------------------------
    --resume
    ------------------------------------------------------

    只有显式提供该参数，
    才允许读取对应：

        .partial.jsonl

    并从断点继续。

    不显式提供 --resume 时，
    发现 Partial Snapshot 会直接拒绝运行。

    这样避免误把一次中断实验
    当成全新的实验覆盖掉。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Prompt v2 full-RAG "
            "answer/refusal evaluation "
            "with case-level checkpointing."
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
            "v1=frozen seed regression benchmark, "
            "v2=Corpus V2 capability benchmark, "
            "v3=final capability benchmark. "
            "Default: v1."
        ),
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Snapshot run identifier, "
            "for example: 001, 002, 003. "
            "Existing final snapshots "
            "will never be overwritten."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run from "
            "its .partial.jsonl checkpoint. "
            "The partial snapshot must be "
            "a valid continuous prefix "
            "of the selected dataset."
        ),
    )

    return parser.parse_args()


def build_output_path(
    dataset_version: str,
    run_id: str,
) -> Path:
    """
    构建最终正式 Snapshot 路径。

    例如：

        dataset = v3
        run_id = 003

    得到：

        data/eval/results/
        answer_eval_v3_run_003.jsonl
    """

    return RESULTS_DIR / (
        f"answer_eval_{dataset_version}"
        f"_run_{run_id}.jsonl"
    )


def build_partial_output_path(
    dataset_version: str,
    run_id: str,
) -> Path:
    """
    构建运行中的 Partial Snapshot 路径。

    例如：

        data/eval/results/
        answer_eval_v3_run_003.partial.jsonl

    Partial Snapshot：

        可以用于断点恢复；

    Final Snapshot：

        才表示完整实验已经完成。
    """

    return RESULTS_DIR / (
        f"answer_eval_{dataset_version}"
        f"_run_{run_id}.partial.jsonl"
    )


def validate_run_configuration(
    *,
    eval_path: Path,
    output_path: Path,
    partial_output_path: Path,
    resume: bool,
) -> None:
    """
    在昂贵推理开始之前
    做实验配置的 fail-fast 检查。

    ------------------------------------------------------
    Dataset
    ------------------------------------------------------

    必须存在。

    ------------------------------------------------------
    Final Snapshot
    ------------------------------------------------------

    只要 Final Snapshot 已存在：

        永远禁止重新运行同一个 run-id。

    即使提供 --resume 也不允许。

    因为 Final Snapshot 表示：
        这次实验已经完成并被冻结。

    ------------------------------------------------------
    Partial Snapshot
    ------------------------------------------------------

    resume=False：

        Partial 不允许存在。

    resume=True：

        Partial 必须存在。

    这样不会出现：

        用户以为自己在恢复，
        实际却重新从 R001 开始；

    也不会出现：

        用户以为自己在新跑，
        实际偷偷复用旧结果。
    """

    if not eval_path.exists():
        raise FileNotFoundError(
            "Evaluation Dataset 不存在："
            f"{eval_path}"
        )

    if output_path.exists():
        raise FileExistsError(
            "Evaluation Final Snapshot 已存在，"
            "为保护历史实验结果，"
            "不允许覆盖或继续写入："
            f"{output_path}"
        )

    if resume:
        if not partial_output_path.exists():
            raise FileNotFoundError(
                "指定了 --resume，"
                "但没有找到 Partial Snapshot："
                f"{partial_output_path}"
            )

    else:
        if partial_output_path.exists():
            raise FileExistsError(
                "发现已有 Partial Snapshot："
                f"{partial_output_path}\n"
                "如果这是一次中断实验，"
                "请显式使用 --resume；"
                "如果想开始新的实验，"
                "请使用新的 --run-id。"
            )


def validate_partial_prefix(
    *,
    cases: list[RetrievalEvalCase],
    completed_results: list[
        AnswerEvalCaseResult
    ],
) -> None:
    """
    验证 Partial Snapshot
    是否是当前 Dataset 的“连续前缀”。

    例如：

    Dataset：

        R001
        R002
        R003
        R004

    合法 Partial：

        R001
        R002

    非法 Partial：

        R001
        R003

    ------------------------------------------------------
    为什么不能只检查 query_id 存不存在？
    ------------------------------------------------------

    因为 Resume 的目标不是：

        任意补洞。

    而是：

        从一个确定的连续断点继续。

    这样：

        Snapshot 顺序稳定；
        运行历史容易解释；
        不会重复调用 LLM；
        不会产生乱序 Result。
    """

    if len(completed_results) >= len(cases):
        raise ValueError(
            "Partial Snapshot 的 Case 数量 "
            "已经达到或超过 Dataset 数量，"
            "它不应该继续作为 Partial Snapshot 使用。"
        )

    for index, completed in enumerate(
        completed_results
    ):
        expected_case = cases[index]

        # --------------------------------------------------
        # 1. Query ID 必须一致。
        # --------------------------------------------------

        if (
            completed.query_id
            != expected_case.query_id
        ):
            raise ValueError(
                "Partial Snapshot 不是当前 "
                "Dataset 的连续前缀："
                f"第 {index + 1} 条 "
                f"Snapshot={completed.query_id}，"
                f"Dataset={expected_case.query_id}"
            )

        # --------------------------------------------------
        # 2. Query 文本必须一致。
        #
        # 防止 Dataset 在实验中途被偷偷修改。
        # --------------------------------------------------

        if (
            completed.query
            != expected_case.query
        ):
            raise ValueError(
                f"{completed.query_id} "
                "的 Query 与当前 Dataset 不一致。"
            )

        # --------------------------------------------------
        # 3. Category 必须一致。
        # --------------------------------------------------

        if (
            completed.category
            != expected_case.category
        ):
            raise ValueError(
                f"{completed.query_id} "
                "的 Category 与当前 Dataset 不一致。"
            )

        # --------------------------------------------------
        # 4. Answerability Gold 必须一致。
        # --------------------------------------------------

        if (
            completed.expected_answerable
            != expected_case.answerable
        ):
            raise ValueError(
                f"{completed.query_id} "
                "的 Answerability Gold "
                "与当前 Dataset 不一致。"
            )

        # --------------------------------------------------
        # 5. Retrieval Gold 必须一致。
        # --------------------------------------------------

        if (
            completed.gold_chunk_ids
            != expected_case.gold_chunk_ids
        ):
            raise ValueError(
                f"{completed.query_id} "
                "的 Retrieval Gold "
                "与当前 Dataset 不一致。"
            )

        # --------------------------------------------------
        # 6. Citation Gold 必须一致。
        # --------------------------------------------------

        if (
            completed.citation_gold_chunk_ids
            != expected_case.citation_gold_chunk_ids
        ):
            raise ValueError(
                f"{completed.query_id} "
                "的 Citation Gold "
                "与当前 Dataset 不一致。"
            )

        # --------------------------------------------------
        # 7. Strict Citation Annotation 必须一致。
        # --------------------------------------------------

        if (
            completed.strict_citation_eval
            != expected_case.strict_citation_eval
        ):
            raise ValueError(
                f"{completed.query_id} "
                "的 strict_citation_eval "
                "与当前 Dataset 不一致。"
            )


def build_aggregate_run_result(
    case_results: list[
        AnswerEvalCaseResult
    ],
) -> AnswerEvalRunResult:
    """
    基于已经完成的全部 Case Result
    重建最终 Aggregate Run Result。

    为什么不能直接使用最后一次
    run_answer_evaluation(cases=[case]) 的结果？

    因为那个 Result 只包含单个 Case。

    Resume 场景下我们真正需要的是：

        已完成 Partial
        +
        本次继续完成的新 Case

    的全量指标。

    ------------------------------------------------------
    Timing
    ------------------------------------------------------

    每个 AnswerEvalCaseResult
    已经保存单条 Query 的 latency_ms。

    因此可以跨进程恢复：

        total =
            sum(case.latency_ms)

        mean =
            total / case_count

    这比单纯使用当前进程的 perf_counter
    更适合 Resume 场景。

    注意：

    这里仍然只是 Full-RAG Evaluation Timing，
    不是正式 API Benchmark。
    """

    if not case_results:
        raise ValueError(
            "case_results 不能为空"
        )

    metrics = aggregate_answer_metrics(
        case_results
    )

    total_latency_ms = sum(
        result.latency_ms
        for result in case_results
    )

    return AnswerEvalRunResult(
        metrics=metrics,
        case_results=tuple(
            case_results
        ),
        total_latency_ms=(
            total_latency_ms
        ),
        mean_latency_ms=(
            total_latency_ms
            / len(case_results)
        ),
    )


def run_with_checkpointing(
    *,
    cases: list[
        RetrievalEvalCase
    ],
    query_service,
    partial_output_path: Path,
    completed_results: list[
        AnswerEvalCaseResult
    ],
) -> list[
    AnswerEvalCaseResult
]:
    """
    Case-Level Checkpoint Evaluation。

    每完成一个 Case：

        1. 得到 AnswerEvalCaseResult；
        2. 加入 accumulated results；
        3. 立即覆盖写入完整 Partial Snapshot。

    ------------------------------------------------------
    为什么不是 append 单行？
    ------------------------------------------------------

    当前已有：

        write_answer_eval_results_jsonl()

    是正式 Persistence Contract。

    我们继续复用它。

    每次虽然会重新写整个 Partial 文件，
    但 V3 只有 46 条结果，
    文件规模很小。

    相比 LLM 调用成本，
    这点磁盘 IO 可以忽略。

    好处是：

        不新增第二套 append protocol；
        Snapshot 始终可以被现有 Reader 完整校验。
    """

    accumulated_results = list(
        completed_results
    )

    start_index = len(
        accumulated_results
    )

    total_case_count = len(
        cases
    )

    for dataset_index in range(
        start_index,
        total_case_count,
    ):
        case = cases[
            dataset_index
        ]

        print()
        print("-" * 100)

        print(
            f"Checkpoint Case "
            f"[{dataset_index + 1}/"
            f"{total_case_count}] "
            f"{case.query_id}"
        )

        print(
            "Role:",
            case.role.value,
        )

        # --------------------------------------------------
        # 只执行当前一个 Case。
        #
        # Runtime / Retriever / LLM Service
        # 并不会重新初始化。
        #
        # query_service 是同一个已初始化对象。
        # --------------------------------------------------

        single_case_result = (
            run_answer_evaluation(
                cases=[
                    case
                ],
                query_service=(
                    query_service
                ),
            )
        )

        current_result = (
            single_case_result
            .case_results[0]
        )

        accumulated_results.append(
            current_result
        )

        # --------------------------------------------------
        # 当前 Case 一旦成功完成，
        # 第一时间写 Partial Snapshot。
        #
        # 如果下一条 Query 崩溃，
        # 当前以及此前全部成功结果仍然保留。
        # --------------------------------------------------

        write_answer_eval_results_jsonl(
            results=tuple(
                accumulated_results
            ),
            output_path=(
                partial_output_path
            ),
        )

        print(
            "Checkpoint saved:",
            f"{len(accumulated_results)}"
            f"/{total_case_count}",
        )

        print(
            "Partial snapshot:",
            partial_output_path,
        )

    return accumulated_results


def print_decision_failures(
    case_results,
) -> None:
    """
    打印 Answer / Refusal Decision Failure。

    也就是：

        Gold Answerable
        !=
        System Answerable
    """

    failures = [
        result
        for result in case_results
        if (
            result.expected_answerable
            != result.actual_answerable
        )
    ]

    print()
    print("=" * 100)
    print(
        "Answer / Refusal Failure Cases"
    )
    print("=" * 100)

    if not failures:
        print(
            "No decision failures."
        )
        return

    for result in failures:
        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Category:",
            result.category.value,
        )

        print(
            "Expected answerable:",
            result.expected_answerable,
        )

        print(
            "Actual answerable:",
            result.actual_answerable,
        )

        print(
            "Gate reason:",
            result.gate_reason,
        )

        print(
            "Retrieval Gold:",
            result.gold_chunk_ids,
        )

        print(
            "Citation Gold:",
            result.citation_gold_chunk_ids,
        )

        print(
            "Actual Citations:",
            result.cited_chunk_ids,
        )

        print(
            "Reason:",
            result.reason,
        )


def print_all_citation_mismatches(
    case_results,
) -> None:
    """
    打印所有 Citation Mismatch。

    Citation Metrics 必须使用：

        citation_gold_chunk_ids

    而不是 Retrieval Gold。
    """

    print()
    print("=" * 100)
    print(
        "All Citation Mismatch Cases"
    )
    print("=" * 100)

    mismatch_count = 0

    for result in case_results:
        if not result.expected_answerable:
            continue

        if not result.actual_answerable:
            continue

        precision = (
            citation_precision_for_case(
                cited_chunk_ids=(
                    result.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    result
                    .citation_gold_chunk_ids
                ),
            )
        )

        recall = (
            citation_recall_for_case(
                cited_chunk_ids=(
                    result.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    result
                    .citation_gold_chunk_ids
                ),
            )
        )

        if (
            precision == 1.0
            and recall == 1.0
        ):
            continue

        mismatch_count += 1

        citation_gold_set = set(
            result.citation_gold_chunk_ids
        )

        cited_set = set(
            result.cited_chunk_ids
        )

        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Strict:",
            result.strict_citation_eval,
        )

        print(
            "Citation Gold:",
            result.citation_gold_chunk_ids,
        )

        print(
            "Actual Citations:",
            result.cited_chunk_ids,
        )

        print(
            "Extra:",
            tuple(
                sorted(
                    cited_set
                    - citation_gold_set
                )
            ),
        )

        print(
            "Missing:",
            tuple(
                sorted(
                    citation_gold_set
                    - cited_set
                )
            ),
        )

        print(
            "Precision:",
            f"{precision:.4f}",
        )

        print(
            "Recall:",
            f"{recall:.4f}",
        )

        print(
            "Reason:",
            result.reason,
        )

    if mismatch_count == 0:
        print(
            "No citation mismatches."
        )


def print_strict_citation_mismatches(
    case_results,
) -> None:
    """
    只打印 Strict Citation Failure。
    """

    print()
    print("=" * 100)
    print(
        "Strict Citation Mismatch Cases"
    )
    print("=" * 100)

    mismatch_count = 0

    for result in case_results:
        if not result.expected_answerable:
            continue

        if not result.actual_answerable:
            continue

        if not result.strict_citation_eval:
            continue

        precision = (
            citation_precision_for_case(
                cited_chunk_ids=(
                    result.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    result
                    .citation_gold_chunk_ids
                ),
            )
        )

        recall = (
            citation_recall_for_case(
                cited_chunk_ids=(
                    result.cited_chunk_ids
                ),
                gold_chunk_ids=(
                    result
                    .citation_gold_chunk_ids
                ),
            )
        )

        if (
            precision == 1.0
            and recall == 1.0
        ):
            continue

        mismatch_count += 1

        citation_gold_set = set(
            result.citation_gold_chunk_ids
        )

        cited_set = set(
            result.cited_chunk_ids
        )

        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Citation Gold:",
            result.citation_gold_chunk_ids,
        )

        print(
            "Actual Citations:",
            result.cited_chunk_ids,
        )

        print(
            "Extra:",
            tuple(
                sorted(
                    cited_set
                    - citation_gold_set
                )
            ),
        )

        print(
            "Missing:",
            tuple(
                sorted(
                    citation_gold_set
                    - cited_set
                )
            ),
        )

        print(
            "Precision:",
            f"{precision:.4f}",
        )

        print(
            "Recall:",
            f"{recall:.4f}",
        )

        print(
            "Answer:",
            result.answer,
        )

        print(
            "Reason:",
            result.reason,
        )

    if mismatch_count == 0:
        print(
            "No strict citation mismatches."
        )


def print_evidence_id_leaks(
    case_results,
) -> None:
    """
    检查最终 Answer 正文
    是否泄露内部 Evidence ID。
    """

    print()
    print("=" * 100)
    print(
        "Answer Evidence-ID Leak Check"
    )
    print("=" * 100)

    leak_count = 0

    evidence_tokens = tuple(
        f"E{index}"
        for index in range(
            1,
            21,
        )
    )

    for result in case_results:
        if not result.answer:
            continue

        leaked_ids = tuple(
            evidence_id
            for evidence_id
            in evidence_tokens
            if evidence_id
            in result.answer
        )

        if not leaked_ids:
            continue

        leak_count += 1

        print()

        print(
            result.query_id,
            "|",
            result.query,
        )

        print(
            "Leaked Evidence IDs:",
            leaked_ids,
        )

        print(
            "Answer:",
            result.answer,
        )

    if leak_count == 0:
        print(
            "No Evidence-ID leaks."
        )


def print_summary(
    result,
) -> None:
    """
    打印 Prompt v2
    Full-RAG Evaluation Summary。
    """

    metrics = result.metrics

    print()
    print("=" * 100)
    print(
        "Prompt v2 Full-RAG "
        "Evaluation Summary"
    )
    print("=" * 100)

    print(
        "Cases:",
        metrics.case_count,
    )

    print(
        "Answerable:",
        metrics.answerable_count,
    )

    print(
        "Unanswerable:",
        metrics.unanswerable_count,
    )

    print()

    print(
        "TP:",
        metrics.true_positive,
    )

    print(
        "TN:",
        metrics.true_negative,
    )

    print(
        "FP:",
        metrics.false_positive,
    )

    print(
        "FN:",
        metrics.false_negative,
    )

    print()

    print(
        "Overall Decision Accuracy:",
        (
            f"{metrics.overall_decision_accuracy:.4f}"
        ),
    )

    print(
        "Answerable Accuracy:",
        f"{metrics.answerable_accuracy:.4f}",
    )

    print(
        "Refusal Accuracy:",
        f"{metrics.refusal_accuracy:.4f}",
    )

    print(
        "Hard Negative Refusal Accuracy:",
        (
            f"{metrics.hard_negative_refusal_accuracy:.4f}"
        ),
    )

    print(
        "OOD Refusal Accuracy:",
        (
            f"{metrics.out_of_domain_refusal_accuracy:.4f}"
        ),
    )

    print()
    print("-" * 100)
    print(
        "All-case Citation Metrics"
    )
    print("-" * 100)

    print(
        "Citation Cases:",
        metrics.citation_case_count,
    )

    print(
        "Citation Precision:",
        f"{metrics.citation_precision:.4f}",
    )

    print(
        "Citation Recall:",
        f"{metrics.citation_recall:.4f}",
    )

    print(
        "Citation Hit Rate:",
        f"{metrics.citation_hit_rate:.4f}",
    )

    print()
    print("-" * 100)
    print(
        "Strict Citation Metrics"
    )
    print("-" * 100)

    print(
        "Strict Citation Cases:",
        metrics.strict_citation_case_count,
    )

    print(
        "Strict Citation Precision:",
        (
            f"{metrics.strict_citation_precision:.4f}"
        ),
    )

    print(
        "Strict Citation Recall:",
        (
            f"{metrics.strict_citation_recall:.4f}"
        ),
    )

    print(
        "Strict Citation Hit Rate:",
        (
            f"{metrics.strict_citation_hit_rate:.4f}"
        ),
    )

    print()

    print(
        "Total Eval Time:",
        (
            f"{result.total_latency_ms / 1000.0:.2f}s"
        ),
    )

    print(
        "Mean End-to-End Time:",
        (
            f"{result.mean_latency_ms:.2f}"
            "ms/query"
        ),
    )


def main() -> None:
    """
    运行 Prompt v2 完整 Full-RAG Evaluation。

    当前脚本额外负责：

        Case-Level Checkpoint
        Partial Snapshot Validation
        Resume
        Final Snapshot Promotion

    这些属于：

        Evaluation Orchestration

    而不是生产 RAG 业务逻辑。
    """

    args = parse_args()

    dataset_version = str(
        args.dataset
    )

    run_id = str(
        args.run_id
    )

    resume = bool(
        args.resume
    )

    eval_path = EVAL_PATHS[
        dataset_version
    ]

    output_path = build_output_path(
        dataset_version=dataset_version,
        run_id=run_id,
    )

    partial_output_path = (
        build_partial_output_path(
            dataset_version=(
                dataset_version
            ),
            run_id=run_id,
        )
    )

    # ======================================================
    # 1. 昂贵推理之前：
    #    校验实验运行状态。
    # ======================================================

    validate_run_configuration(
        eval_path=eval_path,
        output_path=output_path,
        partial_output_path=(
            partial_output_path
        ),
        resume=resume,
    )

    print("=" * 100)
    print(
        "Prompt v2 Full RAG Evaluation"
    )
    print("=" * 100)

    print(
        "Dataset version:",
        dataset_version,
    )

    print(
        "Dataset path:",
        eval_path,
    )

    print(
        "Run ID:",
        run_id,
    )

    print(
        "Resume:",
        resume,
    )

    print(
        "Corpus path:",
        CHUNKS_PATH,
    )

    print(
        "Final snapshot:",
        output_path,
    )

    print(
        "Partial snapshot:",
        partial_output_path,
    )

    # ======================================================
    # 2. 读取冻结 Evaluation Dataset。
    # ======================================================

    cases = list(
        read_retrieval_eval_jsonl(
            eval_path
        )
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

    # ======================================================
    # 3. Resume 时读取并校验 Partial Snapshot。
    # ======================================================

    completed_results: list[
        AnswerEvalCaseResult
    ] = []

    if resume:
        print()

        print(
            "Loading partial snapshot..."
        )

        completed_results = (
            read_answer_eval_results_jsonl(
                partial_output_path
            )
        )

        validate_partial_prefix(
            cases=cases,
            completed_results=(
                completed_results
            ),
        )

        print(
            "Partial snapshot validated."
        )

        print(
            "Completed cases:",
            len(completed_results),
        )

        print(
            "Remaining cases:",
            (
                len(cases)
                - len(completed_results)
            ),
        )

        next_case = cases[
            len(completed_results)
        ]

        print(
            "Resume from:",
            next_case.query_id,
        )

    # ======================================================
    # 4. 初始化真实 RAG Runtime。
    #
    # 无论新跑还是 Resume，
    # Runtime 都只初始化一次。
    # ======================================================

    print()

    print(
        "Initializing RAG runtime..."
    )

    runtime = build_rag_runtime(
        chunks_path=(
            CHUNKS_PATH
        )
    )

    print(
        "RAG runtime initialized."
    )

    # ======================================================
    # 5. Case-Level Evaluation + Checkpoint。
    # ======================================================

    print()

    print(
        "Running Prompt v2 "
        "Full-RAG evaluation "
        "with case-level checkpointing..."
    )

    accumulated_results = (
        run_with_checkpointing(
            cases=cases,
            query_service=(
                runtime.query_service
            ),
            partial_output_path=(
                partial_output_path
            ),
            completed_results=(
                completed_results
            ),
        )
    )

    # ======================================================
    # 6. Final Completeness Check。
    #
    # 在 Partial → Final Promotion 前，
    # 再验证一次：
    #
    #     数量
    #     顺序
    #     Dataset Annotation
    #
    # 必须全部一致。
    # ======================================================

    if (
        len(accumulated_results)
        != len(cases)
    ):
        raise RuntimeError(
            "Evaluation 未完整结束："
            f"completed="
            f"{len(accumulated_results)}, "
            f"expected={len(cases)}"
        )

    # validate_partial_prefix()
    # 只允许 completed < cases，
    # 所以 Final 完整结果这里单独逐条校验。

    for index, result_item in enumerate(
        accumulated_results
    ):
        case = cases[index]

        if (
            result_item.query_id
            != case.query_id
        ):
            raise RuntimeError(
                "Final Result 顺序与 "
                "Dataset 不一致："
                f"index={index}"
            )

        if (
            result_item.query
            != case.query
        ):
            raise RuntimeError(
                f"{case.query_id} "
                "Final Query 与 Dataset 不一致"
            )

    # ======================================================
    # 7. 生成最终 Aggregate Metrics。
    # ======================================================

    result = build_aggregate_run_result(
        accumulated_results
    )

    # ======================================================
    # 8. Promote Partial → Final Snapshot。
    #
    # 这里仍然通过正式 Writer
    # 重新写 Final Snapshot。
    #
    # 不直接 rename Partial，
    # 是为了确保：
    #
    #     Final Snapshot
    #
    # 一定来自当前内存中已经完成
    # 完整性检查的 Results。
    # ======================================================

    write_answer_eval_results_jsonl(
        results=(
            result.case_results
        ),
        output_path=(
            output_path
        ),
    )

    print()

    print(
        "Final snapshot saved:"
    )

    print(
        output_path
    )

    # ======================================================
    # 9. Final 写入成功后，
    #    才删除 Partial。
    #
    # 如果 Final 写入失败，
    # Partial 仍然保留，
    # 还能恢复。
    # ======================================================

    if partial_output_path.exists():
        partial_output_path.unlink()

    print(
        "Partial snapshot removed."
    )

    # ======================================================
    # 10. Report。
    # ======================================================

    print_summary(
        result
    )

    print_decision_failures(
        result.case_results
    )

    print_all_citation_mismatches(
        result.case_results
    )

    print_strict_citation_mismatches(
        result.case_results
    )

    print_evidence_id_leaks(
        result.case_results
    )

    # ======================================================
    # 11. Experiment Boundary。
    # ======================================================

    print()
    print("=" * 100)

    print(
        "⚠ 当前 Full-RAG Evaluation "
        "使用 Dataset：",
        dataset_version,
    )

    print(
        "⚠ 当前 Snapshot Run ID：",
        run_id,
    )

    print(
        "⚠ Evaluation 使用 Case-Level "
        "Partial Snapshot；"
        "中断后必须显式使用 --resume。"
    )

    print(
        "⚠ Final Snapshot 使用独立文件保存，"
        "不允许覆盖已有实验结果。"
    )

    print(
        "⚠ End-to-End Time 为各 Case "
        "已记录 latency_ms 的累计值，"
        "支持跨进程 Resume；"
        "不作为正式 API 性能 Benchmark。"
    )

    print(
        "⚠ Retrieval Gold 与 Citation Gold "
        "保持解耦；"
        "Citation Metrics 使用 "
        "citation_gold_chunk_ids。"
    )

    if dataset_version == "v3":
        print(
            "⚠ V3 是 Final Capability Benchmark；"
            "不根据当前 Benchmark 的 "
            "Answer / Refusal / Citation 结果"
            "反向调整 Retrieval Gold、"
            "Retriever、RRF 或 Reranker。"
        )


if __name__ == "__main__":
    main()