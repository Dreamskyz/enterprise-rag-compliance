"""Full-RAG Evaluation Result JSONL 持久化。"""

import json
from pathlib import Path

from enterprise_rag.evaluation.answer_metrics import (
    AnswerEvalCaseResult,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCategory,
)


def write_answer_eval_results_jsonl(
    *,
    results: tuple[
        AnswerEvalCaseResult,
        ...,
    ],
    output_path: Path,
) -> None:
    """
    将 Full-RAG Evaluation 原始结果保存为 JSONL。

    一行对应一个 Query。

    Snapshot 保存：

        Query
        Retrieval Gold
        Citation Gold
        Strict Citation Annotation
        Answer / Refusal
        Citation
        Gate
        Rerank Score
        Latency

    不保存任何 API Key / Token / Header。
    """

    if not results:
        raise ValueError(
            "results 不能为空"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        for result in results:
            data = {
                "query_id": (
                    result.query_id
                ),
                "query": (
                    result.query
                ),
                "category": (
                    result.category.value
                ),
                "expected_answerable": (
                    result.expected_answerable
                ),

                # Retrieval Gold。
                "gold_chunk_ids": list(
                    result.gold_chunk_ids
                ),

                # Citation Gold。
                "citation_gold_chunk_ids": list(
                    result.citation_gold_chunk_ids
                ),

                # Strict Citation Annotation。
                "strict_citation_eval": (
                    result.strict_citation_eval
                ),

                "actual_answerable": (
                    result.actual_answerable
                ),
                "answer": (
                    result.answer
                ),
                "cited_chunk_ids": list(
                    result.cited_chunk_ids
                ),
                "gate_reason": (
                    result.gate_reason
                ),
                "reason": (
                    result.reason
                ),
                "retrieval_count": (
                    result.retrieval_count
                ),
                "top_rerank_score": (
                    result.top_rerank_score
                ),
                "latency_ms": (
                    result.latency_ms
                ),
            }

            file.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )


def read_answer_eval_results_jsonl(
    input_path: Path,
) -> list[
    AnswerEvalCaseResult
]:
    """
    从 JSONL 读取 Full-RAG Evaluation Snapshot。

    同时兼容旧版 Snapshot。

    旧版 Snapshot 没有：

        citation_gold_chunk_ids
        strict_citation_eval

    读取旧版时：

        citation_gold_chunk_ids
            暂时回退到 gold_chunk_ids

        strict_citation_eval
            暂时设为 False

    后续再通过 Snapshot Enrichment
    注入新的人工 Annotation。
    """

    if not input_path.exists():
        raise FileNotFoundError(
            "Answer Evaluation Result "
            f"不存在：{input_path}"
        )

    results: list[
        AnswerEvalCaseResult
    ] = []

    seen_query_ids: set[
        str
    ] = set()

    with input_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                data = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Answer Evaluation JSONL "
                    f"第 {line_number} 行不是合法 JSON"
                ) from exc

            # ==================================================
            # 1. 基础字段。
            # ==================================================

            query_id = str(
                data.get(
                    "query_id",
                    "",
                )
            ).strip()

            query = str(
                data.get(
                    "query",
                    "",
                )
            ).strip()

            category_raw = str(
                data.get(
                    "category",
                    "",
                )
            ).strip()

            expected_answerable = (
                data.get(
                    "expected_answerable"
                )
            )

            actual_answerable = (
                data.get(
                    "actual_answerable"
                )
            )

            answer_raw = data.get(
                "answer"
            )

            gold_raw = data.get(
                "gold_chunk_ids"
            )

            cited_raw = data.get(
                "cited_chunk_ids"
            )

            gate_reason = str(
                data.get(
                    "gate_reason",
                    "",
                )
            ).strip()

            reason = str(
                data.get(
                    "reason",
                    "",
                )
            ).strip()

            retrieval_count = data.get(
                "retrieval_count"
            )

            top_rerank_score_raw = (
                data.get(
                    "top_rerank_score"
                )
            )

            latency_ms_raw = data.get(
                "latency_ms"
            )

            # ==================================================
            # 2. 基础 Validation。
            # ==================================================

            if not query_id:
                raise ValueError(
                    f"第 {line_number} 行 "
                    "query_id 不能为空"
                )

            if query_id in seen_query_ids:
                raise ValueError(
                    "发现重复 query_id："
                    f"{query_id}"
                )

            if not query:
                raise ValueError(
                    f"{query_id} query 不能为空"
                )

            if not isinstance(
                expected_answerable,
                bool,
            ):
                raise ValueError(
                    f"{query_id} "
                    "expected_answerable "
                    "必须是 bool"
                )

            if not isinstance(
                actual_answerable,
                bool,
            ):
                raise ValueError(
                    f"{query_id} "
                    "actual_answerable "
                    "必须是 bool"
                )

            if not isinstance(
                gold_raw,
                list,
            ):
                raise ValueError(
                    f"{query_id} "
                    "gold_chunk_ids 必须是数组"
                )

            if not isinstance(
                cited_raw,
                list,
            ):
                raise ValueError(
                    f"{query_id} "
                    "cited_chunk_ids 必须是数组"
                )

            if (
                not isinstance(
                    retrieval_count,
                    int,
                )
                or retrieval_count < 0
            ):
                raise ValueError(
                    f"{query_id} "
                    "retrieval_count "
                    "必须是非负整数"
                )

            if not isinstance(
                latency_ms_raw,
                (int, float),
            ):
                raise ValueError(
                    f"{query_id} "
                    "latency_ms 必须是数字"
                )

            latency_ms = float(
                latency_ms_raw
            )

            if latency_ms < 0:
                raise ValueError(
                    f"{query_id} "
                    "latency_ms 不能为负数"
                )

            # ==================================================
            # 3. Category。
            # ==================================================

            try:
                category = (
                    RetrievalEvalCategory(
                        category_raw
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"{query_id} "
                    "category 非法："
                    f"{category_raw}"
                ) from exc

            # ==================================================
            # 4. Retrieval Gold。
            # ==================================================

            gold_chunk_ids = tuple(
                str(chunk_id).strip()
                for chunk_id in gold_raw
                if str(chunk_id).strip()
            )

            # ==================================================
            # 5. Citation Gold。
            #
            # 兼容旧 Snapshot。
            # ==================================================

            citation_gold_raw = data.get(
                "citation_gold_chunk_ids"
            )

            if citation_gold_raw is None:
                # 旧 Snapshot：
                #
                # 先使用 Retrieval Gold 作为临时值。
                #
                # 真正 Citation Gold 后面由
                # enrichment 注入。
                citation_gold_chunk_ids = (
                    gold_chunk_ids
                )

            else:
                if not isinstance(
                    citation_gold_raw,
                    list,
                ):
                    raise ValueError(
                        f"{query_id} "
                        "citation_gold_chunk_ids "
                        "必须是数组"
                    )

                citation_gold_chunk_ids = tuple(
                    str(chunk_id).strip()
                    for chunk_id
                    in citation_gold_raw
                    if str(chunk_id).strip()
                )

            # ==================================================
            # 6. Strict Citation Annotation。
            #
            # 同样兼容旧 Snapshot。
            # ==================================================

            strict_citation_raw = data.get(
                "strict_citation_eval"
            )

            if strict_citation_raw is None:
                strict_citation_eval = False
            else:
                if not isinstance(
                    strict_citation_raw,
                    bool,
                ):
                    raise ValueError(
                        f"{query_id} "
                        "strict_citation_eval "
                        "必须是 bool"
                    )

                strict_citation_eval = (
                    strict_citation_raw
                )

            # ==================================================
            # 7. Answer。
            # ==================================================

            if answer_raw is None:
                answer = None
            else:
                answer = str(
                    answer_raw
                ).strip()

            # ==================================================
            # 8. Actual Citations。
            # ==================================================

            cited_chunk_ids = tuple(
                str(chunk_id).strip()
                for chunk_id in cited_raw
                if str(chunk_id).strip()
            )

            # ==================================================
            # 9. Rerank Score。
            # ==================================================

            if top_rerank_score_raw is None:
                top_rerank_score = None

            elif isinstance(
                top_rerank_score_raw,
                (int, float),
            ):
                top_rerank_score = float(
                    top_rerank_score_raw
                )

            else:
                raise ValueError(
                    f"{query_id} "
                    "top_rerank_score "
                    "必须是数字或 null"
                )

            # ==================================================
            # 10. Answer / Refusal 一致性检查。
            # ==================================================

            if (
                actual_answerable
                and not answer
            ):
                raise ValueError(
                    f"{query_id} "
                    "actual_answerable=true "
                    "但 answer 为空"
                )

            if (
                not actual_answerable
                and answer is not None
            ):
                raise ValueError(
                    f"{query_id} "
                    "actual_answerable=false "
                    "但 answer 非空"
                )

            seen_query_ids.add(
                query_id
            )

            results.append(
                AnswerEvalCaseResult(
                    query_id=(
                        query_id
                    ),
                    query=query,
                    category=category,
                    expected_answerable=(
                        expected_answerable
                    ),
                    gold_chunk_ids=(
                        gold_chunk_ids
                    ),
                    citation_gold_chunk_ids=(
                        citation_gold_chunk_ids
                    ),
                    strict_citation_eval=(
                        strict_citation_eval
                    ),
                    actual_answerable=(
                        actual_answerable
                    ),
                    answer=answer,
                    cited_chunk_ids=(
                        cited_chunk_ids
                    ),
                    gate_reason=(
                        gate_reason
                    ),
                    reason=reason,
                    retrieval_count=(
                        retrieval_count
                    ),
                    top_rerank_score=(
                        top_rerank_score
                    ),
                    latency_ms=(
                        latency_ms
                    ),
                )
            )

    if not results:
        raise ValueError(
            "Answer Evaluation Result 文件为空"
        )

    return results