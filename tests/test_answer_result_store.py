"""测试 Answer Evaluation Result JSONL 持久化。"""

from pathlib import Path

import pytest

from enterprise_rag.evaluation.answer_metrics import (
    AnswerEvalCaseResult,
)
from enterprise_rag.evaluation.answer_result_store import (
    read_answer_eval_results_jsonl,
    write_answer_eval_results_jsonl,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCategory,
)


def make_answerable_result() -> (
    AnswerEvalCaseResult
):
    """构造可回答 Result。"""

    return AnswerEvalCaseResult(
        query_id="R001",
        query="测试问题",
        category=(
            RetrievalEvalCategory.DIRECT
        ),
        expected_answerable=True,

        # Retrieval Gold。
        gold_chunk_ids=(
            "retrieval-gold-1",
        ),

        # Citation Gold 故意与 Retrieval Gold 不同，
        # 用于验证 Snapshot 可以独立保存两类 Gold。
        citation_gold_chunk_ids=(
            "citation-gold-1",
            "citation-gold-2",
        ),

        strict_citation_eval=True,

        actual_answerable=True,

        answer="测试回答",

        cited_chunk_ids=(
            "citation-gold-1",
        ),

        gate_reason="passed",

        reason="证据充分",

        retrieval_count=5,

        top_rerank_score=7.5,

        latency_ms=1234.5,
    )


def make_refusal_result() -> (
    AnswerEvalCaseResult
):
    """构造拒答 Result。"""

    return AnswerEvalCaseResult(
        query_id="R002",
        query="不可回答问题",
        category=(
            RetrievalEvalCategory.HARD_NEGATIVE
        ),
        expected_answerable=False,

        gold_chunk_ids=(),

        citation_gold_chunk_ids=(),

        strict_citation_eval=False,

        actual_answerable=False,

        answer=None,

        cited_chunk_ids=(),

        gate_reason="passed",

        reason="证据不足",

        retrieval_count=5,

        top_rerank_score=4.5,

        latency_ms=987.6,
    )


def test_write_and_read_answer_eval_results(
    tmp_path: Path,
) -> None:
    """
    Citation-aware JSONL 应能够完整 Round-trip。
    """

    output_path = (
        tmp_path
        / "results"
        / "answer_eval.jsonl"
    )

    original_results = (
        make_answerable_result(),
        make_refusal_result(),
    )

    write_answer_eval_results_jsonl(
        results=original_results,
        output_path=output_path,
    )

    assert output_path.exists()

    loaded_results = (
        read_answer_eval_results_jsonl(
            output_path
        )
    )

    assert len(
        loaded_results
    ) == 2

    first = loaded_results[0]

    assert (
        first.query_id
        == "R001"
    )

    assert (
        first.answer
        == "测试回答"
    )

    # Retrieval Gold。
    assert (
        first.gold_chunk_ids
        == (
            "retrieval-gold-1",
        )
    )

    # Citation Gold。
    assert (
        first.citation_gold_chunk_ids
        == (
            "citation-gold-1",
            "citation-gold-2",
        )
    )

    assert (
        first.strict_citation_eval
        is True
    )

    assert (
        first.cited_chunk_ids
        == (
            "citation-gold-1",
        )
    )

    assert (
        first.top_rerank_score
        == pytest.approx(
            7.5
        )
    )

    assert (
        first.latency_ms
        == pytest.approx(
            1234.5
        )
    )

    second = loaded_results[1]

    assert (
        second.actual_answerable
        is False
    )

    assert (
        second.answer
        is None
    )

    assert (
        second.citation_gold_chunk_ids
        == ()
    )

    assert (
        second.strict_citation_eval
        is False
    )


def test_writer_rejects_empty_results(
    tmp_path: Path,
) -> None:
    """空 Result 不允许写入。"""

    with pytest.raises(
        ValueError,
        match="results",
    ):
        write_answer_eval_results_jsonl(
            results=(),
            output_path=(
                tmp_path
                / "empty.jsonl"
            ),
        )


def test_reader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """不存在文件应明确报错。"""

    with pytest.raises(
        FileNotFoundError
    ):
        read_answer_eval_results_jsonl(
            tmp_path
            / "missing.jsonl"
        )


def test_reader_rejects_duplicate_query_id(
    tmp_path: Path,
) -> None:
    """Snapshot 中 query_id 必须唯一。"""

    path = (
        tmp_path
        / "duplicate.jsonl"
    )

    line = (
        '{"query_id":"R001",'
        '"query":"测试",'
        '"category":"direct",'
        '"expected_answerable":true,'
        '"gold_chunk_ids":["A"],'
        '"citation_gold_chunk_ids":["A"],'
        '"strict_citation_eval":true,'
        '"actual_answerable":true,'
        '"answer":"回答",'
        '"cited_chunk_ids":["A"],'
        '"gate_reason":"passed",'
        '"reason":"ok",'
        '"retrieval_count":1,'
        '"top_rerank_score":1.0,'
        '"latency_ms":10.0}'
    )

    path.write_text(
        line
        + "\n"
        + line
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="重复 query_id",
    ):
        read_answer_eval_results_jsonl(
            path
        )


def test_reader_supports_old_snapshot(
    tmp_path: Path,
) -> None:
    """
    旧 Snapshot 没有：

        citation_gold_chunk_ids
        strict_citation_eval

    Reader 必须仍然能够读取。

    临时兼容规则：

        Citation Gold = Retrieval Gold
        strict = False

    后续再通过 Enrichment
    注入正式人工 Annotation。
    """

    path = (
        tmp_path
        / "old_snapshot.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"旧快照问题",'
            '"category":"direct",'
            '"expected_answerable":true,'
            '"gold_chunk_ids":["A"],'
            '"actual_answerable":true,'
            '"answer":"旧回答",'
            '"cited_chunk_ids":["A"],'
            '"gate_reason":"passed",'
            '"reason":"ok",'
            '"retrieval_count":5,'
            '"top_rerank_score":5.0,'
            '"latency_ms":100.0}\n'
        ),
        encoding="utf-8",
    )

    results = (
        read_answer_eval_results_jsonl(
            path
        )
    )

    assert len(results) == 1

    result = results[0]

    assert (
        result.gold_chunk_ids
        == ("A",)
    )

    # 旧 Snapshot 临时 fallback。
    assert (
        result.citation_gold_chunk_ids
        == ("A",)
    )

    assert (
        result.strict_citation_eval
        is False
    )