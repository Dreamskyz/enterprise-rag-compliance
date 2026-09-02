"""测试 Evaluation Dataset Loader。"""

from pathlib import Path

import pytest

from enterprise_rag.evaluation.dataset import (
    read_retrieval_eval_jsonl,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCategory,
)


def test_read_retrieval_eval_jsonl(
    tmp_path: Path,
) -> None:
    """
    合法 Citation-aware Dataset 应正确读取。
    """

    path = (
        tmp_path
        / "eval.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"测试问题",'
            '"gold_chunk_ids":["chunk-1"],'
            '"citation_gold_chunk_ids":["chunk-1","chunk-2"],'
            '"category":"direct",'
            '"answerable":true,'
            '"strict_citation_eval":true,'
            '"note":"测试"}\n'
        ),
        encoding="utf-8",
    )

    cases = (
        read_retrieval_eval_jsonl(
            path
        )
    )

    assert len(cases) == 1

    case = cases[0]

    assert case.query_id == "R001"

    assert (
        case.query
        == "测试问题"
    )

    # Retrieval Gold。
    assert (
        case.gold_chunk_ids
        == ("chunk-1",)
    )

    # Citation Gold 可以与 Retrieval Gold 不同。
    assert (
        case.citation_gold_chunk_ids
        == (
            "chunk-1",
            "chunk-2",
        )
    )

    assert (
        case.category
        == RetrievalEvalCategory.DIRECT
    )

    assert (
        case.answerable
        is True
    )

    assert (
        case.strict_citation_eval
        is True
    )

    assert case.note == "测试"


def test_answerable_requires_gold_chunks(
    tmp_path: Path,
) -> None:
    """
    answerable=true 时，
    Retrieval Gold 不能为空。

    注意：

    这里故意提供合法 Citation Gold，
    避免测试被 Citation Gold Validation
    提前截获。
    """

    path = (
        tmp_path
        / "eval.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"测试问题",'
            '"gold_chunk_ids":[],'
            '"citation_gold_chunk_ids":["chunk-1"],'
            '"category":"direct",'
            '"answerable":true,'
            '"strict_citation_eval":true,'
            '"note":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="gold_chunk_ids 不能为空",
    ):
        read_retrieval_eval_jsonl(
            path
        )


def test_unanswerable_requires_empty_gold_chunks(
    tmp_path: Path,
) -> None:
    """
    answerable=false 时，
    Retrieval Gold 必须为空。

    Citation Gold 这里保持为空，
    避免其它 Validation 干扰测试目标。
    """

    path = (
        tmp_path
        / "eval.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"不可回答问题",'
            '"gold_chunk_ids":["chunk-1"],'
            '"citation_gold_chunk_ids":[],'
            '"category":"hard_negative",'
            '"answerable":false,'
            '"strict_citation_eval":false,'
            '"note":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="gold_chunk_ids 必须为空",
    ):
        read_retrieval_eval_jsonl(
            path
        )


def test_rejects_duplicate_query_id(
    tmp_path: Path,
) -> None:
    """
    query_id 必须唯一。

    两行都必须先是完整合法的
    Citation-aware Dataset Row。

    否则 Loader 会在检测重复 ID 前
    因缺字段而提前失败。
    """

    path = (
        tmp_path
        / "eval.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"问题一",'
            '"gold_chunk_ids":["a"],'
            '"citation_gold_chunk_ids":["a"],'
            '"category":"direct",'
            '"answerable":true,'
            '"strict_citation_eval":true,'
            '"note":""}\n'
            '{"query_id":"R001",'
            '"query":"问题二",'
            '"gold_chunk_ids":["b"],'
            '"citation_gold_chunk_ids":["b"],'
            '"category":"direct",'
            '"answerable":true,'
            '"strict_citation_eval":true,'
            '"note":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="重复 query_id",
    ):
        read_retrieval_eval_jsonl(
            path
        )


def test_answerable_requires_citation_gold(
    tmp_path: Path,
) -> None:
    """
    answerable=true 时，
    Citation Gold 也不能为空。
    """

    path = (
        tmp_path
        / "eval.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"测试问题",'
            '"gold_chunk_ids":["chunk-1"],'
            '"citation_gold_chunk_ids":[],'
            '"category":"direct",'
            '"answerable":true,'
            '"strict_citation_eval":true,'
            '"note":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="citation_gold_chunk_ids",
    ):
        read_retrieval_eval_jsonl(
            path
        )


def test_unanswerable_requires_empty_citation_gold(
    tmp_path: Path,
) -> None:
    """
    不可回答问题不应该存在 Citation Gold。
    """

    path = (
        tmp_path
        / "eval.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"不可回答问题",'
            '"gold_chunk_ids":[],'
            '"citation_gold_chunk_ids":["chunk-1"],'
            '"category":"hard_negative",'
            '"answerable":false,'
            '"strict_citation_eval":false,'
            '"note":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="citation_gold_chunk_ids",
    ):
        read_retrieval_eval_jsonl(
            path
        )


def test_unanswerable_cannot_use_strict_citation_eval(
    tmp_path: Path,
) -> None:
    """
    不可回答问题没有 Citation，
    所以不能进入 Strict Citation Metrics。
    """

    path = (
        tmp_path
        / "eval.jsonl"
    )

    path.write_text(
        (
            '{"query_id":"R001",'
            '"query":"不可回答问题",'
            '"gold_chunk_ids":[],'
            '"citation_gold_chunk_ids":[],'
            '"category":"out_of_domain",'
            '"answerable":false,'
            '"strict_citation_eval":true,'
            '"note":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="strict_citation_eval",
    ):
        read_retrieval_eval_jsonl(
            path
        )