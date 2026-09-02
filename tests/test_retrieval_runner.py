"""测试统一 Retrieval Evaluation Runner。"""

from enterprise_rag.acl.models import (
    AccessContext,
    UserRole,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalCategory,
)
from enterprise_rag.evaluation.retrieval_runner import (
    RetrievalMethod,
    evaluate_retrieval_method,
)
from enterprise_rag.retrieval.models import (
    DenseSearchResult,
    RetrievalCandidate,
)


def make_candidate(
    chunk_id: str,
) -> RetrievalCandidate:
    """构造测试 Candidate。"""

    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="doc",
        title="测试法规",
        document_type="regulation",
        language="zh-CN",
        version="1",
        chapter_number="第一章",
        chapter_title="测试",
        article_number="第一条",
        content="测试正文",
        retrieval_text="测试检索文本",
        source_url=(
            "https://example.com"
        ),
        access_level="public",
        chunk_index=0,
        content_hash="hash",
    )


def make_dense_result(
    chunk_id: str,
    score: float = 1.0,
) -> DenseSearchResult:
    """构造测试 Dense Result。"""

    return DenseSearchResult(
        candidate=make_candidate(
            chunk_id
        ),
        score=score,
    )


def test_runner_only_evaluates_answerable_cases() -> None:
    """
    Unanswerable Query 不应进入 Recall / MRR。
    """

    cases = [
        RetrievalEvalCase(
            query_id="R001",
            query="可回答问题",
            gold_chunk_ids=(
                "gold",
            ),
            category=(
                RetrievalEvalCategory.DIRECT
            ),
            answerable=True,
            note="",
        ),
        RetrievalEvalCase(
            query_id="R002",
            query="不可回答问题",
            gold_chunk_ids=(),
            category=(
                RetrievalEvalCategory.OUT_OF_DOMAIN
            ),
            answerable=False,
            note="",
        ),
    ]

    called_queries: list[
        str
    ] = []

    def fake_retrieve(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        called_queries.append(
            query
        )

        assert (
            access_context.role
            == UserRole.GUEST
        )

        return [
            make_dense_result(
                "gold"
            )
        ][:top_k]

    result = evaluate_retrieval_method(
        method=(
            RetrievalMethod.DENSE
        ),
        cases=cases,
        retrieve_fn=(
            fake_retrieve
        ),
        evaluation_ks=[
            1,
            3,
            5,
        ],
        retrieval_top_k=5,
    )

    assert (
        result.query_count
        == 1
    )

    assert called_queries == [
        "可回答问题"
    ]


def test_runner_calculates_metrics() -> None:
    """Runner 应正确计算 Dataset 指标。"""

    cases = [
        RetrievalEvalCase(
            query_id="R001",
            query="问题一",
            gold_chunk_ids=(
                "gold-1",
            ),
            category=(
                RetrievalEvalCategory.DIRECT
            ),
            answerable=True,
            note="",
        ),
        RetrievalEvalCase(
            query_id="R002",
            query="问题二",
            gold_chunk_ids=(
                "gold-2",
            ),
            category=(
                RetrievalEvalCategory.DIRECT
            ),
            answerable=True,
            note="",
        ),
    ]

    def fake_retrieve(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        if query == "问题一":
            ids = [
                "gold-1",
                "x",
            ]
        else:
            ids = [
                "x",
                "gold-2",
            ]

        return [
            make_dense_result(
                chunk_id
            )
            for chunk_id in ids
        ][:top_k]

    result = evaluate_retrieval_method(
        method=(
            RetrievalMethod.DENSE
        ),
        cases=cases,
        retrieve_fn=(
            fake_retrieve
        ),
        evaluation_ks=[
            1,
            2,
        ],
        retrieval_top_k=2,
    )

    # Query1 Top1 命中
    # Query2 Top1 未命中
    #
    # Mean Recall@1 = 0.5
    assert (
        result
        .aggregate_by_k[1]
        .mean_recall
        == 0.5
    )

    # Query1 RR = 1
    # Query2 RR = 1/2
    #
    # MRR@2 = 0.75
    assert (
        result
        .aggregate_by_k[2]
        .mrr
        == 0.75
    )


def test_runner_rejects_k_larger_than_retrieval_top_k() -> None:
    """Metric K 不能超过实际 Retrieval TopK。"""

    cases = [
        RetrievalEvalCase(
            query_id="R001",
            query="测试",
            gold_chunk_ids=(
                "gold",
            ),
            category=(
                RetrievalEvalCategory.DIRECT
            ),
            answerable=True,
            note="",
        )
    ]

    def fake_retrieve(
        query: str,
        top_k: int,
        access_context: AccessContext,
    ):
        return []

    try:
        evaluate_retrieval_method(
            method=(
                RetrievalMethod.DENSE
            ),
            cases=cases,
            retrieve_fn=(
                fake_retrieve
            ),
            evaluation_ks=[
                10,
            ],
            retrieval_top_k=5,
        )

    except ValueError as exc:
        assert (
            "retrieval_top_k"
            in str(exc)
        )

    else:
        raise AssertionError(
            "预期 ValueError"
        )