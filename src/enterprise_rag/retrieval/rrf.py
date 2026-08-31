"""Reciprocal Rank Fusion（RRF）实现。"""

from enterprise_rag.retrieval.models import (
    BM25SearchResult,
    DenseSearchResult,
    HybridSearchResult,
)


RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[DenseSearchResult],
    bm25_results: list[BM25SearchResult],
    rrf_k: int = RRF_K,
    top_k: int = 20,
) -> list[HybridSearchResult]:
    """
    使用 Reciprocal Rank Fusion 融合
    Dense 与 BM25 排名结果。

    公式：

        score(d) =
            Σ 1 / (rrf_k + rank_i(d))

    注意：
        rank 从 1 开始，而不是 0。

    参数：
        dense_results:
            Dense Retriever 已排序结果。

        bm25_results:
            BM25 Retriever 已排序结果。

        rrf_k:
            RRF 平滑常数。
            V1 baseline 使用 60。

        top_k:
            最终返回前 K 个融合结果。
    """

    if rrf_k < 0:
        raise ValueError(
            "rrf_k 不能小于 0"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0"
        )

    # 按 chunk_id 聚合两路结果。
    #
    # 结构：
    #
    # {
    #   chunk_id: {
    #       "candidate": ...,
    #       "rrf_score": ...,
    #       "dense_rank": ...,
    #       "bm25_rank": ...,
    #       "dense_score": ...,
    #       "bm25_score": ...
    #   }
    # }

    fused: dict[
        str,
        dict[str, object],
    ] = {}

    # --------------------------------------------------
    # 1. Dense 排名贡献
    # --------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        chunk_id = (
            result.candidate.chunk_id
        )

        item = fused.setdefault(
            chunk_id,
            {
                "candidate": result.candidate,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            },
        )

        item["rrf_score"] = (
            float(item["rrf_score"])
            + 1.0 / (rrf_k + rank)
        )

        item["dense_rank"] = rank
        item["dense_score"] = result.score

    # --------------------------------------------------
    # 2. BM25 排名贡献
    # --------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_id = (
            result.candidate.chunk_id
        )

        item = fused.setdefault(
            chunk_id,
            {
                "candidate": result.candidate,
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
            },
        )

        item["rrf_score"] = (
            float(item["rrf_score"])
            + 1.0 / (rrf_k + rank)
        )

        item["bm25_rank"] = rank
        item["bm25_score"] = result.score

    # --------------------------------------------------
    # 3. 转成标准 HybridSearchResult
    # --------------------------------------------------

    results: list[
        HybridSearchResult
    ] = []

    for item in fused.values():
        results.append(
            HybridSearchResult(
                candidate=item["candidate"],
                rrf_score=float(
                    item["rrf_score"]
                ),
                dense_rank=(
                    item["dense_rank"]
                    if isinstance(
                        item["dense_rank"],
                        int,
                    )
                    else None
                ),
                bm25_rank=(
                    item["bm25_rank"]
                    if isinstance(
                        item["bm25_rank"],
                        int,
                    )
                    else None
                ),
                dense_score=(
                    float(
                        item["dense_score"]
                    )
                    if item["dense_score"]
                    is not None
                    else None
                ),
                bm25_score=(
                    float(
                        item["bm25_score"]
                    )
                    if item["bm25_score"]
                    is not None
                    else None
                ),
            )
        )

    # --------------------------------------------------
    # 4. 按 RRF score 从高到低排序
    # --------------------------------------------------

    results.sort(
        key=lambda result: (
            result.rrf_score
        ),
        reverse=True,
    )

    return results[:top_k]