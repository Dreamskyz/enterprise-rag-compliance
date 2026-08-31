"""检查 BGE Reranker 模型基础能力。"""

from enterprise_rag.reranking.bge_reranker import (
    BGERerankerService,
)


QUERY = (
    "生成式人工智能服务"
    "处理训练数据需要遵守什么规定？"
)

PASSAGES = [
    (
        "生成式人工智能服务管理暂行办法\n"
        "第二章 技术发展与治理\n"
        "第七条\n"
        "生成式人工智能服务提供者"
        "应当依法开展预训练、优化训练等"
        "训练数据处理活动，遵守有关规定。"
    ),
    (
        "生成式人工智能服务管理暂行办法\n"
        "第三章 服务规范\n"
        "第十五条\n"
        "提供者应当建立健全投诉、举报机制，"
        "及时受理和处理公众投诉举报。"
    ),
    (
        "今天天气很好，适合出去散步。"
    ),
]


def main() -> None:
    """
    验证：

    1. Reranker 能加载；
    2. GPU 推理正常；
    3. 返回 3 个 score；
    4. 训练数据条款应获得最高分。
    """

    print("=" * 80)
    print("BGE Reranker Check")
    print("=" * 80)

    print(
        "Query:",
        QUERY,
    )

    service = BGERerankerService()

    scores = service.compute_scores(
        query=QUERY,
        passages=PASSAGES,
    )

    print()

    for index, (
        passage,
        score,
    ) in enumerate(
        zip(
            PASSAGES,
            scores,
            strict=True,
        ),
        start=1,
    ):
        print(
            f"Passage {index}"
        )

        print(
            "Score:",
            round(
                score,
                4,
            ),
        )

        print(
            "Text:",
            passage[:100],
        )

        print("-" * 80)

    if len(scores) != len(
        PASSAGES
    ):
        raise RuntimeError(
            "Reranker Score 数量异常"
        )

    best_index = max(
        range(len(scores)),
        key=scores.__getitem__,
    )

    if best_index != 0:
        raise RuntimeError(
            "训练数据相关 Passage "
            "没有获得最高 Rerank Score"
        )

    print()
    print(
        "✅ BGE Reranker 基础能力正常"
    )


if __name__ == "__main__":
    main()