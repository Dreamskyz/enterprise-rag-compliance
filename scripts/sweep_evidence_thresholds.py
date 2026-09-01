"""对 Preliminary Evidence Scores 执行 Threshold Sweep。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceScoreSample:
    """
    一条 Evidence Gate Score 样本。

    query_id:
        实验 Query 编号。

    expected_answerable:
        True:
            应当允许进入 Generation。

        False:
            应当被 Evidence Gate 拒绝。

    top_score:
        当前真实 Retrieval Pipeline
        得到的 Top1 Rerank Score。
    """

    query_id: str
    expected_answerable: bool
    top_score: float


# ==========================================================
# 这些分数来自当前真实：
#
# BGE-M3
# + Qdrant Dense
# + BM25
# + RRF
# + bge-reranker-v2-m3
#
# 对 49 个真实法规 Chunk 的 Preliminary Inspection。
#
# 注意：
# 这不是正式 Evaluation Dataset。
# ==========================================================

SAMPLES = [
    EvidenceScoreSample(
        query_id="A1",
        expected_answerable=True,
        top_score=7.1015625,
    ),
    EvidenceScoreSample(
        query_id="A2",
        expected_answerable=True,
        top_score=6.01171875,
    ),
    EvidenceScoreSample(
        query_id="A3",
        expected_answerable=True,
        top_score=7.78125,
    ),
    EvidenceScoreSample(
        query_id="A4",
        expected_answerable=True,
        top_score=7.78515625,
    ),
    EvidenceScoreSample(
        query_id="A5",
        expected_answerable=True,
        top_score=6.8203125,
    ),
    EvidenceScoreSample(
        query_id="A6",
        expected_answerable=True,
        top_score=6.6875,
    ),
    EvidenceScoreSample(
        query_id="A7",
        expected_answerable=True,
        top_score=5.90625,
    ),
    EvidenceScoreSample(
        query_id="A8",
        expected_answerable=True,
        top_score=7.13671875,
    ),

    EvidenceScoreSample(
        query_id="U1",
        expected_answerable=False,
        top_score=-8.6640625,
    ),
    EvidenceScoreSample(
        query_id="U2",
        expected_answerable=False,
        top_score=-8.40625,
    ),
    EvidenceScoreSample(
        query_id="U3",
        expected_answerable=False,
        top_score=-2.265625,
    ),
    EvidenceScoreSample(
        query_id="U4",
        expected_answerable=False,
        top_score=0.408935546875,
    ),
    EvidenceScoreSample(
        query_id="U5",
        expected_answerable=False,
        top_score=-1.25,
    ),
    EvidenceScoreSample(
        query_id="U6",
        expected_answerable=False,
        top_score=-6.453125,
    ),
    EvidenceScoreSample(
        query_id="U7",
        expected_answerable=False,
        top_score=-8.0546875,
    ),
    EvidenceScoreSample(
        query_id="U8",
        expected_answerable=False,
        top_score=-6.234375,
    ),
]


@dataclass(frozen=True)
class ThresholdMetrics:
    """
    某个 Threshold 对应的 Gate 指标。
    """

    threshold: float

    true_accept: int
    false_reject: int

    true_reject: int
    false_accept: int

    answerable_pass_rate: float
    unanswerable_rejection_rate: float
    accuracy: float


def evaluate_threshold(
    threshold: float,
) -> ThresholdMetrics:
    """
    评估一个 Evidence Threshold。

    Gate 规则：

        score >= threshold
        → PASS

        score < threshold
        → REJECT
    """

    true_accept = 0
    false_reject = 0

    true_reject = 0
    false_accept = 0

    for sample in SAMPLES:
        predicted_pass = (
            sample.top_score
            >= threshold
        )

        if sample.expected_answerable:
            if predicted_pass:
                true_accept += 1
            else:
                false_reject += 1

        else:
            if predicted_pass:
                false_accept += 1
            else:
                true_reject += 1

    answerable_count = sum(             #统计真值为可回答的样本数量
        1
        for sample in SAMPLES
        if sample.expected_answerable
    )

    unanswerable_count = (              #不可回答样本 = 总样本 − 可回答样本
        len(SAMPLES)
        - answerable_count
    )

    answerable_pass_rate = (            #可回答通过率（TPR，召回率）
        true_accept                     #所有本该回答的问题中，模型答对的比例。越高说明模型擅长回答能回答的问题
        / answerable_count
    )

    unanswerable_rejection_rate = (     #不可回答拒答率（TNR，特异度）
        true_reject                     #所有本该拒答的问题中，模型正确拒绝回答的比例。越高代表模型越不会幻觉编造答案
        / unanswerable_count
    )

    accuracy = (                        #总体准确率
        true_accept                     #所有样本上，模型判断（回答 / 拒答）完全正确的占比。样本不均衡时，`accuracy`参考价值有限
        + true_reject
    ) / len(SAMPLES)

    return ThresholdMetrics(
        threshold=threshold,
        true_accept=true_accept,
        false_reject=false_reject,
        true_reject=true_reject,
        false_accept=false_accept,
        answerable_pass_rate=(
            answerable_pass_rate
        ),
        unanswerable_rejection_rate=(
            unanswerable_rejection_rate
        ),
        accuracy=accuracy,
    )


def build_candidate_thresholds(
) -> list[float]:
    """
    构造 Threshold Sweep 候选值。

    不只测试整数。

    我们取：
        所有实际 Score
        +
        相邻 Score 的中点

    这样可以观察分类边界发生变化的位置。
    """

    scores = sorted({
        sample.top_score
        for sample in SAMPLES
    })

    thresholds: set[float] = set(
        scores
    )

    for left, right in zip(
        scores,
        scores[1:],
        strict=False,
    ):
        midpoint = (
            left + right
        ) / 2.0

        thresholds.add(
            midpoint
        )

    # 再加入几个便于阅读的常见候选值。
    thresholds.update({
        -2.0,
        -1.0,
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    })

    return sorted(
        thresholds
    )


def main() -> None:
    """运行 Threshold Sweep。"""

    print("=" * 100)
    print(
        "Evidence Gate Threshold Sweep"
    )
    print("=" * 100)

    thresholds = (
        build_candidate_thresholds()
    )

    perfect_thresholds: list[
        float
    ] = []

    for threshold in thresholds:
        metrics = evaluate_threshold(
            threshold
        )

        # 为了避免输出过于冗长，
        # 重点打印相对有意义的区间。
        if (
            -2.0
            <= threshold
            <= 6.5
        ):
            print(
                f"threshold={threshold:>7.4f} | "
                f"answerable_pass="
                f"{metrics.answerable_pass_rate:.3f} | "
                f"unanswerable_reject="
                f"{metrics.unanswerable_rejection_rate:.3f} | "
                f"false_reject="
                f"{metrics.false_reject} | "
                f"false_accept="
                f"{metrics.false_accept} | "
                f"accuracy="
                f"{metrics.accuracy:.3f}"
            )

        if (
            metrics.false_reject == 0
            and metrics.false_accept == 0
        ):
            perfect_thresholds.append(
                threshold
            )

    print()
    print("=" * 100)
    print(
        "Preliminary Perfect Thresholds"
    )
    print("=" * 100)

    if perfect_thresholds:
        print(
            "Candidate count:",
            len(perfect_thresholds),
        )

        print(
            "Observed minimum candidate:",
            round(
                min(perfect_thresholds),
                6,
            ),
        )

        print(
            "Observed maximum candidate:",
            round(
                max(perfect_thresholds),
                6,
            ),
        )
    else:
        print(
            "当前样本不存在同时实现"
            " 0 False Accept / 0 False Reject "
            "的阈值。"
        )

    print()
    print("=" * 100)

    # --------------------------------------------------
    # 单独检查几个最容易讨论的候选值。
    # --------------------------------------------------

    for threshold in [
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    ]:
        metrics = evaluate_threshold(
            threshold
        )

        print(
            f"threshold={threshold:.1f}"
        )

        print(
            "  Answerable Pass Rate:",
            f"{metrics.answerable_pass_rate:.1%}",
        )

        print(
            "  Unanswerable Rejection Rate:",
            f"{metrics.unanswerable_rejection_rate:.1%}",
        )

        print(
            "  False Reject:",
            metrics.false_reject,
        )

        print(
            "  False Accept:",
            metrics.false_accept,
        )

        print(
            "  Accuracy:",
            f"{metrics.accuracy:.1%}",
        )

        print()

    print(
        "⚠ 当前 Sweep 仅基于 16 条 "
        "Preliminary Samples。"
    )

    print(
        "⚠ 不应据此直接宣布正式 "
        "Evidence Threshold。"
    )


if __name__ == "__main__":
    main()