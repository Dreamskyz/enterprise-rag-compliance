"""观察 Answerable / Unanswerable Query 的 Top1 Rerank Score。"""

from dataclasses import dataclass
from pathlib import Path

from enterprise_rag.embeddings.bge_m3 import (
    BGEEmbeddingService,
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


CHUNKS_PATH = Path(
    "data/processed/chunks.jsonl"
)


@dataclass(frozen=True)
class ScoreCase:
    """
    一条 Evidence Score 检查样本。

    query_id:
        样本编号。

    query:
        实际输入 Retrieval Pipeline 的 Query。

    expected_answerable:
        True:
            当前知识库应该存在足够依据。

        False:
            当前知识库不应该能够可靠回答。

    case_type:
        样本难度 / 类型。

        normal_positive
            普通可回答问题。

        hard_positive
            有答案，但 Query 更短、更口语、
            更模糊或者属于同义改写。

        normal_negative
            普通不可回答问题。

        hard_negative
            与当前知识域高度相关，
            但要求的具体事实不存在。
    """

    query_id: str
    query: str
    expected_answerable: bool
    case_type: str


CASES = [
    # ==================================================
    # Normal Positive
    # ==================================================

    ScoreCase(
        query_id="A1",
        query=(
            "生成式人工智能服务处理训练数据"
            "需要遵守什么规定？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    ScoreCase(
        query_id="A2",
        query=(
            "生成式人工智能研发过程中进行"
            "数据标注需要满足什么要求？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    ScoreCase(
        query_id="A3",
        query=(
            "生成式人工智能服务提供者发现"
            "违法内容后应当如何处理？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    ScoreCase(
        query_id="A4",
        query=(
            "生成式人工智能服务提供者"
            "应当如何保护使用者的输入信息"
            "和使用记录？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    ScoreCase(
        query_id="A5",
        query=(
            "深度合成服务提供者"
            "应当如何管理训练数据？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    ScoreCase(
        query_id="A6",
        query=(
            "深度合成服务提供者提供"
            "人脸、人声编辑功能时"
            "有什么要求？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    ScoreCase(
        query_id="A7",
        query=(
            "深度合成服务提供者开发"
            "具有舆论属性或者社会动员能力"
            "的新功能时需要做什么？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    ScoreCase(
        query_id="A8",
        query=(
            "深度合成服务提供者对"
            "生成合成类算法机制机理"
            "需要进行哪些管理？"
        ),
        expected_answerable=True,
        case_type="normal_positive",
    ),

    # ==================================================
    # Hard Positive
    #
    # 当前知识库明确有依据，
    # 但 Query 表达方式更困难。
    # ==================================================

    ScoreCase(
        query_id="HP1",
        query="数据标注质量评估",
        expected_answerable=True,
        case_type="hard_positive",
    ),

    ScoreCase(
        query_id="HP2",
        query="训练语料有哪些合规要求？",
        expected_answerable=True,
        case_type="hard_positive",
    ),

    ScoreCase(
        query_id="HP3",
        query="违法内容怎么处理？",
        expected_answerable=True,
        case_type="hard_positive",
    ),

    ScoreCase(
        query_id="HP4",
        query=(
            "用户输入和使用记录"
            "怎么保护？"
        ),
        expected_answerable=True,
        case_type="hard_positive",
    ),

    ScoreCase(
        query_id="HP5",
        query=(
            "人脸编辑要不要"
            "取得本人同意？"
        ),
        expected_answerable=True,
        case_type="hard_positive",
    ),

    ScoreCase(
        query_id="HP6",
        query=(
            "新上线有舆论属性的"
            "功能要做什么？"
        ),
        expected_answerable=True,
        case_type="hard_positive",
    ),

    # ==================================================
    # Normal Negative
    # ==================================================

    ScoreCase(
        query_id="U1",
        query="南京明天会下雨吗？",
        expected_answerable=False,
        case_type="normal_negative",
    ),

    ScoreCase(
        query_id="U2",
        query=(
            "Python 中列表和元组"
            "有什么区别？"
        ),
        expected_answerable=False,
        case_type="normal_negative",
    ),

    ScoreCase(
        query_id="U3",
        query=(
            "我国生成式人工智能企业"
            "2026 年的平均研发预算是多少？"
        ),
        expected_answerable=False,
        case_type="normal_negative",
    ),

    ScoreCase(
        query_id="U4",
        query=(
            "生成式人工智能服务管理暂行办法"
            "规定的最高罚款金额是多少？"
        ),
        expected_answerable=False,
        case_type="normal_negative",
    ),

    ScoreCase(
        query_id="U5",
        query=(
            "深度合成服务提供者"
            "必须购买哪一家厂商的"
            "内容审核系统？"
        ),
        expected_answerable=False,
        case_type="normal_negative",
    ),

    ScoreCase(
        query_id="U6",
        query=(
            "公司的 AI 模型服务器"
            "管理员密码是什么？"
        ),
        expected_answerable=False,
        case_type="normal_negative",
    ),

    ScoreCase(
        query_id="U7",
        query=(
            "我们公司目前使用的"
            "内部大模型具体有多少参数？"
        ),
        expected_answerable=False,
        case_type="normal_negative",
    ),

    ScoreCase(
        query_id="U8",
        query=(
            "公司规定开发人员每月最多"
            "可以调用大模型 API 多少次？"
        ),
        expected_answerable=False,
        case_type="normal_negative",
    ),

    # ==================================================
    # Hard Negative
    #
    # 与法规主题高度相关，
    # 但要求的具体事实并不存在。
    # ==================================================

    ScoreCase(
        query_id="HN1",
        query=(
            "生成式人工智能服务管理暂行办法"
            "要求模型训练数据必须保存几年？"
        ),
        expected_answerable=False,
        case_type="hard_negative",
    ),

    ScoreCase(
        query_id="HN2",
        query=(
            "生成式人工智能服务管理暂行办法"
            "规定数据标注人员最低学历是什么？"
        ),
        expected_answerable=False,
        case_type="hard_negative",
    ),

    ScoreCase(
        query_id="HN3",
        query=(
            "生成式人工智能服务管理暂行办法"
            "规定发现违法内容后"
            "必须在几小时内处理？"
        ),
        expected_answerable=False,
        case_type="hard_negative",
    ),

    ScoreCase(
        query_id="HN4",
        query=(
            "互联网信息服务深度合成管理规定"
            "要求训练数据至少达到多少条？"
        ),
        expected_answerable=False,
        case_type="hard_negative",
    ),

    ScoreCase(
        query_id="HN5",
        query=(
            "互联网信息服务深度合成管理规定"
            "要求深度合成服务必须使用"
            "哪一种内容审核算法？"
        ),
        expected_answerable=False,
        case_type="hard_negative",
    ),

    ScoreCase(
        query_id="HN6",
        query=(
            "互联网信息服务深度合成管理规定"
            "规定安全评估必须由"
            "哪一家指定机构完成？"
        ),
        expected_answerable=False,
        case_type="hard_negative",
    ),
]


def print_score_summary(
    name: str,
    scores: list[float],
) -> None:
    """
    打印某一类样本的 Score 简单统计。

    当前只计算：

        count
        min
        max
        avg

    这仍然是 Preliminary Inspection，
    暂时不引入更复杂的统计指标。
    """

    if not scores:
        return

    print()
    print(name + ":")

    print(
        "  count =",
        len(scores),
    )

    print(
        "  min   =",
        round(
            min(scores),
            4,
        ),
    )

    print(
        "  max   =",
        round(
            max(scores),
            4,
        ),
    )

    print(
        "  avg   =",
        round(
            sum(scores)
            / len(scores),
            4,
        ),
    )


def main() -> None:
    """
    运行真实 Retrieval Pipeline，
    观察 Answerable / Unanswerable
    以及 Hard Positive / Hard Negative
    的 Top1 Rerank Score。

    注意：

    本脚本仍然属于 Preliminary Inspection。

    不在这里决定正式 Evidence Threshold。
    """

    print("=" * 100)
    print(
        "Evidence Score Hard Case Inspection"
    )
    print("=" * 100)

    # --------------------------------------------------
    # 1. 读取真实法规 Chunk。
    # --------------------------------------------------

    chunks = read_chunks_jsonl(
        CHUNKS_PATH
    )

    print(
        "Chunk count:",
        len(chunks),
    )

    print(
        "Query count:",
        len(CASES),
    )

    # --------------------------------------------------
    # 2. 初始化真实 Retrieval Pipeline。
    #
    # 模型只初始化一次，
    # 然后连续运行全部 Query。
    # --------------------------------------------------

    embedding_service = (
        BGEEmbeddingService()
    )

    dense_retriever = DenseRetriever(
        embedding_service=(
            embedding_service
        )
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks
    )

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

    reranker_service = (
        BGERerankerService()
    )

    retriever = RerankedRetriever(
        hybrid_retriever=(
            hybrid_retriever
        ),
        reranker_service=(
            reranker_service
        ),
        candidate_top_k=20,
    )

    # --------------------------------------------------
    # 3. 保存各类别分数。
    # --------------------------------------------------

    answerable_scores: list[
        float
    ] = []

    unanswerable_scores: list[
        float
    ] = []

    normal_positive_scores: list[
        float
    ] = []

    hard_positive_scores: list[
        float
    ] = []

    normal_negative_scores: list[
        float
    ] = []

    hard_negative_scores: list[
        float
    ] = []

    # --------------------------------------------------
    # 4. 逐条运行真实 Retrieval Pipeline。
    # --------------------------------------------------

    for case in CASES:
        results = retriever.search(
            query=case.query,
            top_k=5,
        )

        if not results:
            top_score = None
            top_chunk_id = None
            top_title = None
            top_article = None

        else:
            top_result = results[0]

            top_score = float(
                top_result.rerank_score
            )

            top_chunk_id = (
                top_result.candidate.chunk_id
            )

            top_title = (
                top_result.candidate.title
            )

            top_article = (
                top_result.candidate.article_number
            )

            # ------------------------------------------
            # Answerable / Unanswerable 总体统计。
            # ------------------------------------------

            if case.expected_answerable:
                answerable_scores.append(
                    top_score
                )
            else:
                unanswerable_scores.append(
                    top_score
                )

            # ------------------------------------------
            # 更细的难度类别统计。
            # ------------------------------------------

            if (
                case.case_type
                == "normal_positive"
            ):
                normal_positive_scores.append(
                    top_score
                )

            elif (
                case.case_type
                == "hard_positive"
            ):
                hard_positive_scores.append(
                    top_score
                )

            elif (
                case.case_type
                == "normal_negative"
            ):
                normal_negative_scores.append(
                    top_score
                )

            elif (
                case.case_type
                == "hard_negative"
            ):
                hard_negative_scores.append(
                    top_score
                )

            else:
                raise ValueError(
                    "未知 case_type："
                    f"{case.case_type}"
                )

        label = (
            "ANSWERABLE"
            if case.expected_answerable
            else "UNANSWERABLE"
        )

        print()
        print("-" * 100)

        print(
            f"{case.query_id} | "
            f"{label} | "
            f"{case.case_type}"
        )

        print(
            "Query:",
            case.query,
        )

        print(
            "Top1 Rerank Score:",
            top_score,
        )

        print(
            "Top1 Chunk:",
            top_chunk_id,
        )

        print(
            "Top1 Title:",
            top_title,
        )

        print(
            "Top1 Article:",
            top_article,
        )

    # --------------------------------------------------
    # 5. 总体 Score Summary。
    # --------------------------------------------------

    print()
    print("=" * 100)
    print(
        "Overall Score Summary"
    )
    print("=" * 100)

    print_score_summary(
        "Answerable",
        answerable_scores,
    )

    print_score_summary(
        "Unanswerable",
        unanswerable_scores,
    )

    # --------------------------------------------------
    # 6. Hard / Normal 分类统计。
    # --------------------------------------------------

    print()
    print("=" * 100)
    print(
        "Case Type Score Summary"
    )
    print("=" * 100)

    print_score_summary(
        "Normal Positive",
        normal_positive_scores,
    )

    print_score_summary(
        "Hard Positive",
        hard_positive_scores,
    )

    print_score_summary(
        "Normal Negative",
        normal_negative_scores,
    )

    print_score_summary(
        "Hard Negative",
        hard_negative_scores,
    )

    # --------------------------------------------------
    # 7. Overall Separation。
    # --------------------------------------------------

    if (
        answerable_scores
        and unanswerable_scores
    ):
        answerable_min = min(
            answerable_scores
        )

        unanswerable_max = max(
            unanswerable_scores
        )

        gap = (
            answerable_min
            - unanswerable_max
        )

        print()
        print("=" * 100)
        print(
            "Overall Separation"
        )
        print("=" * 100)

        print(
            "Answerable min:",
            round(
                answerable_min,
                4,
            ),
        )

        print(
            "Unanswerable max:",
            round(
                unanswerable_max,
                4,
            ),
        )

        print(
            "Simple gap:",
            round(
                gap,
                4,
            ),
        )

        if gap > 0:
            print()
            print(
                "✅ 当前扩展样本中仍存在"
                " Score Separation"
            )
        else:
            print()
            print(
                "⚠ 当前扩展样本中"
                " Answerable / Unanswerable "
                "Score 已出现重叠"
            )

    # --------------------------------------------------
    # 8. Hard Case Separation。
    #
    # 这一组尤其值得关注：
    #
    # Lowest hard positive
    # vs
    # highest hard negative
    # --------------------------------------------------

    if (
        hard_positive_scores
        and hard_negative_scores
    ):
        hard_positive_min = min(
            hard_positive_scores
        )

        hard_negative_max = max(
            hard_negative_scores
        )

        hard_gap = (
            hard_positive_min
            - hard_negative_max
        )

        print()
        print("=" * 100)
        print(
            "Hard Case Separation"
        )
        print("=" * 100)

        print(
            "Hard Positive min:",
            round(
                hard_positive_min,
                4,
            ),
        )

        print(
            "Hard Negative max:",
            round(
                hard_negative_max,
                4,
            ),
        )

        print(
            "Hard gap:",
            round(
                hard_gap,
                4,
            ),
        )

        if hard_gap > 0:
            print()
            print(
                "✅ 当前 Hard Case 中仍存在"
                "初步 Score Separation"
            )
        else:
            print()
            print(
                "⚠ Hard Positive / "
                "Hard Negative 已发生重叠"
            )

    print()
    print("=" * 100)

    print(
        "⚠ 本结果仍属于 Preliminary "
        "Evidence Calibration，"
        "不是正式 Evaluation。"
    )

    print(
        "⚠ 当前脚本只观察分数，"
        "不决定正式 Evidence Threshold。"
    )


if __name__ == "__main__":
    main()