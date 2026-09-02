"""Evaluation Dataset 数据模型。"""

from dataclasses import dataclass
from enum import StrEnum


class RetrievalEvalCategory(StrEnum):
    """
    Retrieval / RAG Evaluation Query 分类。

    direct:
        可以直接从原文找到答案。

    paraphrase:
        对原文进行同义改写。

    short:
        短 Query。

    ambiguous:
        范围较宽或存在多种合理解释。

    hard_negative:
        领域相关，但知识库中没有足够证据回答。

    out_of_domain:
        明显超出知识库范围。
    """

    DIRECT = "direct"

    PARAPHRASE = "paraphrase"

    SHORT = "short"

    AMBIGUOUS = "ambiguous"

    HARD_NEGATIVE = "hard_negative"

    OUT_OF_DOMAIN = "out_of_domain"


@dataclass(frozen=True)
class RetrievalEvalCase:
    """
    一条 Evaluation Case。

    gold_chunk_ids:
        Retrieval Gold。

        用于：

            Recall@K
            MRR

        它表示：
            Retriever 应该优先找回的核心 Evidence。

    citation_gold_chunk_ids:
        Citation Gold。

        用于：

            Citation Precision
            Citation Recall
            Citation Hit Rate

        它表示：
            最终回答允许直接引用的支持性 Evidence。

        注意：
            Citation Gold 不一定等于 Retrieval Gold。

    strict_citation_eval:
        是否进入严格 Citation Metrics。

        对 ambiguous / scope 不明确 Query，
        可以设置为 False。

        Query 仍然继续参加：

            Retrieval Evaluation
            Answerability Evaluation

        只是不会污染严格 Citation Precision。
    """

    query_id: str

    query: str

    gold_chunk_ids: tuple[
        str,
        ...,
    ]

    category: RetrievalEvalCategory

    answerable: bool

    note: str

    # ------------------------------------------------------
    # 新增 Citation Evaluation Annotation。
    #
    # 保留默认值是为了避免项目中一些手工构造
    # RetrievalEvalCase 的测试立即全部失效。
    #
    # 正式从 Dataset Loader 读取时，
    # 我们仍然会要求这两个字段明确存在。
    # ------------------------------------------------------

    citation_gold_chunk_ids: tuple[
        str,
        ...,
    ] = ()

    strict_citation_eval: bool = False