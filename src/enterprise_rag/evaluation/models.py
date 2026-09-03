"""Evaluation Dataset 数据模型。"""

from dataclasses import dataclass
from enum import StrEnum

from enterprise_rag.acl.models import (
    UserRole,
)


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

            在当前 Case 的授权检索空间中，
            Retriever 应该优先找回的核心 Evidence。

    citation_gold_chunk_ids:

        Citation Gold。

        用于：

            Citation Precision
            Citation Recall
            Citation Hit Rate

        它表示：

            最终回答允许直接引用的
            支持性 Evidence。

        注意：

            Citation Gold
            不一定等于 Retrieval Gold。

    strict_citation_eval:

        是否进入严格 Citation Metrics。

        对 ambiguous / scope 不明确 Query，
        可以设置为 False。

        Query 仍然继续参加：

            Retrieval Evaluation
            Answerability Evaluation

        只是不会污染严格 Citation Precision。

    role:

        当前 Evaluation Case
        以什么用户角色执行 Retrieval。

        例如：

            guest
            developer
            admin

        role 会决定 Retriever
        实际允许访问的 Candidate Space。

        V1 Dataset 历史上没有 role 字段，
        当时 Runner 的默认角色就是：

            guest

        因此这里也将默认值保留为：

            UserRole.GUEST

        这样旧 Dataset 在 schema evolution 后
        仍然保持完全相同的 ACL 评测语义。

    注意：

        role 解决的是：

            “谁在问？”

        answerable 解决的是：

            “当前知识与证据是否足够回答？”

        两者不能混为一个概念。
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
    # Citation Evaluation Annotation。
    #
    # 保留默认值是为了避免项目中一些手工构造
    # RetrievalEvalCase 的测试立即全部失效。
    #
    # 正式从 Dataset Loader 读取时，
    # 我们仍然要求 Citation 字段明确存在。
    # ------------------------------------------------------

    citation_gold_chunk_ids: tuple[
        str,
        ...,
    ] = ()

    strict_citation_eval: bool = False

    # ------------------------------------------------------
    # Role-aware Retrieval Evaluation。
    #
    # 默认 guest 是为了保持
    # retrieval_eval_v1.jsonl 的历史运行语义。
    #
    # V2 Dataset 中则可以显式声明：
    #
    #     "role": "developer"
    #
    # 来评估 developer Candidate Space。
    # ------------------------------------------------------

    role: UserRole = UserRole.GUEST