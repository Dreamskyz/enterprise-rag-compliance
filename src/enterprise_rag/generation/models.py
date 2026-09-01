"""基于检索证据生成回答时使用的数据模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceItem:
    """
    送入 LLM 的一条证据。

    evidence_id:
        当前请求内的临时证据编号，例如：

            E1
            E2
            E3

        该 ID 只在一次 Query 中有效。

    其他字段全部来自真实 RetrievalCandidate，
    不由 LLM 生成。
    """

    evidence_id: str

    chunk_id: str

    title: str

    article_number: str

    content: str

    source_url: str


@dataclass(frozen=True)
class Citation:
    """
    最终返回给用户的确定性 Citation。

    这些字段由程序根据 Evidence ID
    回查生成，而不是相信 LLM 自己写的来源。
    """

    evidence_id: str

    chunk_id: str

    title: str

    article_number: str

    source_url: str


@dataclass(frozen=True)
class GroundedAnswer:
    """
    Evidence-Constrained Generation 最终结果。

    answerable:
        当前证据是否足够支持回答。

    answer:
        如果可回答，则保存回答文本。

        如果不可回答，则为 None。

    reason:
        为什么回答 / 为什么拒答。

        主要用于：
            Debug
            Evaluation
            API
            日志

    citations:
        程序校验后得到的真实 Citation。
    """

    answerable: bool

    answer: str | None

    reason: str

    citations: tuple[Citation, ...]