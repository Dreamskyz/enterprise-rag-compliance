"""构造 Evidence-Constrained Generation Prompt。"""

from collections.abc import Sequence

from enterprise_rag.generation.models import (
    EvidenceItem,
)
from enterprise_rag.retrieval.models import (
    RerankedSearchResult,
)


SYSTEM_PROMPT = """
你是“企业 AI 合规与应用规范助手”。

你必须严格依据用户提供的 Evidence 回答，
不得使用 Evidence 之外的知识补充事实。

你的首要任务不是强行回答，
而是先判断 Evidence 是否足以支持用户问题。

必须遵守以下规则：

1. 只能使用 Evidence 中明确出现的信息。

2. 如果 Evidence 与问题相关，
   但没有明确提供用户所要求的具体事实，
   必须判断为不可回答。

3. 特别注意具体数字、时间、期限、金额、比例、
   人名、机构名、产品名、数量等事实。
   如果 Evidence 没有明确给出，不得猜测。

4. 不得把模糊表述擅自转换成精确事实。
   例如：
       Evidence 只写“及时处理”，
       不得推断成“24 小时内处理”。

5. 如果问题存在依据，
   citations 只能使用提供给你的 Evidence ID，
   例如：
       E1
       E2

6. 不得编造 Evidence ID，
   不得自行编造法规、条款或来源。

7. 如果 Evidence 不足：
       answerable 必须为 false；
       answer 必须为 null；
       citations 必须为空数组；
       reason 应简洁说明缺少什么依据。

8. 如果 Evidence 足够：
       answerable 必须为 true；
       answer 必须直接回答问题；
       citations 必须至少包含一个真实 Evidence ID。

你只能输出一个 JSON 对象。

禁止输出 Markdown。
禁止输出 ```json 代码块。
禁止输出 JSON 之外的解释文字。

JSON 格式必须为：

{
  "answerable": true,
  "answer": "回答文本",
  "reason": "证据为何足够",
  "citations": ["E1"]
}

或者：

{
  "answerable": false,
  "answer": null,
  "reason": "证据不足的原因",
  "citations": []
}
""".strip()


def build_evidence_items(
    results: Sequence[
        RerankedSearchResult
    ],
    max_evidence: int = 5,
) -> list[EvidenceItem]:
    """
    将 Retrieval Result 转换为
    带有 E1 / E2 / ... 编号的 Evidence。

    这里使用 content，
    而不是 retrieval_text。

    原因：
        retrieval_text 是为检索优化的表示；

        Generation 更适合看到：
            标题
            条款
            正文

        避免不必要的重复文本。
    """

    if max_evidence <= 0:
        raise ValueError(
            "max_evidence 必须大于 0"
        )

    items: list[
        EvidenceItem
    ] = []

    for index, result in enumerate(
        results[:max_evidence],
        start=1,
    ):
        candidate = result.candidate

        items.append(
            EvidenceItem(
                evidence_id=f"E{index}",
                chunk_id=candidate.chunk_id,
                title=candidate.title,
                article_number=(
                    candidate.article_number
                ),
                content=candidate.content,
                source_url=(
                    candidate.source_url
                ),
            )
        )

    return items


def build_generation_messages(
    query: str,
    evidence_items: Sequence[
        EvidenceItem
    ],
) -> list[dict[str, str]]:
    """
    构造发送给 LLM 的 Messages。
    """

    if not query.strip():
        raise ValueError(
            "query 不能为空"
        )

    if not evidence_items:
        raise ValueError(
            "evidence_items 不能为空"
        )

    evidence_blocks: list[str] = []

    for item in evidence_items:
        evidence_blocks.append(
            "\n".join([
                f"[{item.evidence_id}]",
                f"文档：{item.title}",
                f"条款：{item.article_number}",
                "正文：",
                item.content,
            ])
        )

    evidence_text = "\n\n".join(
        evidence_blocks
    )

    user_prompt = (
        "请根据以下 Evidence "
        "判断是否足以回答问题。\n\n"
        f"Question:\n{query.strip()}\n\n"
        "Evidence:\n"
        f"{evidence_text}"
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]