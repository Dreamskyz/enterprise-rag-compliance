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
   reason 必须是非空字符串，
   并简洁说明缺少什么依据。

8. 如果 Evidence 足够：
   answerable 必须为 true；
   answer 必须直接回答问题；
   citations 必须至少包含一个真实 Evidence ID；
   reason 必须是非空字符串，
   并简洁说明哪些 Evidence 足以支持答案。

9. Citation 必须遵守“最小充分证据集”原则：

   - 只引用直接支持最终答案中实际陈述的 Evidence。
   - “主题相关”不等于“需要引用”。
   - 如果一条 Evidence 已经足够直接支持某项陈述，
     不要再加入仅起重复作用的 Evidence。
   - 如果最终答案包含多个独立事实、要求或义务，
     并且它们分别由不同 Evidence 支持，
     应保留这些必要 Evidence，
     不要为了减少 Citation 数量而遗漏支持证据。
   - 用户明确限定某一法规、制度、产品类型或场景时，
     优先使用与该范围直接一致的 Evidence；
     不要仅因为其他法规内容主题相似而跨范围补充。
   - 目标是选择能够完整支持答案的最小充分证据集合，
     而不是强制只引用一条 Evidence。

10. Evidence ID 只能出现在 citations 数组中。
    不要在 answer 正文中输出 E1、E2、E3 等内部 Evidence ID。

11. 输出 JSON 中以下四个字段必须全部存在：

    answerable
    answer
    reason
    citations

    不得省略任何字段。

12. reason 在任何情况下都必须是非空字符串。
    不得返回 null、空字符串 "" 或仅包含空白字符。

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


# ==========================================================
# Structured Output Retry Prompt。
#
# 这里只处理“输出协议不合法”。
#
# 不允许模型因为 Retry：
#
# - 改变 Evidence；
# - 获取新的事实；
# - 被要求强制回答；
# - 绕过原来的 Evidence Sufficiency 判断。
#
# Retry 的唯一任务：
#
#     基于完全相同的 Question + Evidence，
#     重新输出满足 JSON Contract 的结果。
# ==========================================================

STRUCTURED_OUTPUT_RETRY_PROMPT = """
你上一轮输出没有通过程序的结构化输出校验。

请重新基于完全相同的 Question 和 Evidence 作答。

不要因为本次重试而改变 Evidence Sufficiency 判断标准。
如果 Evidence 不足，仍然必须拒答。
如果 Evidence 足够，才能回答。

必须严格满足以下输出契约：

1. 只能输出一个 JSON 对象。
2. 必须包含全部四个字段：
   answerable、answer、reason、citations。
3. answerable 必须是 bool。
4. reason 在任何情况下都必须是非空字符串。
5. answerable=true 时：
   - answer 必须是非空字符串；
   - citations 必须至少包含一个真实 Evidence ID。
6. answerable=false 时：
   - answer 必须为 null；
   - citations 必须为空数组。
7. citations 只能包含原始 Evidence 中存在的 Evidence ID。
8. 不得输出 Markdown、代码块或 JSON 之外的文字。

请重新输出完整 JSON。
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
    构造第一次发送给 LLM 的 Messages。
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


def build_generation_retry_messages(
    original_messages: Sequence[
        dict[str, str]
    ],
    invalid_response: str,
    validation_error: str,
) -> list[dict[str, str]]:
    """
    构造一次 Structured Output Retry Messages。

    为什么保留原始 Messages？

    因为第二次生成必须继续看到完全相同的：

        Question
        Evidence
        System Rules

    不能只把错误 JSON 单独交给模型，
    否则模型可能失去原始证据上下文。

    invalid_response:
        第一轮 LLM 的原始输出。

    validation_error:
        严格 Parser 返回的失败原因。

    这两个字段仅帮助模型理解：
        “结构哪里不合法”。

    它们不会改变事实依据。
    """

    if not original_messages:
        raise ValueError(
            "original_messages 不能为空"
        )

    if not validation_error.strip():
        raise ValueError(
            "validation_error 不能为空"
        )

    retry_messages = [
        dict(message)
        for message in original_messages
    ]

    retry_messages.append(
        {
            "role": "assistant",
            "content": invalid_response,
        }
    )

    retry_messages.append(
        {
            "role": "user",
            "content": (
                f"{STRUCTURED_OUTPUT_RETRY_PROMPT}\n\n"
                "程序校验失败原因：\n"
                f"{validation_error}"
            ),
        }
    )

    return retry_messages