"""解析并校验 Evidence-Constrained LLM 输出。"""

import json
from collections.abc import Sequence

from enterprise_rag.generation.models import (
    Citation,
    EvidenceItem,
    GroundedAnswer,
)


def _strip_code_fence(
    text: str,
) -> str:
    """
    对偶发 Markdown JSON Code Fence
    做最小兼容处理。

    例如：

        ```json
        {...}
        ```

    注意：
    这里只去围栏，不修复坏 JSON。
    """

    stripped = text.strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()

    if len(lines) < 3:
        return stripped

    if lines[0].startswith("```"):
        lines = lines[1:]

    if (
        lines
        and lines[-1].strip() == "```"
    ):
        lines = lines[:-1]

    return "\n".join(
        lines
    ).strip()


def parse_grounded_answer(
    raw_content: str,
    evidence_items: Sequence[
        EvidenceItem
    ],
) -> GroundedAnswer:
    """
    解析 LLM JSON 输出并进行业务校验。

    这里是 Generation 的重要安全边界：

    LLM 返回 Citation ID
        ↓
    Parser 验证 Citation 是否真实存在
        ↓
    程序根据 EvidenceItem
    构造真实 Citation Metadata
    """

    if not raw_content.strip():
        raise ValueError(
            "LLM 返回内容不能为空"
        )

    cleaned = _strip_code_fence(
        raw_content
    )

    try:
        payload = json.loads(
            cleaned
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM 返回内容不是合法 JSON"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "LLM JSON 顶层必须是对象"
        )

    answerable = payload.get(
        "answerable"
    )

    answer = payload.get(
        "answer"
    )

    reason = payload.get(
        "reason"
    )

    citation_ids = payload.get(
        "citations"
    )

    # --------------------------------------------------
    # 1. 基础 Schema Validation
    # --------------------------------------------------

    if not isinstance(
        answerable,
        bool,
    ):
        raise ValueError(
            "answerable 必须是 bool"
        )

    if not isinstance(
        reason,
        str,
    ) or not reason.strip():
        raise ValueError(
            "reason 必须是非空字符串"
        )

    if not isinstance(
        citation_ids,
        list,
    ):
        raise ValueError(
            "citations 必须是数组"
        )

    if not all(
        isinstance(
            citation_id,
            str,
        )
        for citation_id
        in citation_ids
    ):
        raise ValueError(
            "citations 中的每个元素"
            "都必须是字符串"
        )

    # --------------------------------------------------
    # 2. 可回答 / 不可回答语义约束
    # --------------------------------------------------

    if answerable:
        if not isinstance(
            answer,
            str,
        ) or not answer.strip():
            raise ValueError(
                "answerable=true 时 "
                "answer 必须是非空字符串"
            )

        if not citation_ids:
            raise ValueError(
                "answerable=true 时 "
                "citations 不能为空"
            )

    else:
        if answer is not None:
            raise ValueError(
                "answerable=false 时 "
                "answer 必须为 null"
            )

        if citation_ids:
            raise ValueError(
                "answerable=false 时 "
                "citations 必须为空数组"
            )

    # --------------------------------------------------
    # 3. Evidence ID 白名单
    # --------------------------------------------------

    evidence_map = {
        item.evidence_id: item
        for item in evidence_items
    }

    unknown_ids = [
        citation_id
        for citation_id in citation_ids
        if citation_id not in evidence_map
    ]

    if unknown_ids:
        raise ValueError(
            "LLM 返回了未知 Evidence ID："
            + ", ".join(
                unknown_ids
            )
        )

    # --------------------------------------------------
    # 4. 去重并生成确定性 Citation
    # --------------------------------------------------

    citations: list[
        Citation
    ] = []

    seen_ids: set[str] = set()

    for citation_id in citation_ids:
        if citation_id in seen_ids:
            continue

        seen_ids.add(
            citation_id
        )

        item = evidence_map[
            citation_id
        ]

        citations.append(
            Citation(
                evidence_id=(
                    item.evidence_id
                ),
                chunk_id=(
                    item.chunk_id
                ),
                title=item.title,
                article_number=(
                    item.article_number
                ),
                source_url=(
                    item.source_url
                ),
            )
        )

    return GroundedAnswer(
        answerable=answerable,
        answer=(
            answer.strip()
            if isinstance(
                answer,
                str,
            )
            else None
        ),
        reason=reason.strip(),
        citations=tuple(
            citations
        ),
    )