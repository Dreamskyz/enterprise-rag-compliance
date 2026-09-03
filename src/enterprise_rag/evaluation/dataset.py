"""Evaluation Dataset JSONL Loader。"""

import json
from pathlib import Path

from enterprise_rag.acl.models import (
    UserRole,
)
from enterprise_rag.evaluation.models import (
    RetrievalEvalCase,
    RetrievalEvalCategory,
)


def read_retrieval_eval_jsonl(
    input_path: Path,
) -> list[
    RetrievalEvalCase
]:
    """
    读取 Retrieval / Full-RAG Evaluation Dataset。

    每一行是一条独立 JSON。

    当前 Dataset 同时可以包含：

        Retrieval Gold
        Citation Gold
        Answerability Gold
        Retrieval Role

    关于 role：

    retrieval_eval_v1.jsonl
    是已经冻结的历史 Regression Dataset。

    它创建时没有显式 role 字段，
    当时 Retrieval Runner 默认使用：

        guest

    所以为了保证历史评测语义不发生变化：

        缺少 role
        -> 默认 UserRole.GUEST

    新的 V2 Dataset
    则应该显式填写：

        "role": "guest"

    或：

        "role": "developer"

    这样 Dataset 本身就能完整描述
    一条 Retrieval Evaluation Case
    所对应的授权 Candidate Space。
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Evaluation Dataset 不存在：{input_path}"
        )

    cases: list[
        RetrievalEvalCase
    ] = []

    seen_query_ids: set[
        str
    ] = set()

    with input_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                data = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Evaluation Dataset "
                    f"第 {line_number} 行不是合法 JSON"
                ) from exc

            # ==================================================
            # 1. 基础字段。
            # ==================================================

            query_id = str(
                data.get(
                    "query_id",
                    "",
                )
            ).strip()

            query = str(
                data.get(
                    "query",
                    "",
                )
            ).strip()

            category_raw = str(
                data.get(
                    "category",
                    "",
                )
            ).strip()

            answerable = data.get(
                "answerable"
            )

            note = str(
                data.get(
                    "note",
                    "",
                )
            ).strip()

            gold_raw = data.get(
                "gold_chunk_ids"
            )

            citation_gold_raw = data.get(
                "citation_gold_chunk_ids"
            )

            strict_citation_eval = data.get(
                "strict_citation_eval"
            )

            # ==================================================
            # 1.1 Retrieval Role。
            # ==================================================
            #
            # V1 Dataset 没有 role 字段。
            #
            # 不能因此修改已经冻结的 V1 文件，
            # 而应该由 Loader 提供历史兼容行为：
            #
            #     missing role
            #     -> guest
            #
            # 这正好与原 Runner 的默认角色一致。
            role_raw = str(
                data.get(
                    "role",
                    UserRole.GUEST.value,
                )
            ).strip()

            # ==================================================
            # 2. 基础 Validation。
            # ==================================================

            if not query_id:
                raise ValueError(
                    f"第 {line_number} 行 "
                    "query_id 不能为空"
                )

            if query_id in seen_query_ids:
                raise ValueError(
                    "发现重复 query_id："
                    f"{query_id}"
                )

            if not query:
                raise ValueError(
                    f"{query_id} query 不能为空"
                )

            if not isinstance(
                answerable,
                bool,
            ):
                raise ValueError(
                    f"{query_id} "
                    "answerable 必须是 bool"
                )

            if not isinstance(
                gold_raw,
                list,
            ):
                raise ValueError(
                    f"{query_id} "
                    "gold_chunk_ids 必须是数组"
                )

            if not isinstance(
                citation_gold_raw,
                list,
            ):
                raise ValueError(
                    f"{query_id} "
                    "citation_gold_chunk_ids "
                    "必须是数组"
                )

            if not isinstance(
                strict_citation_eval,
                bool,
            ):
                raise ValueError(
                    f"{query_id} "
                    "strict_citation_eval "
                    "必须是 bool"
                )

            # ==================================================
            # 3. Category。
            # ==================================================

            try:
                category = (
                    RetrievalEvalCategory(
                        category_raw
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    f"{query_id} "
                    "category 非法："
                    f"{category_raw}"
                ) from exc

            # ==================================================
            # 4. User Role。
            # ==================================================
            #
            # Dataset 不允许未知角色静默回退到 guest。
            #
            # 例如：
            #
            #     "role": "developr"
            #
            # 必须 fail-fast，
            # 否则一个拼写错误就会悄悄改变 ACL Candidate Space。
            try:
                role = UserRole(
                    role_raw
                )
            except ValueError as exc:
                raise ValueError(
                    f"{query_id} "
                    "role 非法："
                    f"{role_raw}"
                ) from exc

            # ==================================================
            # 5. Gold Chunk IDs。
            # ==================================================

            gold_chunk_ids = tuple(
                str(chunk_id).strip()
                for chunk_id in gold_raw
                if str(chunk_id).strip()
            )

            citation_gold_chunk_ids = tuple(
                str(chunk_id).strip()
                for chunk_id
                in citation_gold_raw
                if str(chunk_id).strip()
            )

            # ==================================================
            # 6. Answerability 与 Gold 一致性。
            # ==================================================

            if answerable:
                if not gold_chunk_ids:
                    raise ValueError(
                        f"{query_id} "
                        "answerable=true 时 "
                        "gold_chunk_ids 不能为空"
                    )

                if not citation_gold_chunk_ids:
                    raise ValueError(
                        f"{query_id} "
                        "answerable=true 时 "
                        "citation_gold_chunk_ids "
                        "不能为空"
                    )

            else:
                if gold_chunk_ids:
                    raise ValueError(
                        f"{query_id} "
                        "answerable=false 时 "
                        "gold_chunk_ids 必须为空"
                    )

                if citation_gold_chunk_ids:
                    raise ValueError(
                        f"{query_id} "
                        "answerable=false 时 "
                        "citation_gold_chunk_ids "
                        "必须为空"
                    )

                if strict_citation_eval:
                    raise ValueError(
                        f"{query_id} "
                        "answerable=false 时 "
                        "strict_citation_eval "
                        "必须为 false"
                    )

            # ==================================================
            # 7. 保存 Case。
            # ==================================================

            seen_query_ids.add(
                query_id
            )

            cases.append(
                RetrievalEvalCase(
                    query_id=(
                        query_id
                    ),
                    query=query,
                    gold_chunk_ids=(
                        gold_chunk_ids
                    ),
                    category=category,
                    answerable=(
                        answerable
                    ),
                    note=note,
                    citation_gold_chunk_ids=(
                        citation_gold_chunk_ids
                    ),
                    strict_citation_eval=(
                        strict_citation_eval
                    ),
                    role=role,
                )
            )

    if not cases:
        raise ValueError(
            "Evaluation Dataset 为空"
        )

    return cases