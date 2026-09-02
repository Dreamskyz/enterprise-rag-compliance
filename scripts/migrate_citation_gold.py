"""为现有 Evaluation Dataset 增加 Citation Gold 标注。"""

import json
from pathlib import Path


INPUT_PATH = Path(
    "data/eval/retrieval_eval_v1.jsonl"
)

OUTPUT_PATH = Path(
    "data/eval/retrieval_eval_v1_audited.jsonl"
)


# ==========================================================
# 人工 Citation Gold Audit Override。
#
# 这些不是根据模型输出自动生成，
# 而是经过人工核对法规正文后的标注。
# ==========================================================

CITATION_GOLD_OVERRIDES: dict[
    str,
    list[str],
] = {
    "R001": [
        "cn_genai_interim_2023__第七条",
    ],

    "R006": [
        "cn_deep_synthesis_2022__第十四条",
        "cn_deep_synthesis_2022__第十五条",
        "cn_deep_synthesis_2022__第十六条",
        "cn_deep_synthesis_2022__第十七条",
    ],

    "R008": [
        "cn_deep_synthesis_2022__第七条",
        "cn_deep_synthesis_2022__第十五条",
    ],

    "R010": [
        "cn_genai_interim_2023__第十四条",
        "cn_deep_synthesis_2022__第十条",
    ],

    "R014": [
        "cn_genai_interim_2023__第七条",
        "cn_genai_interim_2023__第四条",
        "cn_deep_synthesis_2022__第十四条",
    ],
}


# ==========================================================
# 人工决定不进入严格 Citation Precision 的 Query。
# ==========================================================

NON_STRICT_CITATION_QUERY_IDS = {
    "R010",
    "R014",
}


def main() -> None:
    """
    将旧 Retrieval Eval Dataset 升级成
    Citation-aware Evaluation Dataset。

    默认规则：

        answerable=true：
            citation_gold = retrieval gold

        answerable=false：
            citation_gold = []

        strict citation：
            answerable
            且 category != ambiguous

    然后应用人工 Audit Override。
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"输入文件不存在：{INPUT_PATH}"
        )

    output_rows: list[
        dict
    ] = []

    with INPUT_PATH.open(
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

            data = json.loads(
                line
            )

            query_id = data[
                "query_id"
            ]

            answerable = data[
                "answerable"
            ]

            category = data[
                "category"
            ]

            retrieval_gold = list(
                data[
                    "gold_chunk_ids"
                ]
            )

            # ==================================================
            # 默认 Citation Gold。
            # ==================================================

            if answerable:
                citation_gold = list(
                    retrieval_gold
                )
            else:
                citation_gold = []

            # ==================================================
            # 人工 Audit Override。
            # ==================================================

            if (
                query_id
                in CITATION_GOLD_OVERRIDES
            ):
                citation_gold = list(
                    CITATION_GOLD_OVERRIDES[
                        query_id
                    ]
                )

            # ==================================================
            # Strict Citation Eval。
            # ==================================================

            strict_citation_eval = (
                answerable
                and category != "ambiguous"
            )

            if (
                query_id
                in NON_STRICT_CITATION_QUERY_IDS
            ):
                strict_citation_eval = False

            data[
                "citation_gold_chunk_ids"
            ] = citation_gold

            data[
                "strict_citation_eval"
            ] = strict_citation_eval

            output_rows.append(
                data
            )

    with OUTPUT_PATH.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        for row in output_rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )

    print(
        "Citation-aware dataset created:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "Cases:",
        len(output_rows),
    )

    print()

    print(
        "⚠ 请人工检查 diff，"
        "确认无误后再替换正式 Dataset。"
    )


if __name__ == "__main__":
    main()