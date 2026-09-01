"""通过 HTTP 检查完整 FastAPI RAG Ask Runtime。"""

import httpx


BASE_URL = (
    "http://127.0.0.1:8000"
)


TEST_CASES = [
    (
        "ANSWERABLE",
        (
            "生成式人工智能服务处理训练数据"
            "需要遵守什么规定？"
        ),
    ),
    (
        "HARD_NEGATIVE",
        (
            "生成式人工智能服务管理暂行办法"
            "规定发现违法内容后"
            "必须在几小时内处理？"
        ),
    ),
    (
        "OUT_OF_DOMAIN",
        "南京明天会下雨吗？",
    ),
]


def main() -> None:
    """
    通过真实 HTTP 验证完整：

        FastAPI
            ↓
        QueryService
            ↓
        Retrieval
            ↓
        Gate
            ↓
        LLM
            ↓
        Answer / Refusal
    """

    print("=" * 100)
    print(
        "FastAPI Ask Runtime Check"
    )
    print("=" * 100)

    # --------------------------------------------------
    # Health
    # --------------------------------------------------

    health_response = httpx.get(
        f"{BASE_URL}/health",
        timeout=10.0,
    )

    health_response.raise_for_status()

    print(
        "Health:",
        health_response.json(),
    )

    # --------------------------------------------------
    # 三类真实 Query。
    # --------------------------------------------------

    for label, query in TEST_CASES:
        print()
        print("=" * 100)

        print(
            "Case:",
            label,
        )

        print(
            "Query:",
            query,
        )

        response = httpx.post(
            f"{BASE_URL}/api/v1/ask",
            json={
                "query": query,
                "role": "guest",
            },
            timeout=120.0,
        )

        response.raise_for_status()

        payload = response.json()

        print(
            "Gate Reason:",
            payload["gate_reason"],
        )

        print(
            "Top Rerank:",
            payload["top_rerank_score"],
        )

        print(
            "Answerable:",
            payload["answerable"],
        )

        print(
            "Answer:",
            payload["answer"],
        )

        print(
            "Reason:",
            payload["reason"],
        )

        print(
            "Citations:",
            payload["citations"],
        )

        # --------------------------------------------------
        # 核心行为断言。
        # --------------------------------------------------

        if label == "ANSWERABLE":
            assert (
                payload["gate_reason"]
                == "passed"
            )

            assert (
                payload["answerable"]
                is True
            )

            assert (
                payload["answer"]
                is not None
            )

            assert (
                len(
                    payload["citations"]
                )
                >= 1
            )

        elif label == "HARD_NEGATIVE":
            assert (
                payload["gate_reason"]
                == "passed"
            )

            assert (
                payload["answerable"]
                is False
            )

            assert (
                payload["answer"]
                is None
            )

            assert (
                payload["citations"]
                == []
            )

        elif label == "OUT_OF_DOMAIN":
            assert (
                payload["gate_reason"]
                == "below_threshold"
            )

            assert (
                payload["answerable"]
                is False
            )

            assert (
                payload["answer"]
                is None
            )

            assert (
                payload["citations"]
                == []
            )

    print()
    print("=" * 100)

    print(
        "✅ FastAPI 完整 Ask Runtime "
        "真实 HTTP 验证通过"
    )


if __name__ == "__main__":
    main()