"""通过 HTTP 检查 FastAPI Retrieval Runtime。"""

import httpx


BASE_URL = (
    "http://127.0.0.1:8000"
)


def main() -> None:
    """
    检查真实 FastAPI Retrieval API。

    前提：

        1. Qdrant 已启动；
        2. FastAPI 已启动；
        3. RAG Runtime 已完成初始化。
    """

    print("=" * 100)
    print(
        "FastAPI Retrieval Runtime Check"
    )
    print("=" * 100)

    # --------------------------------------------------
    # 1. Health
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
    # 2. Real Retrieval
    # --------------------------------------------------

    response = httpx.post(
        f"{BASE_URL}/api/v1/retrieve",
        json={
            "query": (
                "生成式人工智能服务处理训练数据"
                "需要遵守什么规定？"
            ),
            "role": "guest",
            "top_k": 5,
        },
        timeout=60.0,
    )

    response.raise_for_status()

    payload = response.json()

    print()
    print(
        "Query:",
        payload["query"],
    )

    print(
        "Role:",
        payload["role"],
    )

    print(
        "Result Count:",
        payload["result_count"],
    )

    print()

    for item in payload["results"]:
        print(
            f'{item["rank"]}. '
            f'{item["title"]} '
            f'{item["article_number"]} '
            f'| rerank='
            f'{item["rerank_score"]:.4f}'
        )

    # --------------------------------------------------
    # 核心 Sanity Assertion
    # --------------------------------------------------

    assert (
        payload["result_count"]
        >= 1
    )

    top1 = payload[
        "results"
    ][0]

    assert (
        top1["chunk_id"]
        == (
            "cn_genai_interim_2023"
            "__第七条"
        )
    )

    print()
    print(
        "✅ FastAPI + Lifespan + "
        "真实 Retrieval Runtime 验证通过"
    )


if __name__ == "__main__":
    main()