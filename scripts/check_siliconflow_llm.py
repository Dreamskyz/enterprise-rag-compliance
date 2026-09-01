"""检查 SiliconFlow LLM Client 是否可以正常调用。"""

from enterprise_rag.llm.siliconflow import (
    SiliconFlowLLMService,
)


def main() -> None:
    """
    执行一次最小真实 API 调用。

    这里只验证：

    1. .env 能否正确读取；
    2. API Key 是否有效；
    3. 模型名称是否可用；
    4. OpenAI-compatible API 是否正常；
    5. 能否拿到 Usage。
    """

    print("=" * 80)
    print(
        "SiliconFlow LLM Check"
    )
    print("=" * 80)

    service = (
        SiliconFlowLLMService()
    )

    print(
        "Model:",
        service.model,
    )

    print(
        "Base URL:",
        service.base_url,
    )

    response = service.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个企业 AI 合规助手。"
                    "请简洁回答。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请只回复："
                    "LLM连接正常"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=64,
    )

    print()
    print(
        "Response:"
    )

    print(
        response.content
    )

    print()
    print(
        "Prompt Tokens:",
        response.prompt_tokens,
    )

    print(
        "Completion Tokens:",
        response.completion_tokens,
    )

    print(
        "Total Tokens:",
        response.total_tokens,
    )

    print()
    print(
        "✅ SiliconFlow LLM "
        "真实 API 调用正常"
    )


if __name__ == "__main__":
    main()