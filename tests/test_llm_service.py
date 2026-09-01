"""测试 SiliconFlow LLM Service 的基础逻辑。"""

from types import SimpleNamespace

import pytest

from enterprise_rag.llm.siliconflow import (
    SiliconFlowLLMService,
)


class FakeCompletions:
    """
    模拟：

        client.chat.completions.create()
    """

    def __init__(self) -> None:
        self.last_kwargs = None

    def create(
        self,
        **kwargs,
    ):
        self.last_kwargs = kwargs

        return SimpleNamespace(
            model=(
                "deepseek-ai/"
                "DeepSeek-V4-Flash"
            ),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="测试回答"
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class FakeClient:
    """模拟 OpenAI Client。"""

    def __init__(self) -> None:
        self.completions = (
            FakeCompletions()
        )

        self.chat = SimpleNamespace(
            completions=self.completions
        )


def build_service(
) -> SiliconFlowLLMService:
    """
    创建测试 Service。

    显式传入假 API Key，
    不依赖真实 .env。
    """

    service = SiliconFlowLLMService(
        api_key="test-key",
        model=(
            "deepseek-ai/"
            "DeepSeek-V4-Flash"
        ),
    )

    # 用 Fake Client 替换真实网络 Client。
    service.client = FakeClient()

    return service


def test_generate_returns_llm_response() -> None:
    """正常生成应返回项目自己的 Response。"""

    service = build_service()

    response = service.generate(
        messages=[
            {
                "role": "user",
                "content": "你好",
            }
        ],
        temperature=0.1,
        max_tokens=100,
    )

    assert response.content == (
        "测试回答"
    )

    assert response.model == (
        "deepseek-ai/"
        "DeepSeek-V4-Flash"
    )

    assert response.prompt_tokens == 10

    assert (
        response.completion_tokens
        == 5
    )

    assert response.total_tokens == 15


def test_generate_passes_expected_parameters() -> None:
    """调用参数应正确传给 SDK。"""

    service = build_service()

    service.generate(
        messages=[
            {
                "role": "user",
                "content": "测试问题",
            }
        ],
        temperature=0.2,
        max_tokens=256,
    )

    fake_client = service.client

    kwargs = (
        fake_client
        .completions
        .last_kwargs
    )

    assert kwargs is not None

    assert kwargs["model"] == (
        "deepseek-ai/"
        "DeepSeek-V4-Flash"
    )

    assert kwargs[
        "temperature"
    ] == 0.2

    assert kwargs[
        "max_tokens"
    ] == 256

    assert kwargs[
        "stream"
    ] is False


def test_generate_rejects_empty_messages() -> None:
    """messages 不能为空。"""

    service = build_service()

    with pytest.raises(
        ValueError,
        match="messages",
    ):
        service.generate(
            messages=[]
        )


def test_generate_rejects_blank_content() -> None:
    """message content 不能为空。"""

    service = build_service()

    with pytest.raises(
        ValueError,
        match="content",
    ):
        service.generate(
            messages=[
                {
                    "role": "user",
                    "content": "   ",
                }
            ]
        )


def test_generate_rejects_invalid_temperature() -> None:
    """temperature 必须在合法范围。"""

    service = build_service()

    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        service.generate(
            messages=[
                {
                    "role": "user",
                    "content": "测试",
                }
            ],
            temperature=3.0,
        )


def test_generate_rejects_invalid_max_tokens() -> None:
    """max_tokens 必须大于 0。"""

    service = build_service()

    with pytest.raises(
        ValueError,
        match="max_tokens",
    ):
        service.generate(
            messages=[
                {
                    "role": "user",
                    "content": "测试",
                }
            ],
            max_tokens=0,
        )