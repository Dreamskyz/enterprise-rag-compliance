"""LLM 调用相关数据模型。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """
    一次 LLM 文本生成结果。

    content:
        模型最终输出文本。

    model:
        实际调用的模型名称。

    prompt_tokens:
        输入 Token 数。

    completion_tokens:
        输出 Token 数。

    total_tokens:
        总 Token 数。

    当前 V1 先保留最核心的可观测字段。
    """

    content: str

    model: str

    prompt_tokens: int | None

    completion_tokens: int | None

    total_tokens: int | None