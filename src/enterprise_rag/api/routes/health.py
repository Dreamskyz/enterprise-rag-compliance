"""服务健康检查接口。"""

from fastapi import APIRouter


router = APIRouter(
    tags=["health"],
)


@router.get(
    "/health",
    summary="服务健康检查",
)
def health_check() -> dict[str, str]:
    """
    返回当前 FastAPI 服务的基本健康状态。

    当前 V1 的 /health 只负责回答：

        “HTTP 服务进程是否正常工作？”

    暂时不在这里执行：

    - Qdrant 网络检查；
    - BGE-M3 模型推理；
    - Reranker 推理；
    - SiliconFlow API 调用。

    原因：

    /health 应该快速、稳定，
    不应该因为外部依赖暂时异常而阻塞。
    """

    return {
        "status": "ok",
        "service": "enterprise-rag-compliance",
        "version": "0.1.0",
    }