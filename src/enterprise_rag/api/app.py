"""FastAPI 应用入口。"""

import logging
from collections.abc import (
    AsyncIterator,
)
from contextlib import (
    asynccontextmanager,
)
from time import perf_counter

from fastapi import FastAPI

from enterprise_rag.api.routes.ask import (
    router as ask_router,
)
from enterprise_rag.api.routes.health import (
    router as health_router,
)
from enterprise_rag.api.routes.readiness import (
    router as readiness_router,
)
from enterprise_rag.api.routes.retrieve import (
    router as retrieve_router,
)
from enterprise_rag.runtime.builder import (
    build_rag_runtime,
)


APP_TITLE = (
    "Enterprise RAG Compliance API"
)

APP_DESCRIPTION = (
    "企业 AI 合规与应用规范助手 API"
)

APP_VERSION = "0.1.0"


logger = logging.getLogger(
    "uvicorn.error"
)


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """
    管理完整 RAG Runtime 生命周期。

    Startup：
        创建完整 RAG Runtime；
        记录初始化状态与耗时。

    Shutdown：
        清理 Application State。
    """

    startup_started_at = (
        perf_counter()
    )

    logger.info(
        "Starting RAG runtime..."
    )

    # --------------------------------------------------
    # 在真正构建前先明确状态为空。
    #
    # 即使后面代码扩展，
    # 也不会误把旧状态视为 Ready。
    # --------------------------------------------------

    application.state.rag_runtime = None

    application.state.reranked_retriever = (
        None
    )

    application.state.query_service = (
        None
    )

    try:
        # ==============================================
        # Startup
        # ==============================================

        runtime = build_rag_runtime()

        application.state.rag_runtime = (
            runtime
        )

        application.state.reranked_retriever = (
            runtime.reranked_retriever
        )

        application.state.query_service = (
            runtime.query_service
        )

        startup_elapsed_ms = (
            perf_counter()
            - startup_started_at
        ) * 1000.0

        logger.info(
            (
                "RAG runtime ready | "
                "chunks=%d | "
                "startup_ms=%.2f | "
                "llm_model=%s"
            ),
            len(runtime.chunks),
            startup_elapsed_ms,
            runtime.llm_service.model,
        )

        yield

    except Exception:
        # --------------------------------------------------
        # 不记录 API Key、Prompt、隐私数据。
        #
        # logger.exception 会保留异常栈，
        # 方便本地排查 Startup Failure。
        # --------------------------------------------------

        logger.exception(
            "RAG runtime startup failed"
        )

        raise

    finally:
        # ==============================================
        # Shutdown
        # ==============================================

        logger.info(
            "Shutting down RAG runtime..."
        )

        application.state.query_service = (
            None
        )

        application.state.reranked_retriever = (
            None
        )

        application.state.rag_runtime = (
            None
        )

        logger.info(
            "RAG runtime shutdown complete"
        )


def create_app(
    *,
    use_lifespan: bool = True,
) -> FastAPI:
    """
    创建 FastAPI Application。

    use_lifespan=False：
        供 API Unit Test 使用，
        避免加载真实 GPU / Qdrant / LLM Runtime。
    """

    application = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        lifespan=(
            lifespan
            if use_lifespan
            else None
        ),
    )

    # --------------------------------------------------
    # Infrastructure
    # --------------------------------------------------

    application.include_router(
        health_router
    )

    application.include_router(
        readiness_router
    )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    application.include_router(
        retrieve_router
    )

    # --------------------------------------------------
    # Full RAG Ask
    # --------------------------------------------------

    application.include_router(
        ask_router
    )

    return application


app = create_app()