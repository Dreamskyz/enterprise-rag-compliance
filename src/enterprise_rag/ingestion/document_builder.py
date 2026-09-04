"""构建标准化文档对象。"""

from pathlib import Path

from enterprise_rag.ingestion.loaders.html_loader import (
    extract_cac_article,
    extract_fastapi_article,
    extract_owasp_article,
    extract_qdrant_article,
)
from enterprise_rag.ingestion.manifest import DocumentManifest
from enterprise_rag.ingestion.models import NormalizedDocument
from enterprise_rag.ingestion.normalizer import (
    normalize_structured_text,
    normalize_text,
)


def build_normalized_document(
    manifest: DocumentManifest,
) -> NormalizedDocument:
    """
    根据 Manifest 加载原始文档，
    并构建统一的 NormalizedDocument。

    当前支持三类“语义文档类型”：

    1. regulation

       国家网信办法规 HTML
       -> extract_cac_article()
       -> normalize_text()
       -> 普通标准化文本

    2. security_guideline

       OWASP GenAI Security HTML
       -> extract_owasp_article()
       -> normalize_structured_text()
       -> Markdown-like 结构化文本

    3. technical_documentation

       当前包含两个技术文档来源：

       FastAPI
       -> extract_fastapi_article()

       Qdrant
       -> extract_qdrant_article()

       两者最终都继续：

       -> normalize_structured_text()
       -> Markdown-like 结构化文本

    虽然不同文档：

    - Loader 不同；
    - DOM 结构不同；
    - 数据源特有清洗规则不同；
    - Normalizer 策略可能不同；

    但最终都会转换成统一的：

        NormalizedDocument

    这就是异构文档 ingestion 的统一边界。

    一个重要设计原则：

        document_type

    表示知识的“语义类型”，例如：

        regulation
        security_guideline
        technical_documentation

    它不是：

        HTML
        PDF
        Markdown

    这种文件格式字段。

    同样：

        source_type = official

    表示资料来源/权威性，
    也不应该被用来表达网页格式。

    ------------------------------------------------------
    关于 FastAPI / Qdrant 的 Loader Routing
    ------------------------------------------------------

    当前 Manifest 尚未引入：

        provider
        loader_type

    这样的显式字段。

    因此当前 V1 对：

        technical_documentation

    使用稳定的 document_id 前缀判断：

        fastapi_*
        qdrant_*

    来选择 site-specific Loader。

    这是当前作品级项目中比较简单透明的方案。

    如果未来数据源继续扩展，
    可以进一步把：

        loader_type

    显式放入 Manifest，

    从而避免 document_id 与 Loader
    之间存在隐含约定。
    """

    # Manifest 中保存的是原始文件本地路径。
    path = Path(
        manifest.local_path
    )

    # 原始文件不存在时尽早失败。
    #
    # 这样比后面在 HTML Parser 中出现
    # 难理解的异常更加容易定位问题。
    if not path.exists():
        raise FileNotFoundError(
            f"原始文档不存在：{path}"
        )

    # --------------------------------------------------
    # 1. 法规文档
    # --------------------------------------------------
    if (
        manifest.document_type
        == "regulation"
    ):
        # CAC Loader：
        #
        # 负责从 HTML 中找到
        # BodyLabel 正文区域。
        raw_text = extract_cac_article(
            path
        )

        # 法规的结构主要依靠：
        #
        # 第一章
        # 第一条
        #
        # Regulation Parser 会自行识别这些结构。
        #
        # 因此这里继续使用原来的 normalize_text()，
        # 删除多余空行不会影响法规 Parser。
        #
        # 保持这个行为不变，
        # 也是为了避免破坏现有法规 ingestion。
        normalized_text = normalize_text(
            raw_text
        )

    # --------------------------------------------------
    # 2. OWASP 安全规范
    # --------------------------------------------------
    elif (
        manifest.document_type
        == "security_guideline"
    ):
        # OWASP Loader：
        #
        # 不是简单抽取纯文本，
        # 而是把 HTML Heading 转换成：
        #
        # ## ...
        # ### ...
        # #### ...
        #
        # 同时保留 paragraph / list 等正文。
        raw_text = extract_owasp_article(
            path=path,
            title=manifest.title,
        )

        # OWASP 这类结构化文档，
        # 不仅 Heading 层级有意义，
        # 自然段之间的空行同样有意义。
        #
        # Generic Section Chunker 后续会使用：
        #
        #     \n\n
        #
        # 识别段落边界，
        # 从而优先按自然段组织 Chunk。
        normalized_text = (
            normalize_structured_text(
                raw_text
            )
        )

    # --------------------------------------------------
    # 3. 技术文档
    # --------------------------------------------------
    elif (
        manifest.document_type
        == "technical_documentation"
    ):
        # --------------------------------------------------
        # 3.1 FastAPI
        # --------------------------------------------------
        if manifest.document_id.startswith(
            "fastapi_"
        ):
            # FastAPI Loader：
            #
            # 正文：
            #
            # <article
            #     class="md-content__inner md-typeset"
            # >
            #
            # 并过滤：
            #
            # - 全站导航；
            # - Header anchor；
            # - details 中重复版本示例。
            raw_text = extract_fastapi_article(
                path
            )

        # --------------------------------------------------
        # 3.2 Qdrant
        # --------------------------------------------------
        elif manifest.document_id.startswith(
            "qdrant_"
        ):
            # Qdrant Loader：
            #
            # 正文：
            #
            # <article
            #     class="documentation-article"
            # >
            #
            # 同时执行：
            #
            # 1. 保留 Heading / Paragraph / List；
            # 2. 保留重要 aside 技术提示；
            # 3. 保留 JSON / HTTP / Python 示例；
            # 4. 删除重复的其他 SDK 示例。
            raw_text = extract_qdrant_article(
                path
            )

        # --------------------------------------------------
        # 3.3 technical_documentation 有新来源，
        #     但尚未定义 Loader。
        # --------------------------------------------------
        else:
            raise ValueError(
                "当前 technical_documentation "
                "没有对应 Loader："
                f"{manifest.document_id}"
            )

        # FastAPI / Qdrant 虽然来自不同网站，
        # 但是经过 site-specific Loader 后，
        # 都已经被转换成：
        #
        # Markdown-like structured text
        #
        # 因此继续共享统一 Normalizer。
        #
        # 架构：
        #
        # FastAPI Loader ─┐
        #                 ├─> Structured Text
        # Qdrant Loader ──┘
        #                         ↓
        #              Structured Normalizer
        #                         ↓
        #               Generic Parser
        #                         ↓
        #               Generic Chunker
        normalized_text = (
            normalize_structured_text(
                raw_text
            )
        )

    # --------------------------------------------------
    # 4. 当前版本还不认识的文档类型
    # --------------------------------------------------
    else:
        # 当前采用 fail-fast。
        #
        # 如果 Manifest 中出现一个新的
        # document_type，
        # 但 ingestion pipeline 尚未为它定义
        # Loader / Normalizer 策略，
        # 就直接报错。
        #
        # 不允许系统静默使用错误的默认策略。
        raise ValueError(
            "当前不支持的 document_type："
            f"{manifest.document_type}"
        )

    # --------------------------------------------------
    # 5. 收敛成统一 NormalizedDocument
    # --------------------------------------------------

    # 无论上游是哪种文档：
    #
    # regulation
    # security_guideline
    # technical_documentation
    #
    # 到这一层之后都使用统一的数据模型。
    #
    # 后续 Parser / Chunker 不需要再关心
    # 原始 HTML 网站到底是什么结构。
    return NormalizedDocument(
        document_id=manifest.document_id,
        title=manifest.title,

        document_type=manifest.document_type,
        language=manifest.language,
        version=manifest.version,

        text=normalized_text,

        source_url=manifest.source_url,
        access_level=manifest.access_level,
    )