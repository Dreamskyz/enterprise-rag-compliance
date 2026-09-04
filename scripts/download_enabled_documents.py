"""根据 Document Manifest 下载启用的远程文档到本地 Raw Corpus。"""

import argparse
from pathlib import Path

import httpx

from enterprise_rag.ingestion.manifest import (
    DocumentManifest,
    load_manifest,
)


# ==========================================================
# Manifest 路径。
# ==========================================================

MANIFEST_PATH = Path(
    "data/manifest/documents.yaml"
)


# ==========================================================
# HTTP 下载配置。
#
# timeout:
#     单次请求最长等待时间。
#
# follow_redirects:
#     官方网站有时会执行 301 / 302 跳转，
#     因此需要允许跟随重定向。
#
# Headers:
#
#     早期版本 Downloader 使用：
#
#         enterprise-rag-compliance/document-downloader
#
#     作为 User-Agent。
#
#     该方式能够正常下载：
#
#         OWASP
#         FastAPI
#         Qdrant
#
#     但在扩展中国法规 Corpus 时，
#     实测部分中国网信办旧页面会返回：
#
#         HTTP 403 Forbidden
#
#     同一 URL 在加入普通浏览器式请求头后：
#
#         status = 200
#
#     因此这里使用常规浏览器 HTTP Header，
#     提高公开文档下载的站点兼容性。
#
#     注意：
#
#     这里没有加入：
#
#         Cookie
#         登录凭据
#         Token
#         JS Challenge 绕过
#
#     也没有关闭 TLS 校验。
#
#     它仍然只是一个用于下载
#     公开官方文档的普通 HTTP Client。
# ==========================================================

HTTP_TIMEOUT_SECONDS = 30.0

DEFAULT_HEADERS = {
    # 使用正常浏览器 User-Agent。
    #
    # 当前不是为了冒充某个登录用户，
    # 而是为了避免部分公开网站直接拒绝
    # 非浏览器式默认请求。
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),

    # 明确告诉服务器：
    # 当前希望获取 HTML / XHTML 等网页正文。
    "Accept": (
        "text/html,"
        "application/xhtml+xml,"
        "application/xml;q=0.9,"
        "*/*;q=0.8"
    ),

    # 当前知识库以中文法规为重要数据源，
    # 因此优先中文响应。
    #
    # 对 OWASP / FastAPI / Qdrant 等英文网站，
    # 英文仍然属于允许语言。
    "Accept-Language": (
        "zh-CN,zh;q=0.9,en;q=0.8"
    ),
}


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    默认行为：

        只下载 enabled=true
        且本地文件不存在的文档。

    --force：

        即使 local_path 已经存在，
        也重新下载并覆盖。

    --document-id：

        只处理指定 document_id。

        可以重复传入多次，例如：

            python scripts/download_enabled_documents.py \
                --document-id qdrant_payload \
                --document-id qdrant_filtering

    这个参数非常适合逐批扩展 Corpus：

        先下载少量文档
        ↓
        检查 HTML / Loader
        ↓
        再批量下载剩余文档
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download enabled documents "
            "from the knowledge-base manifest."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "重新下载并覆盖已经存在的 "
            "local_path 文件。"
        ),
    )

    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help=(
            "只下载指定 document_id。"
            "可以重复传入多次。"
        ),
    )

    return parser.parse_args()


def should_download_document(
    manifest: DocumentManifest,
    selected_document_ids: set[str],
) -> bool:
    """
    判断当前 Manifest 是否属于本次下载目标。

    第一层：

        enabled 必须为 true。

    第二层：

        如果命令行没有传 --document-id，
        则处理全部 enabled 文档。

        如果传了，
        则只处理指定 document_id。
    """

    if not manifest.enabled:
        return False

    if not selected_document_ids:
        return True

    return (
        manifest.document_id
        in selected_document_ids
    )


def validate_source_url(
    manifest: DocumentManifest,
) -> None:
    """
    对 source_url 做最基本的输入检查。

    当前 Downloader 只负责公开 HTTP / HTTPS 文档。

    不接受：

        file://
        ftp://
        javascript:

    等其他 scheme。

    这里不是复杂的安全网关，
    只是避免 Manifest 中明显错误的 URL
    被交给 HTTP Client。
    """

    source_url = manifest.source_url.strip()

    if not (
        source_url.startswith("https://")
        or source_url.startswith("http://")
    ):
        raise ValueError(
            "Document source_url 必须使用 "
            "http:// 或 https://："
            f"{manifest.document_id} -> "
            f"{manifest.source_url}"
        )


def download_document(
    *,
    client: httpx.Client,
    manifest: DocumentManifest,
    force: bool,
) -> str:
    """
    下载单篇文档。

    返回值用于最终统计：

        downloaded
        skipped

    下载流程：

        source_url
        ↓
        HTTP GET
        ↓
        raise_for_status()
        ↓
        检查响应非空
        ↓
        创建 local_path 父目录
        ↓
        临时文件
        ↓
        原子替换 Raw File

    ------------------------------------------------------
    为什么保存 response.content，而不是 response.text？
    ------------------------------------------------------

    Raw Corpus 的职责应该尽量保存：

        “服务器实际返回的原始字节”

    而不是在下载阶段先进行文本编码转换。

    HTML 的：

        Decode
        Parse
        Normalize

    应该继续交给 Ingestion Loader。
    """

    local_path = Path(
        manifest.local_path
    )

    # ------------------------------------------------------
    # 默认不覆盖已有 Raw Snapshot。
    #
    # 这是为了保证：
    #
    #     已经审计过的 Corpus
    #
    # 不会因为官网后来发生变化，
    # 在一次普通下载命令中被悄悄替换。
    #
    # 如果确实要刷新快照，
    # 用户必须显式使用：
    #
    #     --force
    # ------------------------------------------------------

    if (
        local_path.exists()
        and not force
    ):
        print(
            "[SKIP] "
            f"{manifest.document_id}"
            " -> "
            f"{local_path}"
        )

        return "skipped"

    # 在真正发起 HTTP 请求之前，
    # 先检查 URL Scheme。
    validate_source_url(
        manifest
    )

    print(
        "[GET ] "
        f"{manifest.document_id}"
    )

    print(
        "       "
        f"{manifest.source_url}"
    )

    # ------------------------------------------------------
    # Headers 已经统一配置在共享 httpx.Client 中。
    #
    # 因此这里不再针对：
    #
    #     CAC
    #     OWASP
    #     FastAPI
    #     Qdrant
    #
    # 写不同的 client.get() 分支。
    #
    # 当前实测浏览器式通用 Header
    # 可以兼容这些公开官方站点，
    # 所以优先保持 Downloader 简单。
    # ------------------------------------------------------

    response = client.get(
        manifest.source_url
    )

    # ------------------------------------------------------
    # 4xx / 5xx 直接失败。
    #
    # 不允许把：
    #
    #     403 页面
    #     404 页面
    #     500 页面
    #
    # 当成正常 Corpus 保存。
    # ------------------------------------------------------

    response.raise_for_status()

    content = response.content

    if not content:
        raise ValueError(
            "下载结果为空："
            f"{manifest.document_id}"
        )

    # ------------------------------------------------------
    # local_path 的目录如果还不存在，
    # 自动创建。
    # ------------------------------------------------------

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # 先写临时文件，再 replace。
    #
    # 好处：
    #
    # 如果程序在写文件过程中异常退出，
    # 不容易留下一个被截断的正式 Raw File。
    #
    # 举例：
    #
    #     document.html.tmp
    #              ↓
    #     写入成功
    #              ↓
    #     replace()
    #              ↓
    #     document.html
    #
    # 这比直接写正式文件更加稳妥。
    # ------------------------------------------------------

    temporary_path = (
        local_path.with_suffix(
            local_path.suffix + ".tmp"
        )
    )

    temporary_path.write_bytes(
        content
    )

    temporary_path.replace(
        local_path
    )

    print(
        "[SAVE] "
        f"{manifest.document_id}"
        " -> "
        f"{local_path}"
        " "
        f"({len(content)} bytes)"
    )

    return "downloaded"


def validate_selected_document_ids(
    *,
    manifests: list[DocumentManifest],
    selected_document_ids: set[str],
) -> None:
    """
    如果用户显式传入 --document-id，
    检查这些 ID 是否真的存在于 Manifest。

    为什么 fail-fast：

    如果手滑写成：

        owasp_llm002...

    最糟糕的行为不是报错，
    而是：

        “什么都没下载，但脚本成功退出。”

    因此这里直接提示未知 ID。
    """

    if not selected_document_ids:
        return

    known_ids = {
        manifest.document_id
        for manifest in manifests
    }

    unknown_ids = (
        selected_document_ids
        - known_ids
    )

    if unknown_ids:
        unknown_text = ", ".join(
            sorted(unknown_ids)
        )

        raise ValueError(
            "以下 document_id "
            "不存在于 Manifest："
            f"{unknown_text}"
        )


def main() -> None:
    """
    Downloader 主入口。
    """

    args = parse_args()

    selected_document_ids = set(
        args.document_id
    )

    # ======================================================
    # 1. 读取 Manifest。
    # ======================================================

    manifests = load_manifest(
        MANIFEST_PATH
    )

    validate_selected_document_ids(
        manifests=manifests,
        selected_document_ids=(
            selected_document_ids
        ),
    )

    # ======================================================
    # 2. 过滤本次真正要处理的文档。
    # ======================================================

    target_manifests = [
        manifest
        for manifest in manifests
        if should_download_document(
            manifest,
            selected_document_ids,
        )
    ]

    print("=" * 100)
    print(
        "Document Downloader"
    )
    print("=" * 100)

    print(
        "Manifest:",
        MANIFEST_PATH,
    )

    print(
        "Total manifests:",
        len(manifests),
    )

    print(
        "Download targets:",
        len(target_manifests),
    )

    print(
        "Force overwrite:",
        bool(args.force),
    )

    if selected_document_ids:
        print(
            "Selected document IDs:",
            ", ".join(
                sorted(
                    selected_document_ids
                )
            ),
        )

    print()

    # ======================================================
    # 3. 共享一个 HTTP Client。
    #
    # 不为每篇文档重新创建连接池。
    #
    # Client 统一负责：
    #
    # - timeout；
    # - redirect；
    # - 浏览器式公共请求头。
    # ======================================================

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    with httpx.Client(
        timeout=HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers=DEFAULT_HEADERS,
    ) as client:
        for manifest in target_manifests:
            try:
                result = download_document(
                    client=client,
                    manifest=manifest,
                    force=bool(args.force),
                )

                if result == "downloaded":
                    downloaded_count += 1

                elif result == "skipped":
                    skipped_count += 1

            except Exception as exc:
                # ------------------------------------------
                # 一篇文档失败，不阻断剩余文档下载。
                #
                # 批量下载 20~30 篇时，
                # 单个站点临时失败不应该让整个 Batch
                # 完全失去结果。
                #
                # 最后仍通过 failed_count 明确告诉用户
                # 当前 Batch 并非完全成功。
                # ------------------------------------------

                failed_count += 1

                print(
                    "[FAIL] "
                    f"{manifest.document_id}"
                )

                print(
                    "       "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print()

    # ======================================================
    # 4. Batch Summary。
    # ======================================================

    print()
    print("=" * 100)
    print(
        "Download Summary"
    )
    print("=" * 100)

    print(
        "Targets:",
        len(target_manifests),
    )

    print(
        "Downloaded:",
        downloaded_count,
    )

    print(
        "Skipped:",
        skipped_count,
    )

    print(
        "Failed:",
        failed_count,
    )

    # ------------------------------------------------------
    # 如果存在失败文档，
    # 让脚本以异常结束。
    #
    # 原因：
    #
    # 对人工查看来说，
    # 上面的 Summary 已经很清楚；
    #
    # 对未来 CI / 自动化流程来说，
    # exit code 也应该体现：
    #
    #     Batch 并未完全成功。
    # ------------------------------------------------------

    if failed_count > 0:
        raise RuntimeError(
            f"{failed_count} document(s) "
            "failed to download."
        )


if __name__ == "__main__":
    main()