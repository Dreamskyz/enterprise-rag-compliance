"""知识库文档清单读取模块。"""

from dataclasses import dataclass        #专门用来快速定义“数据对象”的工具
from pathlib import Path                 #Path 是 Python 标准库 pathlib 提供的路径对象

import  yaml                             #把 .yaml 文件转换成 Python 的 dict、list、str、bool 等数据结构


@dataclass(frozen=True)                  #Python的装饰器 decorator,请把下面这个普通类加工成一个适合存数据的dataclass，对象创建以后，不允许修改字段
class DocumentManifest:
    """
    描述一篇知识库原始文档。

    注意：
    这里描述的是“文档级”信息，而不是 Chunk 级信息。
    """

    document_id: str                     #文档唯一 ID
    title: str                           #文档标题
    source_url: str                      #原始来源 URL，保证可追溯性 / provenance
    source_type: str                     #文档来源是什么类型
    document_type: str                   #文档本身是什么类别
    language: str                        #文档语言
    version: str                         #文档版本
    published_at: str | None             #文档发布日期
    effective_at: str | None             #生效日期
    access_level: str                    #Access Control List / 访问控制
    local_path: str                      #本地保存在哪里
    enabled: bool                         #文档是否启用


def load_manifest(manifest_path: Path) -> list[DocumentManifest]:
    """
    从 YAML 文件读取文档清单。

    返回：
        DocumentManifest 列表。

    函数名：load_manifest
    意思：加载文档清单。
    参数：manifest_path: Path
    表示传入：YAML 文件路径。
    返回值：-> list[DocumentManifest]
    表示：返回一个 DocumentManifest 对象组成的列表。
    """

    with manifest_path.open(                    #打开 YAML 文件
        "r",                              #read，只读模式。
        encoding="utf-8",                       #指定 UTF-8 编码
    ) as file:
        data = yaml.safe_load(file)             #safe_load()只解析普通 YAML 数据，不允许 YAML 构造任意 Python 对象，更加安全。
                                                #返回data 通常就是 Python 字典
        documents = data.get("documents", [])   #读取 documents

        return[
            DocumentManifest(**document)        #** 把字典拆成关键字参数
            for document in documents           #列表推导式
        ]