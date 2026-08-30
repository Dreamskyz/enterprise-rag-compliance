"""HTML 原始文件加载器。"""

from pathlib import Path                                        #从 Python 标准库 pathlib 中导入 Path

from bs4 import BeautifulSoup                                   #HTML/XML 解析库
                                                                #去除大量HTML标签

def load_html(path: Path) -> str:
    """
    读取完整 HTML 页面中的可见文本。

    主要用于调试。
    正式 ingestion 不应直接使用这个结果，
    因为其中通常包含导航栏、页脚等网页噪声。
    """

    html = path.read_text(encoding='utf-8')                    #读取整个文件，得到html是一个普通 Python 字符串

    soup = BeautifulSoup(html, 'html.parser')          #"html.parser"表示使用 Python 内置 HTML 解析器，执行后的soup变成一个BeautifulSoup 对象，可以理解成一棵 HTML 树

    # script/style 内容不属于知识正文，应提前移除。                  数据清洗
    for tag in soup(["script", "style"]):                      #可以理解为soup.find_all(["script", "style"])，找出 HTML 中所有 <script> 和 <style> 标签
        tag.decompose()                                        #decompose()把这个标签以及里面的全部内容，从 BeautifulSoup 树中彻底删除。
                                                               #提取文本
    return soup.get_text(                                      #get_text() 去掉 HTML 标签，只保留其中的文本内容
        separator="\n",                                        #不同 HTML 文本节点之间，用换行分隔
        strip=True,                                            #去掉每段文本首尾多余的空格和换行
    )


def extract_cac_article(path: Path) -> str:
    """
    从中国网信网页面提取正式文章正文。

    国家网信办当前文章页面的正文位于：

        <div id="BodyLabel">...</div>

    因此直接基于 DOM 容器抽取，
    避免把页头、页尾、导航栏、二维码等内容混入知识库。
    """

    html = path.read_text(encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")

    # 找到文章正文容器。
    article = soup.find(id="BodyLabel")             #在整棵 HTML 树中寻找 id="BodyLabel" 的第一个元素
                                                    #article得到的不是字符串,通常是 BeautifulSoup 的Tag对象
    if article is None:
        raise ValueError(
            f"未找到国家网信办正文容器 BodyLabel：{path}"
        )

    # 即使正文容器里意外出现 script/style，也先清掉。
    for tag in article(["script", "style"]):
        tag.decompose()

    return article.get_text(
        separator="\n",
        strip=True,
    )