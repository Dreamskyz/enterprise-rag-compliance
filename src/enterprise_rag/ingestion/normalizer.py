"""文档文本标准化模块。"""

import re                   #re 是 Python 内置的正则表达式模块
                            #这个文件里主要用它处理line = re.sub(r"[ \t]+", " ", line)
                            #把连续多个普通空格或 Tab 压缩成一个空格
def normalize_text(text: str) -> str:
    """
    对已经抽取出的正文文本进行基础标准化。

    只处理“格式问题”，不负责判断网页哪些区域属于正文。

    当前规则：
    1. 统一换行符；
    2. 清理行首行尾空白；
    3. 删除连续空行；
    4. 统一中文文本中常见的特殊空格。
    """

    # Windows 换行和旧式换行统一为 \n。
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 全角空格 / 不换行空格等统一成普通空格。
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")

    lines: list[str] = []           #创建一个空列表,以后用来保存清洗后的有效文本行

    for raw_line in text.splitlines():           #按行遍历文本,text.splitlines()按换行符把字符串拆成多行
        # 清理每一行两端空白。
        line = raw_line.strip()                  #strip()删除字符串开头和结尾的空白字符

        # 行内部连续的空白压缩成一个普通空格。
        line = re.sub(r"[ \t]+", " ", line)    #re.sub(要找什么, 替换成什么, 在哪个字符串中找),在 line 中寻找连续的空格或 Tab，把它们替换成一个普通空格
                                                           #[ \t]+表示一个或多个连续的空格 / Tab

        if line:                                           #判断是不是空行
            lines.append(line)                             #保存有效行

    return "\n".join(lines)                                #把所有行重新拼起来