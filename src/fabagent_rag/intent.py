from typing import Literal


Intent = Literal["lookup", "summarize", "chat"]


SUMMARY_KEYWORDS = (
    "总结",
    "概括",
    "归纳",
    "摘要",
    "梳理",
    "提炼",
    "summarize",
    "summary",
)

CHAT_EXACT_MATCHES = {
    "你好",
    "您好",
    "hi",
    "hello",
    "hey",
    "谢谢",
    "多谢",
    "感谢",
    "辛苦了",
    "早上好",
    "下午好",
    "晚上好",
}

CHAT_KEYWORDS = (
    "你是谁",
    "你叫什么",
    "介绍一下你自己",
    "你能做什么",
    "讲个笑话",
    "聊聊天",
    "随便聊",
)


def detect_intent(question: str) -> Intent:
    """用保守规则识别当前问题的处理路径。

    这里故意不使用 LLM：当前只区分少量意图，规则更便宜也更容易 review。
    默认返回 `lookup`，避免把需要资料库支撑的问题误分到闲聊路径。
    """

    normalized = question.strip().lower()
    if not normalized:
        return "lookup"

    if any(keyword in normalized for keyword in SUMMARY_KEYWORDS):
        return "summarize"

    if normalized in CHAT_EXACT_MATCHES:
        return "chat"

    if any(keyword in normalized for keyword in CHAT_KEYWORDS):
        return "chat"

    return "lookup"
