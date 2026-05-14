import json

from openai import OpenAI
from openai import OpenAIError

from fabagent_rag.intent import INTENT_VALUES, Intent


def classify_intent_with_llm(
    question: str,
    api_key: str,
    base_url: str,
    model: str,
) -> Intent | None:
    """优先用 LLM 判断用户问题应该走哪条路径。

    这一步只允许模型在三种意图里选择，不让它直接生成业务回答。解析失败或模型不可用时
    返回 None，由调用方使用规则兜底。
    """

    if not api_key or not base_url or not model:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你只负责判断用户问题在 FabAgent RAG 中应该走哪条路径。"
                        "只能返回 JSON，不要返回 Markdown。"
                        "可选 intent："
                        "lookup=需要查询资料库回答；"
                        "summarize=需要基于资料库总结归纳；"
                        "chat=普通闲聊或助手自我介绍，不需要资料库。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{question}\n"
                        '请只返回：{"intent":"lookup|summarize|chat"}'
                    ),
                },
            ],
            temperature=0,
        )
    except OpenAIError:
        return None

    if not response.choices:
        return None

    content = response.choices[0].message.content or ""
    return parse_intent_json(content)


def parse_intent_json(content: str) -> Intent | None:
    """解析 LLM 返回的最小 JSON，兼容模型偶尔包一层说明文本的情况。"""

    stripped = content.strip()
    if not stripped:
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None

    intent = str(payload.get("intent") or "").strip().lower()
    if intent in INTENT_VALUES:
        return intent  # type: ignore[return-value]
    return None


def build_chat_answer(
    question: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """不经过知识库检索，直接用推理模型处理闲聊类问题。"""

    if not api_key or not base_url or not model:
        return "当前没有完整配置推理模型，暂时无法处理闲聊问题。"

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 FabAgent RAG 的助手。"
                        "当前问题被识别为闲聊，不需要检索资料库。"
                        "请简洁回答；如果用户提出实时信息或专业事实查询，提醒用户改为基于资料库提问。"
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.5,
        )
    except OpenAIError as exc:
        return f"推理模型调用失败：{exc}"

    if not response.choices:
        return "推理模型没有返回回答内容：choices 为空"

    content = response.choices[0].message.content
    if not content:
        return "推理模型返回了空回答"

    return content


def build_answer(
    question: str,
    contexts: list[dict[str, object]],
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """基于检索上下文生成最终回答。

    推理模型是可选能力：如果没有配置，或者接口失败，就返回检索上下文本身。
    这样 RAG 的“检索”能力和“生成”能力可以独立排障。
    """

    if not api_key or not base_url or not model:
        return format_contexts(contexts)

    client = OpenAI(api_key=api_key, base_url=base_url)
    context_text = "\n\n".join(
        f"[{index + 1}] {format_source_location(item)}\n{item['text']}"
        for index, item in enumerate(contexts)
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "只能根据提供的上下文回答问题。"
                        "如果上下文不足，请说明还缺少哪些信息。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n上下文：\n{context_text}",
                },
            ],
            temperature=0.2,
        )
    except OpenAIError as exc:
        return format_contexts(contexts, f"推理模型调用失败：{exc}")

    # 一些 OpenAI 兼容接口会返回 HTTP 200 但 choices 为空。这里显式兜底，
    # 避免用户只看到 IndexError，而看不到已经检索到的上下文。
    if not response.choices:
        return format_contexts(contexts, "推理模型没有返回回答内容：choices 为空")

    content = response.choices[0].message.content
    if not content:
        return format_contexts(contexts, "推理模型返回了空回答")

    return content


def format_contexts(contexts: list[dict[str, object]], reason: str | None = None) -> str:
    """把检索结果格式化成人类可读文本，作为无生成模型时的降级输出。"""

    if not contexts:
        return "没有找到匹配的上下文。"

    if reason:
        lines = [f"{reason}，以下是检索到的上下文："]
    else:
        lines = ["未完整配置推理模型，以下是检索到的上下文："]
    for index, item in enumerate(contexts, start=1):
        lines.append(
            "\n".join(
                [
                    f"\n[{index}] 相似度={item['score']:.4f}",
                    f"来源={format_source_location(item)}",
                    str(item["text"]),
                ]
            )
        )
    return "\n".join(lines)


def format_source_location(item: dict[str, object]) -> str:
    """把 metadata 格式化为员工可读的位置描述。"""

    parts = [str(item.get("source") or "未知来源")]
    page = item.get("page")
    if isinstance(page, int):
        parts.append(f"第 {page} 页")
    section_title = str(item.get("section_title") or "").strip()
    if section_title:
        parts.append(section_title)
    return " / ".join(parts)
