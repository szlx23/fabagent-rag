from openai import OpenAI
from openai import OpenAIError


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
        f"[{index + 1}] {item['source']}#{item['chunk_index']}\n{item['text']}"
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
                    f"来源={item['source']}#{item['chunk_index']}",
                    str(item["text"]),
                ]
            )
        )
    return "\n".join(lines)
