from openai import OpenAI
from openai import OpenAIError


def build_answer(
    question: str,
    contexts: list[dict[str, object]],
    api_key: str,
    base_url: str,
    model: str,
) -> str:
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

    if not response.choices:
        return format_contexts(contexts, "推理模型没有返回回答内容：choices 为空")

    content = response.choices[0].message.content
    if not content:
        return format_contexts(contexts, "推理模型返回了空回答")

    return content


def format_contexts(contexts: list[dict[str, object]], reason: str | None = None) -> str:
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
