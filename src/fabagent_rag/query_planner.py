from dataclasses import dataclass
import json

from openai import OpenAI
from openai import OpenAIError

from fabagent_rag.intent import Intent


@dataclass(frozen=True)
class QueryPlan:
    """检索前的查询计划。

    `original_query` 保留用户原问题，避免重写失败时丢失真实意图。
    `rewritten_query` 是更适合向量检索的短查询。
    `expanded_queries` 用来覆盖同义词、中文/英文术语、行业写法差异。
    """

    original_query: str
    rewritten_query: str
    expanded_queries: list[str]

    def queries(self) -> list[str]:
        """返回实际用于检索的 query 列表，并按文本去重。"""

        queries = [self.original_query, self.rewritten_query, *self.expanded_queries]
        unique_queries = []
        seen = set()
        for query in queries:
            normalized = query.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                unique_queries.append(normalized)
                seen.add(key)
        return unique_queries

    def to_dict(self) -> dict[str, object]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "expanded_queries": self.expanded_queries,
            "queries": self.queries(),
        }


def fallback_query_plan(question: str) -> QueryPlan:
    """LLM 不可用或返回异常时，只用原问题检索。"""

    normalized = question.strip()
    return QueryPlan(
        original_query=normalized,
        rewritten_query=normalized,
        expanded_queries=[],
    )


def build_query_plan(
    question: str,
    intent: Intent,
    api_key: str,
    base_url: str,
    model: str,
) -> QueryPlan:
    """在确认需要 RAG 后，让 LLM 生成更适合检索的 query。

    这一步不回答问题，只规划“怎么搜”。如果 LLM 不可用，返回只包含原问题的计划，
    保证检索链路仍可工作。
    """

    fallback = fallback_query_plan(question)
    if not api_key or not base_url or not model:
        return fallback

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你只负责为 RAG 检索生成查询计划，不回答用户问题。"
                        "只能返回 JSON，不要返回 Markdown。"
                        "rewritten_query 应该短、清晰、适合向量检索。"
                        "lookup 最多给 3 条 expanded_queries；"
                        "summarize 通常不扩写，最多给 1 条。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"intent：{intent}\n"
                        f"用户问题：{question}\n"
                        "请返回 JSON："
                        '{"rewritten_query":"...", "expanded_queries":["..."]}'
                    ),
                },
            ],
            temperature=0.1,
        )
    except OpenAIError:
        return fallback

    if not response.choices:
        return fallback

    content = response.choices[0].message.content or ""
    return parse_query_plan_json(content, fallback, intent)


def parse_query_plan_json(content: str, fallback: QueryPlan, intent: Intent) -> QueryPlan:
    """解析 LLM 生成的 Query Plan，并强制收敛扩写数量。"""

    payload = parse_json_object(content)
    if not payload:
        return fallback

    rewritten_query = clean_query(payload.get("rewritten_query")) or fallback.rewritten_query
    expanded_queries = clean_expanded_queries(
        payload.get("expanded_queries"),
        excluded={fallback.original_query.lower(), rewritten_query.lower()},
        limit=3 if intent == "lookup" else 1,
    )
    return QueryPlan(
        original_query=fallback.original_query,
        rewritten_query=rewritten_query,
        expanded_queries=expanded_queries,
    )


def parse_json_object(content: str) -> dict[str, object] | None:
    """从模型返回内容中提取 JSON object。"""

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
    return payload


def clean_query(value: object) -> str:
    """把 LLM 返回的 query 清洗成单行文本。"""

    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def clean_expanded_queries(
    value: object,
    excluded: set[str],
    limit: int,
) -> list[str]:
    """清洗扩写 query，过滤空值、重复值和已经存在的 query。"""

    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []

    expanded_queries = []
    seen = set(excluded)
    for item in candidates:
        query = clean_query(item)
        key = query.lower()
        if not query or key in seen:
            continue
        expanded_queries.append(query)
        seen.add(key)
        if len(expanded_queries) >= limit:
            break
    return expanded_queries
