from functools import cached_property
from math import sqrt

from openai import OpenAI


class EmbeddingModel:
    def __init__(self, model_name: str, api_key: str, base_url: str) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @cached_property
    def client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("未配置 EMBEDDING_API_KEY 或 ARK_API_KEY，无法调用嵌入模型。")
        if not self.base_url:
            raise ValueError("未配置 EMBEDDING_BASE_URL，无法调用嵌入模型。")
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    @cached_property
    def dimension(self) -> int:
        return len(self.encode(["dimension probe"])[0])

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(model=self.model_name, input=texts)
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise ValueError(f"嵌入接口返回 {len(data)} 条结果，但请求了 {len(texts)} 条文本。")

        embeddings = [item.embedding for item in data]
        return [_normalize(embedding) for embedding in embeddings]


def _normalize(embedding: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding
    return [value / norm for value in embedding]
