from functools import cached_property
from math import sqrt

from openai import OpenAI


class EmbeddingModel:
    """OpenAI 兼容的 embedding 客户端封装。

    当前 `.env` 指向火山方舟/豆包的 OpenAI 兼容接口。这里把外部服务调用包住，
    让调用方只关心 `encode(texts) -> vectors`。
    """

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
        # Milvus 创建 collection 时必须提前知道向量维度。远程模型 SDK 通常
        # 不暴露维度元数据，所以用一次很小的探测请求得到实际维度并缓存。
        return len(self.encode(["dimension probe"])[0])

    def encode(self, texts: list[str]) -> list[list[float]]:
        """将文本批量转换为向量，并做 L2 归一化。

        Milvus 里使用 IP（内积）作为相似度。向量归一化后，IP 和 cosine similarity
        等价，更适合语义检索。
        """

        if not texts:
            return []

        response = self.client.embeddings.create(model=self.model_name, input=texts)
        # OpenAI 兼容响应带 index。排序后再 zip 回 chunks，避免服务端返回顺序变化
        # 导致“文本 A 写入了文本 B 的向量”。
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise ValueError(f"嵌入接口返回 {len(data)} 条结果，但请求了 {len(texts)} 条文本。")

        embeddings = [item.embedding for item in data]
        return [_normalize(embedding) for embedding in embeddings]


def _normalize(embedding: list[float]) -> list[float]:
    """把向量缩放到单位长度。"""

    norm = sqrt(sum(value * value for value in embedding))
    if norm == 0:
        return embedding
    return [value / norm for value in embedding]
