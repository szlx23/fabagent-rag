from pymilvus import (
    DataType,
    MilvusClient,
)

from fabagent_rag.chunking import Chunk


class MilvusStore:
    """Milvus 向量库访问层。

    这个类只负责 collection/schema/index、插入和搜索，不关心文档如何解析、
    embedding 如何生成。这样后续换向量库或调索引参数时，影响范围比较明确。
    """

    def __init__(self, host: str, port: str, collection_name: str, dimension: int) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        self.client = MilvusClient(uri=f"http://{host}:{port}")

    def ensure_collection(self) -> None:
        """确保 Milvus collection 存在并已加载到内存。"""

        if not self.client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("source", DataType.VARCHAR, max_length=1024)
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("text", DataType.VARCHAR, max_length=8192)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                # embedding 已经归一化，所以使用 IP 可以近似 cosine similarity。
                metric_type="IP",
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )

        self.client.load_collection(self.collection_name)

    def insert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """把 chunk 和对应向量写入 Milvus。"""

        self.ensure_collection()
        rows = [
            {
                "source": chunk.source,
                "chunk_index": chunk.index,
                "text": chunk.text[:8192],
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        if not rows:
            return 0
        result = self.client.insert(collection_name=self.collection_name, data=rows)
        self.client.flush(collection_name=self.collection_name)
        return int(result["insert_count"])

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, object]]:
        """按查询向量召回 top_k 个最相似 chunk。"""

        self.ensure_collection()
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["source", "chunk_index", "text"],
            search_params={"metric_type": "IP", "params": {}},
            anns_field="embedding",
        )

        matches: list[dict[str, object]] = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {})
            matches.append(
                {
                    "score": float(hit.get("distance", hit.get("score", 0.0))),
                    "source": entity.get("source"),
                    "chunk_index": entity.get("chunk_index"),
                    "text": entity.get("text"),
                }
            )
        return matches

    def drop_collection(self) -> bool:
        """删除当前 collection，常用于换 embedding 模型后重建索引。"""

        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            return True
        return False
