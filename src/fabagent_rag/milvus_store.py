import json

from pymilvus import (
    DataType,
    MilvusClient,
)

from fabagent_rag.chunking import Chunk, build_chunk_id


REQUIRED_FIELDS = {
    "id",
    "source",
    "page",
    "section_title",
    "file_ext",
    "content_type",
    "sheet_name",
    "parser",
    "chunk_id",
    "ingested_at",
    "text",
    "embedding",
}


class MilvusSchemaError(RuntimeError):
    """当前 Milvus collection schema 和代码要求不一致。"""


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
            schema.add_field("page", DataType.INT64)
            schema.add_field("section_title", DataType.VARCHAR, max_length=1024)
            schema.add_field("file_ext", DataType.VARCHAR, max_length=32)
            schema.add_field("content_type", DataType.VARCHAR, max_length=64)
            schema.add_field("sheet_name", DataType.VARCHAR, max_length=256)
            schema.add_field("parser", DataType.VARCHAR, max_length=64)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=64)
            schema.add_field("ingested_at", DataType.VARCHAR, max_length=64)
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
        else:
            self.validate_collection_schema()

        self.client.load_collection(self.collection_name)

    def insert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """把 chunk 和对应向量写入 Milvus。"""

        self.ensure_collection()
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            row = {
                "source": chunk.source,
                "page": chunk.page or 0,
                "section_title": chunk.section_title[:1024],
                "file_ext": chunk.file_ext[:32],
                "content_type": chunk.content_type[:64],
                "sheet_name": chunk.sheet_name[:256],
                "parser": chunk.parser[:64],
                "chunk_id": (chunk.chunk_id or build_chunk_id(chunk))[:64],
                "ingested_at": chunk.ingested_at[:64],
                "text": chunk.text[:8192],
                "embedding": embedding,
            }
            rows.append(row)

        if not rows:
            return 0
        result = self.client.insert(collection_name=self.collection_name, data=rows)
        self.client.flush(collection_name=self.collection_name)
        return int(result["insert_count"])

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        source_filter: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """按查询向量召回 top_k 个最相似 chunk。"""

        self.ensure_collection()
        output_fields = [
            "source",
            "page",
            "section_title",
            "file_ext",
            "content_type",
            "sheet_name",
            "parser",
            "chunk_id",
            "ingested_at",
            "text",
        ]
        search_kwargs = {
            "collection_name": self.collection_name,
            "data": [query_embedding],
            "limit": top_k,
            "output_fields": output_fields,
            "search_params": {"metric_type": "IP", "params": {}},
            "anns_field": "embedding",
        }
        source_expr = build_source_filter_expr(source_filter)
        if source_expr:
            search_kwargs["filter"] = source_expr

        results = self.client.search(**search_kwargs)

        matches: list[dict[str, object]] = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {})
            matches.append(
                {
                    "score": float(hit.get("distance", hit.get("score", 0.0))),
                    "source": entity.get("source"),
                    "page": normalize_page(entity.get("page")),
                    "section_title": entity.get("section_title") or "",
                    "file_ext": entity.get("file_ext") or "",
                    "content_type": entity.get("content_type") or "",
                    "sheet_name": entity.get("sheet_name") or "",
                    "parser": entity.get("parser") or "",
                    "chunk_id": entity.get("chunk_id") or "",
                    "ingested_at": entity.get("ingested_at") or "",
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

    def validate_collection_schema(self) -> None:
        """检查已存在 collection 是否符合当前 schema。

        项目现在不兼容旧 schema。如果用户在升级 metadata 或混合检索后没有 reset，
        这里会给出明确提示，避免后续 search/insert 抛出难理解的底层 Milvus 错误。
        """

        existing_fields = self.collection_field_names()
        missing_fields = sorted(REQUIRED_FIELDS - existing_fields)
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise MilvusSchemaError(
                f"Milvus collection {self.collection_name!r} 缺少字段：{missing}。"
                "请先执行 `python scripts/reset_milvus.py` 删除旧 collection 和 BM25 索引，"
                "然后重新入库。"
            )

    def collection_field_names(self) -> set[str]:
        """读取 collection 字段名，用于 schema 校验。"""

        description = self.client.describe_collection(self.collection_name)
        fields = description.get("fields", [])
        return {
            str(field.get("name") or field.get("field_name"))
            for field in fields
            if isinstance(field, dict) and (field.get("name") or field.get("field_name"))
        }


def normalize_page(value: object) -> int | None:
    """把 Milvus 中的页码转换为对外 metadata。

    页码未知时内部使用 0，API 返回时转为 None，避免前端展示“第 0 页”。
    """

    if not isinstance(value, int) or value <= 0:
        return None
    return value


def build_source_filter_expr(source_filter: list[str] | None) -> str:
    """构造 Milvus source 过滤表达式。"""

    if not source_filter:
        return ""
    values = ", ".join(json.dumps(source) for source in sorted(set(source_filter)))
    return f"source in [{values}]"
