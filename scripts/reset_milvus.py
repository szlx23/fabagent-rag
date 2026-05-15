from fabagent_rag.config import load_settings
from fabagent_rag.embeddings import EmbeddingModel
from fabagent_rag.keyword_store import KeywordStore
from fabagent_rag.milvus_store import MilvusStore


def main() -> None:
    settings = load_settings()
    embedder = EmbeddingModel(
        settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_base_url,
    )
    store = MilvusStore(
        settings.milvus_host,
        settings.milvus_port,
        settings.milvus_collection,
        embedder.dimension,
    )
    dropped = store.drop_collection()
    print(f"已删除集合 {settings.milvus_collection!r}: {dropped}")
    keyword_dropped = KeywordStore(settings.keyword_index_path).drop_index()
    print(f"已删除关键词索引 {settings.keyword_index_path!r}: {keyword_dropped}")


if __name__ == "__main__":
    main()
