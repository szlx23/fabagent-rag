from fabagent_rag.config import load_settings
from fabagent_rag.keyword_store import KeywordStore
from fabagent_rag.milvus_store import MilvusStore


def main() -> None:
    settings = load_settings()
    store = MilvusStore(
        settings.milvus_host,
        settings.milvus_port,
        settings.milvus_collection,
        # 删除 collection 不需要真实 embedding 维度；这里传 1 避免 reset 时调用外部
        # embedding 服务，保证离线也能重置本地索引。
        dimension=1,
    )
    dropped = store.drop_collection()
    print(f"已删除集合 {settings.milvus_collection!r}: {dropped}")
    keyword_dropped = KeywordStore(settings.keyword_index_path).drop_index()
    print(f"已删除关键词索引 {settings.keyword_index_path!r}: {keyword_dropped}")


if __name__ == "__main__":
    main()
