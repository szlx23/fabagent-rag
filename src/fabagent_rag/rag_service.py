from pathlib import Path

from fabagent_rag.chunking import batch, split_text
from fabagent_rag.config import Settings
from fabagent_rag.documents import load_documents
from fabagent_rag.embeddings import EmbeddingModel
from fabagent_rag.llm import build_answer
from fabagent_rag.milvus_store import MilvusStore


def build_embedder(settings: Settings) -> EmbeddingModel:
    """根据配置创建 embedding 客户端。"""

    return EmbeddingModel(
        settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_base_url,
    )


def build_store(settings: Settings, dimension: int) -> MilvusStore:
    """根据配置创建 Milvus 访问对象。"""

    return MilvusStore(
        settings.milvus_host,
        settings.milvus_port,
        settings.milvus_collection,
        dimension,
    )


def ingest_path(settings: Settings, path: Path, pattern: str, batch_size: int) -> dict[str, int]:
    """完整入库流程：解析文档 -> 切块 -> 向量化 -> 写入 Milvus。"""

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dimension)

    documents = load_documents(path, pattern)
    # 到这里时，MinerU 已经把复杂文档转换成 Markdown；后续流程统一处理文本。
    chunks = [
        chunk
        for source, text in documents
        for chunk in split_text(text, source, settings.chunk_size, settings.chunk_overlap)
    ]

    inserted = 0
    for chunk_batch in batch(chunks, batch_size):
        embeddings = embedder.encode([chunk.text for chunk in chunk_batch])
        inserted += store.insert(chunk_batch, embeddings)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "inserted": inserted,
    }


def answer_question(settings: Settings, question: str, top_k: int) -> dict[str, object]:
    """完整问答流程：问题向量化 -> Milvus 召回 -> LLM 生成/降级返回上下文。"""

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dimension)
    query_embedding = embedder.encode([question])[0]
    contexts = store.search(query_embedding, top_k=top_k)
    answer = build_answer(
        question,
        contexts,
        settings.inference_api_key,
        settings.inference_base_url,
        settings.inference_model,
    )
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }
