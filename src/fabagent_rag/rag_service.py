from pathlib import Path

from fabagent_rag.chunking import (
    Chunk,
    ChunkConfig,
    batch,
    infer_section_title,
    merge_small_text_chunks,
    split_text,
)
from fabagent_rag.config import Settings
from fabagent_rag.documents import load_document_text
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


DEFAULT_EMBEDDING_BATCH_SIZE = 10


def build_chunk_config(
    settings: Settings,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_size: int | None = None,
) -> ChunkConfig:
    """合并默认配置和请求级覆盖配置。"""

    return ChunkConfig(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        min_chunk_size=min_chunk_size if min_chunk_size is not None else settings.min_chunk_size,
    )


def ingest_path(settings: Settings, path: Path, batch_size: int) -> dict[str, int]:
    """完整单文件入库流程：解析文档 -> 切块 -> 向量化 -> 写入 Milvus。"""

    return ingest_documents(settings, [(str(path), load_document_text(path))], batch_size=batch_size)


def ingest_documents(
    settings: Settings,
    documents: list[tuple[str, str]],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    chunk_config: ChunkConfig | None = None,
) -> dict[str, int]:
    """把已经解析好的文档文本写入 Milvus。

    `source` 由调用方传入：CLI 使用本地路径，上传接口使用原始文件名。
    这样前端上传文件后，检索结果不会暴露服务端临时文件路径。
    """

    # 到这里时，MinerU 已经把复杂文档转换成 Markdown；后续流程统一处理文本。
    chunks = [
        chunk
        for source, text in documents
        for chunk in split_text(text, source, chunk_config or build_chunk_config(settings))
    ]
    return ingest_chunks(settings, chunks, batch_size=batch_size, document_count=len(documents))


def ingest_manual_chunks(
    settings: Settings,
    documents: list[tuple[str, list[str]]],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    chunk_config: ChunkConfig | None = None,
) -> dict[str, int]:
    """把前端人工确认过的 chunk 写入 Milvus。

    手动分块模式下，chunk 边界由用户在前端决定；后端只过滤空文本并重新编号。
    """

    config = chunk_config or build_chunk_config(settings)
    chunks = []
    for source, texts in documents:
        merged_texts = merge_small_text_chunks(texts, config)
        document_text = "\n\n".join(merged_texts)
        chunks.extend(
            Chunk(
                text=text.strip(),
                source=source,
                index=index,
                section_title=infer_section_title(document_text, text.strip()),
            )
            for index, text in enumerate(merged_texts)
            if text.strip()
        )
    return ingest_chunks(settings, chunks, batch_size=batch_size, document_count=len(documents))


def ingest_chunks(
    settings: Settings,
    chunks: list[Chunk],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    document_count: int = 0,
) -> dict[str, int]:
    """统一的 chunk 入库流程，自动分块和手动分块都会走这里。"""

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dimension)

    inserted = 0
    for chunk_batch in batch(chunks, batch_size):
        embeddings = embedder.encode([chunk.text for chunk in chunk_batch])
        inserted += store.insert(chunk_batch, embeddings)

    return {
        "documents": document_count,
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
