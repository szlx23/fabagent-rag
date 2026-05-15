from pathlib import Path
from datetime import UTC, datetime

from fabagent_rag.chunking import (
    Chunk,
    ChunkConfig,
    batch,
    detect_content_type,
    infer_section_title,
    merge_small_text_chunks,
    split_text,
)
from fabagent_rag.config import Settings
from fabagent_rag.documents import ParsedDocument, parse_document
from fabagent_rag.embeddings import EmbeddingModel
from fabagent_rag.intent import detect_intent
from fabagent_rag.llm import build_answer, build_chat_answer, classify_intent_with_llm
from fabagent_rag.milvus_store import MilvusStore
from fabagent_rag.query_planner import QueryPlan, build_query_plan


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

    return ingest_documents(
        settings,
        [parse_document(path, str(path), mineru_backend=settings.mineru_backend)],
        batch_size=batch_size,
    )


def ingest_documents(
    settings: Settings,
    documents: list[ParsedDocument | tuple[str, str]],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    chunk_config: ChunkConfig | None = None,
) -> dict[str, int]:
    """把已经解析好的文档文本写入 Milvus。

    `source` 由调用方传入：CLI 使用本地路径，上传接口使用原始文件名。
    这样前端上传文件后，检索结果不会暴露服务端临时文件路径。
    """

    # 到这里时，MinerU 已经把复杂文档转换成 Markdown；后续流程统一处理文本。
    parsed_documents = normalize_documents(documents)
    chunks = [
        chunk
        for document in parsed_documents
        for chunk in split_text(
            document.text,
            document.source,
            chunk_config or build_chunk_config(settings),
            metadata=enrich_document_metadata(document),
        )
    ]
    return ingest_chunks(
        settings,
        chunks,
        batch_size=batch_size,
        document_count=len(parsed_documents),
    )


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
    ingested_at = current_ingested_at()
    chunks = []
    for source, texts in documents:
        source_path = Path(source)
        merged_texts = merge_small_text_chunks(texts, config)
        document_text = "\n\n".join(merged_texts)
        chunks.extend(
            Chunk(
                text=text.strip(),
                source=source,
                index=index,
                section_title=infer_section_title(document_text, text.strip()),
                file_ext=source_path.suffix.lower(),
                content_type=detect_content_type(text.strip()),
                parser="manual",
                ingested_at=ingested_at,
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
    """完整问答流程：LLM 优先判断意图，失败时用规则兜底。"""

    intent = classify_intent_with_llm(
        question,
        settings.inference_api_key,
        settings.inference_base_url,
        settings.inference_model,
    ) or detect_intent(question)

    # 最终 intent 为 chat 时不访问 Milvus，避免把无关上下文带进闲聊回答。
    if intent == "chat":
        answer = build_chat_answer(
            question,
            settings.inference_api_key,
            settings.inference_base_url,
            settings.inference_model,
        )
        return {
            "question": question,
            "intent": intent,
            "query_plan": None,
            "answer": answer,
            "contexts": [],
        }

    query_plan = build_query_plan(
        question,
        intent,
        settings.inference_api_key,
        settings.inference_base_url,
        settings.inference_model,
    )
    contexts = search_contexts(settings, query_plan, top_k)
    answer = build_answer(
        question,
        contexts,
        settings.inference_api_key,
        settings.inference_base_url,
        settings.inference_model,
    )
    return {
        "question": question,
        "intent": intent,
        "query_plan": query_plan.to_dict(),
        "answer": answer,
        "contexts": contexts,
    }


def search_contexts(
    settings: Settings,
    query_plan: QueryPlan,
    top_k: int,
) -> list[dict[str, object]]:
    """使用 Query Plan 中的多个 query 召回上下文，并合并去重。"""

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dimension)
    queries = query_plan.queries()
    embeddings = embedder.encode(queries)

    candidates: dict[tuple[object, object, object, object], dict[str, object]] = {}
    for query, embedding in zip(queries, embeddings, strict=True):
        for context in store.search(embedding, top_k=top_k):
            context_with_query = {**context, "matched_query": query}
            key = (
                context_with_query.get("source"),
                context_with_query.get("page"),
                context_with_query.get("section_title"),
                context_with_query.get("text"),
            )
            existing = candidates.get(key)
            if existing is None or score_of(context_with_query) > score_of(existing):
                candidates[key] = context_with_query

    return sorted(candidates.values(), key=score_of, reverse=True)[:top_k]


def score_of(context: dict[str, object]) -> float:
    """读取召回结果分数，用于多 query 结果合并排序。"""

    score = context.get("score")
    if isinstance(score, int | float):
        return float(score)
    return 0.0


def normalize_documents(documents: list[ParsedDocument | tuple[str, str]]) -> list[ParsedDocument]:
    """兼容旧的 `(source, text)` 调用形式，并统一成 ParsedDocument。"""

    normalized = []
    for document in documents:
        if isinstance(document, ParsedDocument):
            normalized.append(document)
        else:
            source, text = document
            normalized.append(ParsedDocument(source=source, text=text))
    return normalized


def enrich_document_metadata(document: ParsedDocument) -> dict[str, str]:
    """补齐 chunk 需要继承的文档级内部 metadata。"""

    source_path = Path(document.source)
    return {
        **document.metadata,
        "file_ext": document.metadata.get("file_ext") or source_path.suffix.lower(),
        "ingested_at": document.metadata.get("ingested_at") or current_ingested_at(),
    }


def current_ingested_at() -> str:
    """生成 UTC 入库时间，供后续增量更新和调试使用。"""

    return datetime.now(UTC).replace(microsecond=0).isoformat()
