from pathlib import Path
from datetime import UTC, datetime
from collections.abc import Callable

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
from fabagent_rag.documents import ParsedDocument, discover_supported_documents, parse_document
from fabagent_rag.embeddings import EmbeddingModel
from fabagent_rag.intent import Intent, detect_intent
from fabagent_rag.keyword_store import KeywordStore, extract_search_terms
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


def build_keyword_store(settings: Settings) -> KeywordStore:
    """根据配置创建关键词索引访问对象。"""

    return KeywordStore(settings.keyword_index_path)


DEFAULT_EMBEDDING_BATCH_SIZE = 10
ProgressCallback = Callable[[str, Path, int, int, str], None]


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


def ingest_directory(
    settings: Settings,
    directory: Path,
    batch_size: int,
    chunk_config: ChunkConfig | None = None,
    reset: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """批量入库目录下所有支持的文件。
    """

    files = discover_supported_documents(directory)
    errors: list[dict[str, str]] = []
    parsed_files = 0
    inserted = 0
    keyword_indexed = 0
    chunk_total = 0
    config = chunk_config or build_chunk_config(settings)

    if reset:
        reset_indexes(settings)

    if not files:
        return {
            "documents": 0,
            "chunks": 0,
            "inserted": 0,
            "keyword_indexed": 0,
            "scanned_files": 0,
            "parsed_files": 0,
            "failed_files": 0,
            "errors": [],
        }

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dimension)
    keyword_store = build_keyword_store(settings)

    for file_index, path in enumerate(files, start=1):
        report_progress(progress_callback, "解析", path, file_index, len(files))
        try:
            parsed_document = parse_document(path, str(path), mineru_backend=settings.mineru_backend)
        except Exception as exc:  # noqa: BLE001 - 批量入库不能因为单文件失败直接停掉
            error_message = summarize_error(exc)
            errors.append({"source": str(path), "error": error_message})
            report_progress(progress_callback, "失败", path, file_index, len(files), error_message)
            continue

        parsed_files += 1
        chunks = split_text(
            parsed_document.text,
            parsed_document.source,
            config,
            metadata=enrich_document_metadata(parsed_document),
        )
        report_progress(
            progress_callback,
            "解析完成",
            path,
            file_index,
            len(files),
            f"{len(parsed_document.text)} 字符",
        )
        report_progress(
            progress_callback,
            "切块完成",
            path,
            file_index,
            len(files),
            f"{len(chunks)} 个分块",
        )
        if not chunks:
            report_progress(progress_callback, "跳过", path, file_index, len(files), "空内容")
            continue

        chunk_total += len(chunks)
        batch_total = max(1, (len(chunks) + batch_size - 1) // batch_size)
        for batch_index, chunk_batch in enumerate(batch(chunks, batch_size), start=1):
            report_progress(
                progress_callback,
                "向量化",
                path,
                file_index,
                len(files),
                f"{batch_index}/{batch_total}",
            )
            embeddings = embedder.encode([chunk.text for chunk in chunk_batch])
            report_progress(
                progress_callback,
                "入库",
                path,
                file_index,
                len(files),
                f"{batch_index}/{batch_total}",
            )
            inserted += store.insert(chunk_batch, embeddings)

        report_progress(progress_callback, "索引完成", path, file_index, len(files))
        keyword_indexed += keyword_store.insert(chunks)
        report_progress(
            progress_callback,
            "完成",
            path,
            file_index,
            len(files),
            f"{len(chunks)} 个分块",
        )

    return {
        "documents": parsed_files,
        "chunks": chunk_total,
        "inserted": inserted,
        "keyword_indexed": keyword_indexed,
        "scanned_files": len(files),
        "parsed_files": parsed_files,
        "failed_files": len(errors),
        "errors": errors,
    }


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


def reset_indexes(settings: Settings) -> dict[str, bool]:
    """清空当前向量库和关键词索引，便于全量重建。"""

    store = MilvusStore(settings.milvus_host, settings.milvus_port, settings.milvus_collection, 1)
    keyword_store = KeywordStore(settings.keyword_index_path)
    return {
        "milvus": store.drop_collection(),
        "keyword": keyword_store.drop_index(),
    }


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
    keyword_store = build_keyword_store(settings)

    inserted = 0
    for chunk_batch in batch(chunks, batch_size):
        embeddings = embedder.encode([chunk.text for chunk in chunk_batch])
        inserted += store.insert(chunk_batch, embeddings)
    keyword_indexed = keyword_store.insert(chunks)

    return {
        "documents": document_count,
        "chunks": len(chunks),
        "inserted": inserted,
        "keyword_indexed": keyword_indexed,
    }


def report_progress(
    callback: ProgressCallback | None,
    stage: str,
    path: Path,
    current: int,
    total: int,
    detail: str = "",
) -> None:
    """把批量入库的状态交给 CLI 展示，核心流程本身不依赖具体输出格式。"""

    if callback:
        callback(stage, path, current, total, detail)


def summarize_error(exc: Exception) -> str:
    """把长异常压成一行，避免全量入库时把终端撑爆。"""

    message = str(exc).strip().splitlines()[-1] if str(exc).strip() else exc.__class__.__name__
    return message[:180]


def list_ingested_documents(settings: Settings) -> list[dict[str, object]]:
    """列出当前知识库中已经入库的 source。"""

    return build_keyword_store(settings).list_documents()


def answer_question(
    settings: Settings,
    question: str,
    top_k: int,
    source_filter: list[str] | None = None,
) -> dict[str, object]:
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
    contexts = search_contexts(settings, query_plan, top_k, intent, source_filter=source_filter)
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
    intent: Intent,
    source_filter: list[str] | None = None,
) -> list[dict[str, object]]:
    """使用 Query Plan 中的多个 query 做向量+BM25混合检索。"""

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dimension)
    keyword_store = build_keyword_store(settings)
    weights = hybrid_weights(settings, intent)
    queries = query_plan.queries()
    embeddings = embedder.encode(queries)
    normalized_source_filter = normalize_source_filter(source_filter)

    candidates: dict[tuple[object, object, object, object, object], dict[str, object]] = {}
    for query, embedding in zip(queries, embeddings, strict=True):
        for context in store.search(
            embedding,
            top_k=top_k * 2,
            source_filter=normalized_source_filter,
        ):
            merge_candidate(
                candidates,
                {
                    **context,
                    "matched_query": query,
                    "vector_score": normalize_vector_score(context.get("score")),
                },
            )

        for context in keyword_store.search(
            query,
            top_k=top_k * 2,
            source_filter=normalized_source_filter,
        ):
            merge_candidate(
                candidates,
                {
                    **context,
                    "matched_query": query,
                    "keyword_score": float(context.get("keyword_score") or 0.0),
                },
            )

    for context in candidates.values():
        context["metadata_boost"] = metadata_boost(context, queries)
        context["score"] = fused_score(context, weights)
        context["retrieval_mode"] = retrieval_mode(context)

    return sorted(candidates.values(), key=score_of, reverse=True)[:top_k]


def hybrid_weights(settings: Settings, intent: Intent) -> tuple[float, float]:
    """根据意图选择向量检索和 BM25 的融合权重。"""

    if intent == "summarize":
        return normalize_weights(
            settings.summarize_vector_weight,
            settings.summarize_keyword_weight,
        )
    return normalize_weights(settings.lookup_vector_weight, settings.lookup_keyword_weight)


def normalize_weights(vector_weight: float, keyword_weight: float) -> tuple[float, float]:
    total = vector_weight + keyword_weight
    if total <= 0:
        return 1.0, 0.0
    return vector_weight / total, keyword_weight / total


def normalize_source_filter(source_filter: list[str] | None) -> list[str]:
    """清理用户选择的 source，空列表表示不过滤。"""

    if not source_filter:
        return []
    return sorted({source.strip() for source in source_filter if source.strip()})


def merge_candidate(
    candidates: dict[tuple[object, object, object, object, object], dict[str, object]],
    context: dict[str, object],
) -> None:
    """合并向量检索和关键词检索返回的同一个 chunk。"""

    key = candidate_key(context)
    existing = candidates.get(key)
    if existing is None:
        candidates[key] = {
            **context,
            "vector_score": float(context.get("vector_score") or 0.0),
            "keyword_score": float(context.get("keyword_score") or 0.0),
            "matched_queries": [context.get("matched_query")],
        }
        return

    existing["vector_score"] = max(
        float(existing.get("vector_score") or 0.0),
        float(context.get("vector_score") or 0.0),
    )
    existing["keyword_score"] = max(
        float(existing.get("keyword_score") or 0.0),
        float(context.get("keyword_score") or 0.0),
    )
    if context.get("bm25_score") is not None:
        existing["bm25_score"] = context["bm25_score"]
    matched_queries = existing.setdefault("matched_queries", [])
    if isinstance(matched_queries, list) and context.get("matched_query") not in matched_queries:
        matched_queries.append(context.get("matched_query"))


def candidate_key(context: dict[str, object]) -> tuple[object, object, object, object, object]:
    """优先用 chunk_id 去重；没有 chunk_id 时退回来源位置和文本。"""

    chunk_id = context.get("chunk_id")
    if chunk_id:
        return (chunk_id, None, None, None, None)
    return (
        context.get("source"),
        context.get("page"),
        context.get("section_title"),
        context.get("text"),
        None,
    )


def normalize_vector_score(score: object) -> float:
    """把向量 IP 分数收敛到 0-1，便于和 BM25 rank 分数融合。"""

    if not isinstance(score, int | float):
        return 0.0
    return max(0.0, min(1.0, float(score)))


def fused_score(context: dict[str, object], weights: tuple[float, float]) -> float:
    vector_weight, keyword_weight = weights
    return (
        vector_weight * float(context.get("vector_score") or 0.0)
        + keyword_weight * float(context.get("keyword_score") or 0.0)
        + float(context.get("metadata_boost") or 0.0)
    )


def metadata_boost(context: dict[str, object], queries: list[str]) -> float:
    """用轻量 metadata 加权处理表格、标题和 sheet 命中。"""

    query_text = " ".join(queries)
    query_terms = set(extract_search_terms(query_text))
    if not query_terms:
        return 0.0

    boost = 0.0
    if context.get("content_type") == "table" and has_structured_term(query_terms):
        boost += 0.05

    location_terms = set(
        extract_search_terms(
            " ".join(
                [
                    str(context.get("section_title") or ""),
                    str(context.get("sheet_name") or ""),
                ]
            )
        )
    )
    if query_terms & location_terms:
        boost += 0.03
    return boost


def has_structured_term(terms: set[str]) -> bool:
    """判断 query 是否含有更适合关键词检索的编号、参数或代码。"""

    return any(any(char.isdigit() for char in term) or "-" in term or "_" in term for term in terms)


def retrieval_mode(context: dict[str, object]) -> str:
    has_vector = float(context.get("vector_score") or 0.0) > 0
    has_keyword = float(context.get("keyword_score") or 0.0) > 0
    if has_vector and has_keyword:
        return "hybrid"
    if has_keyword:
        return "keyword"
    return "vector"


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
